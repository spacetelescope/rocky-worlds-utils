#!/usr/bin/env python3

import os
import sys
import stat
import time
import argparse
import requests
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from astroquery.mast import Observations


MAX_RETRIES = 3

# Auto-disable colors if stdout is redirected
USE_COLOR = sys.stdout.isatty()


def color(code):
    return code if USE_COLOR else ""


# ANSI color codes (conditionally applied)
RESET = color("\033[0m")
GREEN = color("\033[92m")
RED = color("\033[91m")
YELLOW = color("\033[93m")
CYAN = color("\033[96m")

TAG_WIDTH = 13  # Space reserved for tags like "[DOWNLOADING]"


@contextmanager
def temporary_umask(new_umask):
    """
    Context manager to temporarily change the umask.

    Parameters
    ----------
    new_umask : int
        The umask value to set temporarily.
    """
    old_umask = os.umask(new_umask)
    try:
        yield
    finally:
        os.umask(old_umask)


def make_shared_dir(foldername, output_dir, exist_ok=True, dry_run=False, mode=0o2750):
    """
    Create a directory with specific permissions for owner and group.

    Common modes:
    - 0o2750: setgid, owner=rwx, group=rx, others=---
    - 0o2770: setgid, owner=rwx, group=rwx, others=---

    Parameters
    ----------
    foldername : str
        Directory name relative to output_dir.
    output_dir : str
        The base directory path to create the folder inside.
    exist_ok : bool
        Whether to suppress error if directory already exists.
    dry_run : bool
        If True, directory is not actually created.
    mode : int, optional
        Octal permission mode to use when creating the directory.
        Defaults to 0o2750 (owner rwx, group rx, setgid).

    Returns
    -------
    str
        Full path to the created directory.
    """
    fullpath = os.path.join(output_dir, foldername)

    # Decide umask dynamically based on whether group-write is intended
    if mode & 0o0020:  # Check group write bit
        desired_umask = 0o0007  # Allow group write
    else:
        desired_umask = 0o0027  # Block group write, all others

    if not os.path.exists(fullpath):
        if dry_run:
            perm_str = stat.filemode(stat.S_IFDIR | mode)
            log("[DRY-RUN]", YELLOW, f"Would create: {fullpath} with permissions {perm_str}")
        else:
            with temporary_umask(desired_umask):
                os.makedirs(fullpath, mode=mode, exist_ok=exist_ok)

    return fullpath


def log(tag, color_code, message):
    """
    Print a color-coded message with a prefixed tag.

    Parameters
    ----------
    tag : str
        Label to prepend, like "[INFO]".
    color_code : str
        ANSI color code.
    message : str
        The message to print.
    """
    padded_tag = f"{color_code}{tag:<{TAG_WIDTH}}{RESET}"
    print(f"{padded_tag} {message}", flush=True)


def query_MAST(proposal_id, observation_id, visit_id, subgroup='UNCAL'):
    """
    Query MAST for data products matching given visit/subgroup.

    Parameters
    ----------
    proposal_id : str or int
        JWST proposal number.
    observation_id : str or int
        JWST observation number.
    visit_id : str or int
        JWST visit number.
    subgroup : str
        Product subgroup to retrieve (e.g., 'UNCAL').

    Returns
    -------
    astropy.table.Table
        Filtered product list.
    """
    # Normalize ID formats
    proposal_id = str(proposal_id).zfill(5)
    observation_id = str(observation_id).zfill(3)
    visit_id = str(visit_id).zfill(3)

    # Determine calibration level based on subgroup
    if subgroup in ['UNCAL', 'GS-ACQ1', 'GS-ACQ2', 'GS-FG', 'GS-ID', 'GS-TRACK']:
        calib_level = [1]
    elif subgroup in ['CAL', 'CALINTS', 'RATE', 'RATEINTS', 'ANNNN_CRFINTS',
                      'GS-ACQ1', 'GS-ACQ2', 'GS-FG', 'GS-ID', 'GS-TRACK', 'RAMP']:
        calib_level = [2]
    elif subgroup in ['X1DINTS', 'WHTLT']:
        calib_level = [3]
    else:
        raise ValueError(f"Unknown subgroup: {subgroup}")

    obs_id = f'jw{proposal_id}{observation_id}{visit_id}_03101*'

    sci_table = Observations.query_criteria(proposal_id=proposal_id,
                                            obs_id=obs_id)
    if len(sci_table) == 0:
        raise ValueError(f"No data found for proposal {proposal_id}, "
                         f"observation {observation_id}, visit {visit_id}")

    data_products = Observations.get_product_list(sci_table)
    table = Observations.filter_products(data_products,
                                         productSubGroupDescription=subgroup,
                                         calib_level=calib_level)

    if len(table) == 0:
        raise ValueError(f"No data found for subgroup {subgroup}, "
                         f"calib_level {calib_level}")

    if 'dataURI' not in table.colnames:
        raise ValueError("No dataURI column found. No downloadable files.")

    table.sort('dataURI')
    return table


def download_files(proposal_id, obs_id, visit_id, download_dir,
                   subgroup='UNCAL', flat=True, verbose=False,
                   max_retries=MAX_RETRIES, dry_run=False):
    """
    Download files from MAST with retries and dry-run option.

    Parameters
    ----------
    proposal_id : int
        JWST proposal ID.
    obs_id : int
        Observation ID.
    visit_id : int
        Visit ID.
    download_dir : str
        Path to download directory.
    subgroup : str
        File type, e.g., 'UNCAL', 'RATEINTS'.
    flat : bool
        If True, do not preserve folder structure.
    verbose : bool
        Print detailed logs if True.
    max_retries : int
        Number of retry attempts for failed downloads.
    dry_run : bool
        If True, skip actual download.

    Returns
    -------
    bool
        True if all files succeeded, False otherwise.
    """
    if verbose:
        log("[STARTING]", CYAN,
            f"Downloading {subgroup} files for proposal {proposal_id}, "
            f"observation {obs_id}, visit {visit_id}")

    table = query_MAST(proposal_id, obs_id, visit_id, subgroup=subgroup)

    if verbose:
        log("[INFO]", CYAN, f"Scheduling {len(table)} files for download")

    all_success = True

    for product in table:
        filename = os.path.basename(product["productFilename"])

        if dry_run:
            log("[DRY-RUN]", YELLOW,
                f"Would download: {filename} → {download_dir}")
            continue

        # Retry logic with exponential backoff
        for attempt in range(1, max_retries + 1):
            try:
                manifest = Observations.download_products(
                    product, download_dir=download_dir, flat=flat)
                entry = manifest[0]
                if (entry["Message"] and
                        entry["Message"].startswith("HTTPError")):
                    Observations.login()
                    manifest = Observations.download_products(
                        product, download_dir=download_dir, flat=flat)
                    entry = manifest[0]
                if entry["Status"] == "COMPLETE":
                    break  # success
                else:
                    raise Exception(f"Incomplete status: {entry['Status']}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                wait_time = 2 ** (attempt - 1)
                log("[RETRY]", YELLOW,
                    f"{filename} failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(wait_time)
                else:
                    all_success = False
                    log("[FAILED]", RED, f"{filename}")
                    break

    return all_success


def main():
    """
    Entry point to parse arguments and orchestrate directory setup and downloads.
    """
    parser = argparse.ArgumentParser(
        description="Download JWST _uncal and _rateints files from MAST "
                    "for a given proposal/visit. "
                    "Creates organized, group-writable directories.")

    parser.add_argument("planet_name", type=str,
                        help="Target planet name, e.g., GJ3929b")
    parser.add_argument("proposal_id", type=int,
                        help="JWST proposal number (e.g., 9235)")
    parser.add_argument("observation_ids", type=str,
                        help="Comma-separated observation IDs (e.g., 1 or 4,5)")
    parser.add_argument("visit_id", type=int,
                        help="Visit ID to label output directories")
    parser.add_argument("--base_dir", default="../JWST/",
                        help="Root directory under which data is organized")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show actions without performing them")
    args = parser.parse_args()

    planet_name = args.planet_name.strip()
    proposal_id = args.proposal_id
    observation_ids = [int(x.strip())
                       for x in args.observation_ids.split(",")]
    visit_id = args.visit_id
    base_dir = args.base_dir
    verbose = not args.quiet
    dry_run = args.dry_run

    # Fetch visit metadata XML
    url = f"https://www.stsci.edu/jwst-program-info/visits/?program={proposal_id}&download="
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        log("[ERROR]", RED, f"Failed to fetch program XML: {e}")
        sys.exit(1)
    root = ET.fromstring(response.content)

    output_dir = os.path.join(base_dir, planet_name, f"visit{visit_id}")
    make_shared_dir(output_dir, '/', dry_run=dry_run, mode=0o2770)

    # Directory structure
    uncalibrated_dir = make_shared_dir('Uncalibrated', output_dir, dry_run=dry_run)
    stage1_dir = make_shared_dir('MAST_Stage1',
                                 output_dir, dry_run=dry_run)

    all_downloads_succeeded = True
    for obs_id in observation_ids:
        visit = root.find(f".//visit[@observation='{obs_id}']")
        if visit is None:
            raise ValueError(f"Visit with observation={obs_id} not found")

        visit_id = int(visit.attrib['visit'])
        status = visit.findtext("status", default="N/A")
        if status != "Executed":
            raise ValueError(f"Observation {obs_id} is not 'Executed'")

        log("[INFO]", CYAN,
            f"Processing observation {obs_id} into {output_dir}")

        # Download RATEINTS first (smaller)
        rateints_success = download_files(proposal_id, obs_id, visit_id,
                                          stage1_dir, 'RATEINTS',
                                          verbose=verbose, dry_run=dry_run)

        # Download UNCAL files next
        uncal_success = download_files(proposal_id, obs_id, visit_id,
                                       uncalibrated_dir, 'UNCAL',
                                       verbose=verbose, dry_run=dry_run)

        if rateints_success and uncal_success:
            log("[SUCCESS]", GREEN,
                f"All files downloaded for observation {obs_id}")
        else:
            log("[WARNING]", RED,
                f"Some downloads failed for observation {obs_id}")
            all_downloads_succeeded = False

    if all_downloads_succeeded:
        log("[DONE]", GREEN, "All downloads complete.")
    else:
        log("[DONE]", RED, "Some downloads failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
