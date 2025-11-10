"""Script to generate plot of interactive visibility windows via plotly.

Authors
-------
Tyler Baines
Mees B. Fix
"""

from collections import defaultdict
from datetime import datetime, timedelta
import os

from astropy.time import Time
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px

from jwst_gtvt.jwst_tvt import Ephemeris
from jwst_gtvt.display_results import get_visibility_windows

RWDDT_TARGETS = [
    ("GJ 3929 b", ("239.57833", "35.40675")),
    ("LTT 1445 A c", ("45.46250", "-16.59334")),
    ("LHS 1140 b", ("11.247", "-15.271")),
    ("LTT 1445 A b", ("45.46250", "-16.59334")),
    ("TOI-198 b", ("2.2715", "27.1217")),
    ("TOI-406 c", ("49.2626", "-42.2441")),
    ("TOI-771 b", ("164.1138", "-72.98517")),
    ("HD 260655 c", ("99.308", "18.727")),
    ("TOI-244 b", ("10.57058", "-36.71817")),
]


def build_visibility_window_dataset(
    date_window=("2025-01-01", "2028-01-01"), outfile_path=None
):
    """Build CSV file used by rwddtTargetVisibilityWindows. Using the JWST GTVT
    to obtain the visibility windows automatically for our targets with JWST.

    Parameters
    ----------
    date_window : tuple
        Date range to get visibilities (start_date, end_date) format YYYY-MM-DD.
    outfile_path : str
        Directory where dataset will be saved.
    """

    def _convert_date_series(df):
        """Convert MJD column into YYYY-MM-DD and add new column.

        Parameters
        ----------
        df : pandas.dataFrame
            Dataframe generated from results of calculating visibility windows.
        """
        mjd_epoch = datetime(1858, 11, 17, 0, 0, 0)

        # Convert MJD to datetime objects
        for date in ["Start", "End"]:
            df[date] = df[date].apply(lambda x: mjd_epoch + timedelta(days=x))
            df[date] = df[date].dt.strftime("%Y-%m-%d")

        return df

    target_windows_dfs = []
    for target_data in RWDDT_TARGETS:
        target_name = target_data[0]
        ra, dec = target_data[1]

        # Build visibility dataframe
        if date_window:
            start_date = Time(date_window[0])
            end_date = Time(date_window[1])
            target_eph = Ephemeris(start_date, end_date)
        else:
            target_eph = Ephemeris()
        target_df = target_eph.get_fixed_target_positions(ra, dec)

        # Only select position angles that are in field of regard
        target_df = target_df.loc[target_df["in_FOR"]]

        # Find continuous viewing windows
        idx = target_df.index.tolist()
        visibile_idx = get_visibility_windows(idx)

        # Build dataset, rows look like
        # target name, window start date, window end date
        target_dict = defaultdict(list)
        for window_start, window_end in visibile_idx:
            target_dict["Target"].append(target_name)
            target_dict["Start"].append(target_df.loc[window_start]["MJD"])
            target_dict["End"].append(target_df.loc[window_end]["MJD"])
        target_windows_df = pd.DataFrame(target_dict)
        target_windows_dfs.append(target_windows_df)

    # Collapse all separate data frames into one.
    combined_window_df = pd.concat(target_windows_dfs).reset_index(drop=True)
    combined_window_df = _convert_date_series(combined_window_df)

    # Write data set of just display it to screen
    if outfile_path:
        # Set run date by day, visibility won't change at smaller than day scales.
        now = datetime.now().strftime("%Y-%m-%d")
        filename = f"jwst_visibility_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_run_date_{now}.csv"
        outfile_full_path = os.path.join(outfile_path, filename)
        combined_window_df.to_csv(outfile_full_path, index=False)
    else:
        print(combined_window_df)


class RwddtJwstTargetVisibilityWindows:
    def __init__(self, vis_window_data):
        """Build the colormap and figure displayed on RWDDT website.

        Parameters
        ----------
        vis_window_data : str
            Absolute path to csv file
        """
        self.filename = os.path.basename(vis_window_data)
        self.vis_window_df = pd.read_csv(vis_window_data, index_col=False)
        self.unique_targets = self.vis_window_df["Target"].unique()
        self.generate_color_map()

        self.colors_primary = {
            "Cerulean": "#29a6df",
            "Scarlet": "#d12717",
            "Nightlife": "#272f4f",
        }
        self.colors_auxiliary = {
            "Oasis": "#286d97",
            "Starry Night": "#2b3e60",
            "Vermillion": "#dc4014",
            "Snow": "#f8f7f7",
            "Zinc": "#8d898c",
        }
        self.font_title = dict(
            family="Quicksilver, Arial",
            color=self.colors_primary["Nightlife"],
            size=30,
            weight="bold",
        )

    def generate_color_map(self):
        """Generate Rocky Worlds DDT color map for visiblity plot"""

        def _rgb_to_hex(rgb_str):
            """Convert 'rgb(R, G, B)' or 'rgba(R, G, B, A)' to '#rrggbb'."""
            s = rgb_str.strip()
            if s.startswith("rgb"):
                nums = s[s.find("(") + 1 : s.find(")")].split(",")
                r, g, b = [int(float(x)) for x in nums[:3]]
                return "#{:02x}{:02x}{:02x}".format(r, g, b)
            return s  # assume it's already hex

        # Rocky Worlds sequential colorscale (positions)
        rockyworlds_colorscale = [
            [0.0, "#d12717"],  # Nightlife
            [0.5, "#29a6df"],  # Cerulean
            [1.0, "#272f4f"],  # Scarlet
        ]

        # sample evenly across the colorscale
        vals = np.linspace(0.0, 1.0, len(self.unique_targets))
        sampled_rgb = [pc.sample_colorscale(rockyworlds_colorscale, v)[0] for v in vals]
        sampled_hex = [_rgb_to_hex(s) for s in sampled_rgb]

        # build map target -> hex color
        self.target_color_map = dict(zip(self.unique_targets, sampled_hex))

    def plot_visibility_windows(self, outfile_path=None):
        """Generate visibility plot with plotly

        Parameters
        ----------
        outfile_path : str
            Directory to write figure to. Figure name will be input csv filename
            with .html instead of .csv extension. If not provided, figure displays
            in default web browser.
        """
        fig_gantt = px.timeline(
            self.vis_window_df,
            x_start="Start",
            x_end="End",
            y="Target",
            color="Target",
            color_discrete_map=self.target_color_map,
            title="JWST Visibility Windows — Rocky Worlds",
            height=600,
        )

        fig_gantt.update_layout(
            plot_bgcolor="#e2e1e1",  # colors_auxiliary["Snow"],  # Snow
            paper_bgcolor=self.colors_auxiliary["Snow"],
            title_font=self.font_title,  # dict(family="Quicksilver, Arial", size=20),
            font=dict(family="Montserrat, Arial", size=20),
            hoverlabel=dict(
                font=dict(
                    family="Montserrat, Arial",
                    size=20,
                )
            ),
            legend_title_text="",  # optional: cleaner legend
            margin=dict(t=80, b=60, l=160, r=40),
            xaxis_title="Visibility Windows",
            yaxis_title="Targets",
        )

        # Make sure y ordering is sensible (optional)
        fig_gantt.update_yaxes(
            categoryorder="array", categoryarray=self.unique_targets[::-1]
        )  # reversed for top-first

        if outfile_path:
            outfile_name = self.filename.replace(".csv", ".html")
            outfile_full_path = os.path.join(outfile_path, outfile_name)
            fig_gantt.write_html(outfile_full_path)
        else:
            fig_gantt.show()
