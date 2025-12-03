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
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, show
import numpy as np
import pandas as pd

from jwst_gtvt.jwst_tvt import Ephemeris
from jwst_gtvt.display_results import get_visibility_windows

from rocky_worlds_utils.figure_utils.write_figure import write_figure

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
        self.unique_targets = np.flip(self.vis_window_df["Target"].unique())

        self.vis_window_df["Start"] = pd.to_datetime(self.vis_window_df["Start"])
        self.vis_window_df["End"] = pd.to_datetime(self.vis_window_df["End"])

        self.assign_color_palette()

    def assign_color_palette(self):
        """Create column for dataframe."""
        color_pallete = {
            "GJ 3929 b": "#d12717",
            "LTT 1445 A c": "#a74749",
            "LHS 1140 b": "#7d667b",
            "LTT 1445 A b": "#5386ad",
            "TOI-198 b": "#29a6df",
            "TOI-406 c": "#2988bb",
            "TOI-771 b": "#286a97",
            "HD 260655 c": "#284d73",
            "TOI-244 b": "#272f4f",
        }

        self.vis_window_df["Color"] = [
            color_pallete[target] for target in self.vis_window_df["Target"]
        ]

    def plot_visibility_windows(self, outfile_path):
        """Generate visibility plot with bokeh

        Parameters
        ----------
        outfile_path : str
            Directory to write figure to. Figure name will be input csv filename
            with .html instead of .csv extension. If not provided, figure displays
            in default web browser.
        """
        source = ColumnDataSource(self.vis_window_df)

        p = figure(
            width=1650,
            height=900,
            x_axis_type="datetime",
            y_range=self.unique_targets,
            title="JWST Visibility Windows -- Rocky Worlds",
            tools="box_zoom,reset",
        )

        p.hbar(
            y="Target",
            left="Start",
            right="End",
            line_color="Color",
            fill_color="Color",
            height=0.75,
            source=source,
        )

        hover_tooltips = """
                        <div style="font-size: 16pt; font-family:Montserrat;">
                            <b>Target:</b> @Target <br>
                            <b>Start Date:</b> @Start{%Y-%m-%d} <br>
                            <b>End Date:</b> @End{%Y-%m-%d}
                        </div>
                    """

        p.add_tools(
            HoverTool(
                tooltips=hover_tooltips,
                formatters={"@Start": "datetime", "@End": "datetime"},
            )
        )

        # Plot title.
        p.title.text_font_size = "30pt"

        # x axis settings
        p.xaxis.axis_label = "Visibility Windows"
        p.xaxis.axis_label_text_font_size = "25pt"
        p.xaxis.axis_label_text_font_style = "bold"
        p.xaxis.major_label_text_font_size = "15pt"
        p.xaxis.major_label_text_font_style = "bold"

        # y axis settings
        p.yaxis.axis_label = "Targets"
        p.yaxis.axis_label_text_font_size = "25pt"
        p.yaxis.axis_label_text_font_style = "bold"
        p.yaxis.major_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_style = "bold"
        p.y_range.range_padding = 0.1
        p.ygrid.grid_line_color = None

        if outfile_path:
            filename = self.filename.replace(".csv", ".html")
            abs_file_path = os.path.join(outfile_path, filename)
            write_figure(p, abs_file_path)
        else:
            show(p)
