"""Module that contains the class to generate the eclipse depth output served via
the Rocky Worlds website.

Authors
-------
- Mees Fix

"""

from astropy.io import ascii
from astropy.table import Table, unique
from bokeh.models import (
    Band,
    ColumnDataSource,
    FixedTicker,
    Span,
    Whisker,
)

from bokeh.plotting import figure, show, output_file, save
import numpy as np


def plot_eclipse_depths(
    eclipse_table_path, figure_output=None, plot_width=1600, plot_height=600
):
    """Generate bokeh figure for eclipse depths. Plot is of eclipse depth
    values as a function of eclipse number. The eclipse table that these plots
    are generated from can be genreated if one does not exist. See `rocky_worlds_utils.eclipse_utils.pull_eclipse_metadata.py`

    Parameters
    ----------
    eclipse_depths_data : astropy.Table.table
        An astropy table containing all of the metadata associated with eclipse
        depths for the RWDDT.
        See: `rocky_worlds_utils.eclipse_utils.pull_eclipse_metadata.eclipseDepthTable`
    """

    eclipse_data = eclipse_data = ascii.read(
        eclipse_table_path, format="fixed_width", header_rows=["name", "dtype"]
    )
    unique_planet_names = unique(eclipse_data, keys="planet_name")["planet_name"]
    for planet_name in unique_planet_names:
        # Find table rows for specific planet name
        planet_name_indices = np.where(eclipse_data["planet_name"] == planet_name)[0]
        planet_data = eclipse_data[planet_name_indices]

        eclipse_depths_mean = np.mean(planet_data["eclipse_depth"])

        upper_err = (
            planet_data["eclipse_depth"] + planet_data["eclipse_depth_upper_err"]
        )
        lower_err = (
            planet_data["eclipse_depth"] - planet_data["eclipse_depth_lower_err"]
        )

        if len(planet_data["eclipse_depth"]) == 1:
            eclipse_numbers = np.array([1])
            x_tick_labels = ["Eclipse 1"]
        else:
            eclipse_numbers = np.arange(1, len(planet_data["eclipse_depth"]))
            x_tick_labels = [f"Eclipse {val}" for val in eclipse_numbers]
        source = ColumnDataSource(
            data=dict(
                time=planet_data["eclipse_time"],
                eclipse_numbers=eclipse_numbers,
                eclipse_depths=planet_data["eclipse_depth"],
                err=planet_data["eclipse_depth_err"],
                upp_err=upper_err,
                lwr_err=lower_err,
            )
        )

        p = figure(
            width=plot_width,
            height=plot_height,
            x_range=(0.5, max(eclipse_numbers) + 1),
            y_range=(min(lower_err - 200), max(upper_err + 200)),
            tooltips=[
                ("Time", "@time"),
                ("Eclipse Depth", "@eclipse_depths"),
                ("Eclipse Depth Error", "@err"),
            ],
        )

        custom_labels = {
            ecl_num: label for (ecl_num, label) in zip(eclipse_numbers, x_tick_labels)
        }
        p.xaxis.ticker = eclipse_numbers
        p.xaxis.major_label_overrides = custom_labels

        p.axis.axis_label_text_font_style = "bold"

        p.xaxis.minor_tick_line_color = None
        p.yaxis.minor_tick_line_color = None

        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"

        p.xaxis.major_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_size = "15pt"

        p.yaxis.axis_label = "Eclipse Depth [ppm]"

        plot_title = f"Eclipse Depths for Planet: {planet_name}"
        p.title.text = plot_title
        p.title.text_font_size = "25pt"

        p.scatter(
            x="eclipse_numbers",
            y="eclipse_depths",
            color="black",
            size=8,
            line_alpha=0,
            source=source,
            legend_label="Eclipse Depth",
        )

        error = Whisker(
            base="eclipse_numbers",
            upper="upp_err",
            lower="lwr_err",
            source=source,
            level="annotation",
            line_width=1,
        )

        error.upper_head.size = 20
        error.lower_head.size = 20
        p.add_layout(error)

        hline = Span(
            location=eclipse_depths_mean,
            dimension="width",
            line_color="red",
            line_width=3,
            line_dash="dashed",
        )

        p.renderers.extend([hline])

        # Add a line glyph with minimal data to represent the Span in the legend
        r_line = p.line([0], [0], legend_label='Mean Eclipse Depth', line_dash='dashed',
                        line_color="red", line_width=3)
        r_line.visible = False  # Set this fake line to invisible

        # band = Band(
        #     base=eclipse_numbers,
        #     lower=eclipse_depths_mean - 20,
        #     upper=eclipse_depths_mean + 20,
        #     source=source,
        #     fill_alpha=0.3,
        #     fill_color="red",
        #     line_color="red",
        # )
        # p.add_layout(band)

        show(p)
