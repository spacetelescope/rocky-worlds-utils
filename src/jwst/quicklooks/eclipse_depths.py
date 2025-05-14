#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module that contains the class to generate the eclipse depth output served via
the Rocky Worlds website.

Authors
-------
- Mees Fix <<mfix@stsci.edu>>
"""

from astropy.io import ascii
from astropy.table import unique
from bokeh.models import (
    Band,
    ColumnDataSource,
    Legend,
    Span,
    Whisker,
)
from bokeh.plotting import figure, show, output_file, save
import numpy as np
from pathlib import Path


def plot_eclipse_depths(
    eclipse_table_path, figure_out_path=None, plot_width=1600, plot_height=600
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

    figure_out_path : str
        Path to write out bokeh html file

    plot_width : int
        Plot width for bokeh figure

    plot_height : int
        Plot width for bokeh figure
    """

    eclipse_data = ascii.read(
        eclipse_table_path, format="fixed_width", header_rows=["name", "dtype"]
    )
    unique_planet_names = unique(eclipse_data, keys="planet_name")["planet_name"]
    for planet_name in unique_planet_names:
        # Find table rows for specific planet name
        planet_name_indices = np.where(eclipse_data["planet_name"] == planet_name)[0]
        planet_data = eclipse_data[planet_name_indices]

        eclipse_depths_mean = np.mean(planet_data["eclipse_depth"])
        eclipse_depths_std = np.std(planet_data["eclipse_depth"])

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
            eclipse_numbers = np.arange(1, len(planet_data["eclipse_depth"]) + 1)
            x_tick_labels = [f"Eclipse {val}" for val in eclipse_numbers]
        source = ColumnDataSource(
            data=dict(
                filename=planet_data["filename"],
                propid=planet_data["propid"],
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
            x_range=(0.5, max(eclipse_numbers) + 0.5),
            y_range=(min(lower_err - 200), max(upper_err + 200)),
            tooltips=[
                ("Filename", "@filename"),
                ("Proposal ID", "@propid"),
                ("Time [BJD]", "@time{0.0000}"),
                ("Eclipse Depth [ppm]", "@eclipse_depths{0.0000}"),
                ("Eclipse Depth Error [ppm]", "@err{0.0000}"),
            ],
        )

        # Define parameters of plot
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

        # Scatter eclipse depths
        scatter = p.scatter(
            x="eclipse_numbers",
            y="eclipse_depths",
            color="black",
            size=8,
            line_alpha=0,
            source=source,
            legend_label="Eclipse Depth",
        )

        # Only allow hover for scatter points
        p.hover.renderers = [scatter]

        # Create Error Bars for scatter
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

        # Plot dashed line for mean value of dataset
        hline = Span(
            location=eclipse_depths_mean,
            dimension="width",
            line_color="red",
            line_width=3,
            line_dash="dashed",
        )

        # Add a line glyph with minimal data to represent the Span in the legend
        r_line = p.line(
            [0],
            [0],
            legend_label=f"Mean Eclipse Depth = {eclipse_depths_mean:.2f} ppm",
            line_dash="dashed",
            line_color="red",
            line_width=3,
        )
        r_line.visible = False  # Set this fake line to invisible

        # Create banded areas for standard deviation
        std_upper = Span(
            location=eclipse_depths_mean + eclipse_depths_std,
            dimension="width",
            line_color="red",
            line_width=2,
        )

        std_lower = Span(
            location=eclipse_depths_mean - eclipse_depths_std,
            dimension="width",
            line_color="red",
            line_width=2,
        )

        y1 = np.tile(eclipse_depths_mean + eclipse_depths_std, len(eclipse_numbers))
        y2 = np.tile(eclipse_depths_mean - eclipse_depths_std, len(eclipse_numbers))

        p.varea(
            x=eclipse_numbers,
            y1=y1,
            y2=y2,
            color="red",
            alpha=0.25,
            legend_label="Error Band",
        )

        # Extend the statistical lines in x infinitely.
        p.renderers.extend([std_lower, std_upper, hline])

        if figure_out_path:
            filename = f"{planet_name}_eclipse_depths.html"
            full_file_path = Path(figure_out_path) / filename
            output_file(full_file_path)
            save(p)
        else:
            show(p)
