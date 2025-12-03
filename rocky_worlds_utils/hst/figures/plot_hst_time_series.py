#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module that contains the class to generate the HST time series quicklooks served
on the Rocky Worlds Website.

Authors
-------
- Mees Fix <<mfix@stsci.edu>>
"""

from astropy.io import ascii, fits
from bokeh.models import ColumnDataSource, HoverTool, Range1d
from bokeh.plotting import figure, show
from pathlib import Path

from rocky_worlds_utils.figure_utils.write_figure import write_figure


class rockyWorldsTimeSeries:
    def __init__(self, data, model, plot_height=600, plot_width=1400):
        """
        Class to build Hubble time series quicklooks for Rocky Worlds DDT.
        """

        self.timeseries = ascii.read(data)
        self.model = ascii.read(model)

        self.timeseries_time = self.timeseries["time"]
        self.timeseries_flux = self.timeseries["flux"]
        self.timeseries_error = self.timeseries["error"]

        self.model_time = self.model["time"]
        self.model_flux = self.model["flux"]

        self.targetname = "TEMPNAME"

        self.plot_height = plot_height
        self.plot_width = plot_width

    def build_timeseries_plot(self, figure_out_path=None):
        """
        Make plot of timeseries data.

        Parameters
        ----------
        figure_out_path : str
            Path to save quick look spectrum plot to
        """

        data_source = ColumnDataSource(
            data=dict(
                datatime=self.timeseries_time,
                dataflux=self.timeseries_flux,
                dataerror=self.timeseries_error,
            )
        )

        model_source = ColumnDataSource(
            data=dict(modeltime=self.model_time, modelflux=self.model_flux)
        )

        p = figure(
            width=self.plot_width,
            height=self.plot_height,
        )

        p.x_range = Range1d(
            min(self.timeseries_time) - 200, max(self.timeseries_time) + 200
        )

        p.axis.axis_label_text_font_style = "bold"

        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"

        p.xaxis.major_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_size = "15pt"

        p.xaxis.axis_label = "Time From Exposure Start (s)"
        p.yaxis.axis_label = "Flux (erg cm⁻² s⁻¹)"

        plot_title = f"Target: {self.targetname}"
        p.title.text = plot_title
        p.title.text_font_size = "25pt"

        data_line = p.line(
            x="datatime",
            y="dataflux",
            source=data_source,
            color="black",
            line_width=2,
        )
        p.add_tools(
            HoverTool(
                renderers=[data_line],
                tooltips=[
                    ("Time", "@datatime"),
                    ("Flux", "@dataflux"),
                    ("Error", "@dataerror"),
                ],
                mode="vline",
            )
        )

        model_line = p.line(
            x="modeltime",
            y="modelflux",
            source=model_source,
            color="red",
            line_width=1,
        )

        p.add_tools(
            HoverTool(
                renderers=[model_line],
                tooltips=[("Model Flux", "@modelflux")],
                mode="vline",
            )
        )

        if figure_out_path:
            filename = f"{self.targetname}_timeseries_ql.html"
            full_file_path = Path(figure_out_path) / filename
            write_figure(p, full_file_path)
        else:
            show(p)
