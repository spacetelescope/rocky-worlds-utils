#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module that contains the class to generate the HST spectrum quicklooks served
on the Rocky Worlds Website.

Authors
-------
- Mees Fix <<mfix@stsci.edu>>
"""

from astropy.io import fits
from bokeh.models import (
    ColumnDataSource,
)
from bokeh.plotting import figure, show
from bokeh.models import Whisker
from pathlib import Path

from rocky_worlds_utils.figure_utils.write_figure import write_figure


class rockyWorldsSpectrum:
    def __init__(self, filename, plot_height=600, plot_width=1400):
        """
        Class to build Hubble spectrum quicklooks for Rocky Worlds DDT.

        Parameters
        ----------
        filename : str
            Path of Rocky Worlds spectrum data product

        plot_height : int
            Size of bokeh plot height (default: 600)

        plot_width : int
            Size of bokeh plot width (default: 1400)
        """

        self.filename = Path(filename)
        self.hdu = fits.open(self.filename)
        self.wavelength = self.hdu[1].data["WAVELENGTH"].flatten()
        self.flux = self.hdu[1].data["FLUX"].flatten()
        self.error = self.hdu[1].data["FLUXERROR"].flatten()
        self.model_wavelength = self.hdu[2].data["WAVELENGTH"].flatten()
        self.model_flux = self.hdu[2].data["FLUX"].flatten()

        self.targetname = self.hdu[0].header["HLSPTARG"]

        self.plot_height = plot_height
        self.plot_width = plot_width

    def build_spectrum_plot(self, figure_out_path=None):
        """
        Make plot of COS/STIS spectrum

        Parameters
        ----------
        figure_out_path : str
            Path to save quick look spectrum plot to
        """

        source = ColumnDataSource(
            data=dict(
                wavelength=self.wavelength,
                flux=self.flux,
                error=self.error,
                upp_err=self.flux + self.error,
                lwr_err=self.flux - self.error,
            )
        )

        tooltips = [
            ("Wavelength", "@wavelength{0.000}"),
            ("Flux", "@flux"),
            ("Flux Error +/-", "@error"),
        ]

        p = figure(
            width=self.plot_width,
            height=self.plot_height,
            tooltips=tooltips,
        )

        p.x_range.start = min(self.wavelength) - 2
        p.x_range.end = max(self.wavelength) + 2

        p.axis.axis_label_text_font_style = "bold"

        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"

        p.xaxis.major_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_size = "15pt"

        p.xaxis.axis_label = "Wavelength (Å)"
        p.yaxis.axis_label = "Flux Density (erg s⁻¹ cm⁻² Å⁻¹)"

        plot_title = f"Target: {self.targetname}"
        p.title.text = plot_title
        p.title.text_font_size = "25pt"

        p.line(
            self.model_wavelength,
            self.model_flux,
            color="red",
            line_width=2,
            legend_label="Lyman α Model",
        )

        scatter = p.scatter(
            x="wavelength",
            y="flux",
            source=source,
            color="black",
            alpha=0.25,
            size=5,
            legend_label="Flux",
        )

        p.hover.renderers = [scatter]

        error = Whisker(
            base="wavelength",
            upper="upp_err",
            lower="lwr_err",
            source=source,
            level="annotation",
            line_width=1,
            line_alpha=0.25,
        )

        error.upper_head.size = 20
        error.lower_head.size = 20
        error.upper_head.line_alpha = 0.3
        error.lower_head.line_alpha = 0.3

        p.add_layout(error)

        if figure_out_path:
            filename = self.filename.name.replace("fits", "html")
            full_file_path = Path(figure_out_path) / filename
            write_figure(p, full_file_path)
        else:
            show(p)
