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
        self.error = self.hdu[1].data["ERROR"].flatten()

        self.targetname = self.hdu[0].header["TARGNAME"]

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
            )
        )

        p = figure(
            width=self.plot_width,
            height=self.plot_height,
            tooltips=[
                ("wavelength", "@wavelength{0.000}"),
                ("Flux", "@flux"),
                ("Flux Error", "@error"),
            ],
        )

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
            x="wavelength",
            y="flux",
            source=source,
            color="black",
            line_width=2,
        )

        if figure_out_path:
            filename = f"{self.targetname}_spectrum_ql.html"
            full_file_path = Path(figure_out_path) / filename
            write_figure(p, full_file_path)
        else:
            show(p)
