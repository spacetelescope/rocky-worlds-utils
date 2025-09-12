#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Module that contains the class to generate the lightcurve quicklooks served via
the Rocky Worlds website.

Authors
-------
- Mees Fix <<mfix@stsci.edu>>
"""

from bokeh.layouts import column
from bokeh.models import (
    TabPanel,
    Tabs,
    ColumnDataSource,
    CustomJS,
    Styles,
    Slider,
    Whisker,
)
from bokeh.plotting import figure, show
from copy import deepcopy
import numpy as np
from pathlib import Path
import xarray as xr


from rocky_worlds_utils.figure_utils.write_figure import write_figure


class rockyWorldsLightCurve:
    def __init__(self, filename, plot_height=600, plot_width=1400):
        """
        Class to build light curve quicklooks for Rocky Worlds DDT.

        Parameters
        ----------
        filename : str
            Path of Rocky Worlds light curve data product file (extension `lc.h5`)

        plot_height : int
            Size of bokeh plot height (default: 600)

        plot_width : int
            Size of bokeh plot width (default: 1400)
        """
        self.filename = Path(filename)
        self.plot_width = plot_width
        self.plot_height = plot_height

        self.data = xr.load_dataset(filename)

        # Data Attributes
        self.time = self.data.time.data
        self.raw_flux = self.data.rawFlux[0].data
        self.flux_err = self.data.rawFluxErr[0].data
        self.cleaned_flux = self.data.cleanedFlux[0].data
        self.full_model = self.data.fullModel[0].data
        self.astro_model = self.data.astroModel[0].data

        # Interpolated data attributes
        # NOTE: Data from HLSP sometimes are incomplete (NaNs), this allows our models to be continuous
        self.interp_time = self.interpolate_data(deepcopy(self.time))
        self.interp_full_model = self.interpolate_data(deepcopy(self.full_model))
        self.interp_astro_model = self.interpolate_data(deepcopy(self.astro_model))

        # Metadata
        self.planet_name = self.data.PLANET
        self.telescope = self.data.TELESCOP
        self.instrument = self.data.INSTRUME
        self.filter = self.data.FILTER
        self.propid = self.data.PROPOSID

    def build_time_bin_slider(self, start, end, step):
        return Slider(title="Bin Size", value=start, start=start, end=end, step=step)

    def build_light_curve_plot(self, time, flux, flux_err, model):
        """Build bokeh light curve figure. This takes the input and returns a single
        figure object.

        Parameters
        ----------
        time : array like
            Light curve time values

        flux : array like
            Flux values associated with light curve

        flux_err : array like
            Error in flux measurements

        model : array like
            Model fit for light curve data.

        Returns
        -------
        layout : bokeh.models.layouts.Column
            Bokeh figure containing light curve data and bin slider
        """
        source = ColumnDataSource(
            data=dict(
                time=time,
                flux=flux,
                flux_err=flux_err,
                model=model,
                upper=flux + flux_err,
                lower=flux - flux_err,
            )
        )

        original = ColumnDataSource(
            data=dict(
                time=time,
                flux=flux,
                flux_err=flux_err,
                model=model,
                upper=flux + flux_err,
                lower=flux - flux_err,
            )
        )

        p = figure(
            width=self.plot_width,
            height=self.plot_height,
            tooltips=[
                ("Time", "@time"),
                ("Flux", "@flux"),
                ("Flux Error", "@flux_err"),
            ],
        )

        p.axis.axis_label_text_font_style = "bold"

        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"

        p.xaxis.major_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_size = "15pt"

        p.xaxis.axis_label = "Time (BJD_TDB)"
        p.yaxis.axis_label = "Normalized Flux"

        plot_title = f"Target: {self.planet_name} Configuration: {self.telescope} | {self.instrument} | {self.filter}"
        p.title.text = plot_title
        p.title.text_font_size = "25pt"

        scatter = p.scatter(
            x="time",
            y="flux",
            color="black",
            size=5,
            line_alpha=0,
            source=source,
            legend_label="Measured Flux",
        )

        error = Whisker(
            base="time",
            upper="upper",
            lower="lower",
            source=source,
            level="annotation",
            line_width=1,
        )

        error.upper_head.size = 20
        error.lower_head.size = 20
        p.add_layout(error)

        p.line("time", "model", source=source, color="red", legend_label="Model Fit")

        with open(Path(__file__).parent / "js" / "lightcurve.js", "r") as file:
            js_content = file.read()

        callback = CustomJS(
            args=dict(source=source, original=original), code=js_content
        )
        slider = self.build_time_bin_slider(1, 20, 1)
        slider.js_on_change("value", callback)
        p.legend.click_policy = "hide"
        p.hover.renderers = [scatter]

        layout = column(slider, p)

        return layout

    def get_raw_light_curve(self):
        """Convenience method to build raw light curve"""
        self.raw_light_curve = self.build_light_curve_plot(
            self.interp_time, self.raw_flux, self.flux_err, self.interp_full_model
        )

    def get_cleaned_light_curve(self):
        """Convenience method to build cleaned light curve"""
        self.cleaned_light_curve = self.build_light_curve_plot(
            self.interp_time, self.cleaned_flux, self.flux_err, self.interp_astro_model
        )

    def interpolate_data(self, data):
        """One dimensional interpolation routine to replace NaNs

        Parameters
        ----------
        data : np.array
            Array with NaNs. If no NaNs present, array is unchanged.

        Returns
        -------
        data : np.array
            Array with interpolated values replacing NaNs.
        """
        nans, x = np.isnan(data), lambda z: z.nonzero()[0]
        data[nans] = np.interp(x(nans), x(~nans), data[~nans])

        return data

    def run(self, figure_out_path=None):
        """Convenience method to build figure served in the Rocky Worlds Website.

        Parameters
        ----------
        plot_outname : string
            Absolute path to save figure to
        """
        self.get_raw_light_curve()
        self.get_cleaned_light_curve()

        clean_tab = TabPanel(
            child=self.cleaned_light_curve, title="Cleaned Light Curve"
        )
        raw_tab = TabPanel(child=self.raw_light_curve, title="Raw Light Curve")

        tabbed_plots = Tabs(
            tabs=[clean_tab, raw_tab],
            stylesheets=[{".bk-tab": Styles(font_size="1.0rem")}],
        )

        if figure_out_path:
            filename = self.filename.name.replace("h5", "ql.html")
            full_file_path = Path(figure_out_path) / filename
            write_figure(tabbed_plots, full_file_path)
        else:
            show(tabbed_plots)
