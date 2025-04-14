"""Module that contains the class to generate the lightcurve output served via
the Rocky Worlds website.

Authors
-------
- Mees Fix

Use
---
>>> from rocky_worlds_utils.lightcurve_plot import rockyWorldsLightCurve
>>> filename = "hlsp_for_rockyworlds_lc.h5"
>>> light_curve = rockyWorldsLightCurve(filename)
"""

from bokeh.models import TabPanel, Tabs, ColumnDataSource
from bokeh.plotting import figure, show, output_file, save

import xarray as xr


class rockyWorldsLightCurve:
    """Class to build light curves for Rocky Worlds DDT data products."""

    def __init__(self, filename, plot_height=600, plot_width=1400):
        """
        Parameters
        ----------
        filename : str
            Path of Rocky Worlds light curve data product file (extension `lc.h5`)
        plot_height : int
            Size of bokeh plot height (default)
        plot_width : int
            Size of bokeh plot width
        """
        self.filename = filename
        self.plot_width = plot_width
        self.plot_height = plot_height

        self.data = xr.load_dataset(filename)

        # Data Attributes
        self.time = self.data.time
        self.raw_flux = self.data.rawFlux[0]
        self.flux_err = self.data.rawFluxErr[0]
        self.cleaned_flux = self.data.cleanedFlux[0]
        self.full_model = self.data.fullModel[0]
        self.astro_model = self.data.astroModel[0]

        # Metadata
        self.planet_name = self.data.PLANET
        self.telescope = self.data.TELESCOP
        self.instrument = self.data.INSTRUME
        self.filter = self.data.FILTER
        self.propid = self.data.PROPOSID

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
        p : bokeh.plotting.figure
            Bokeh figure containing light curve data
        """
        source = ColumnDataSource(
            data=dict(time=time, flux=flux, flux_err=flux_err, model=model)
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

        output_file(filename=self.filename.replace("h5", "html"))

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

        x_err_bar, flux_err_bar = self.build_error_bars(time, flux, flux_err)
        p.multi_line(x_err_bar, flux_err_bar, color="black", legend_label="Error Bar")

        p.line(time, model, color="red", legend_label="Model Fit")

        p.legend.click_policy = "hide"
        p.hover.renderers = [scatter]

        return p

    def get_raw_light_curve(self):
        """Convenience method to build raw light curve"""
        self.raw_light_curve = self.build_light_curve_plot(
            self.time, self.raw_flux, self.flux_err, self.full_model
        )
        self.raw_light_curve.xaxis.axis_label = "Time (BJD_TDB)"
        self.raw_light_curve.yaxis.axis_label = "Normalized Raw Flux"

    def get_cleaned_light_curve(self):
        """Convenience method to build cleaned light curve"""
        self.cleaned_light_curve = self.build_light_curve_plot(
            self.time, self.cleaned_flux, self.flux_err, self.astro_model
        )
        self.cleaned_light_curve.xaxis.axis_label = "Time (BJD_TDB)"
        self.cleaned_light_curve.yaxis.axis_label = "Normalized Cleaned Flux"

    def build_error_bars(self, time, flux, flux_err):
        """Bokeh does not currently have an error bar plotting glyph.
        This creates a line glyph the lays on top of the scatter points.

        Parameters
        ----------
        time : array like
            Light curve time values
        flux : array like
            Flux values associated with light curve
        flux_err : array like
            Error in flux measurements

        Returns
        -------
        err_bar_time : array like
            List of tuples (time, time)
        err_bar_flux : array like
            List of tuples (flux - err, flux + err)
        """
        err_bar_time = []
        err_bar_flux = []
        for t, rflx, rflx_err in zip(time, flux, flux_err):
            err_bar_time.append((t, t))
            err_bar_flux.append((rflx - rflx_err, rflx + rflx_err))
        return err_bar_time, err_bar_flux

    def run(self, save_plot=False):
        """Convenience method to build figure served in the Rocky Worlds Website.

        Parameters
        ----------
        save_plot : bool
            Flag to save plot, if False, plot is displayed to browser.
            Saved figure name is `self.filename` with `h5` replaced with `html`

        Returns
        -------
        None
        """
        self.get_raw_light_curve()
        self.get_cleaned_light_curve()

        clean_tab = TabPanel(
            child=self.cleaned_light_curve, title="Cleaned Light Curve"
        )
        raw_tab = TabPanel(child=self.raw_light_curve, title="Raw Light Curve")

        tabbed_plots = Tabs(tabs=[clean_tab, raw_tab])

        if save_plot:
            save(Tabs(tabs=[clean_tab, raw_tab]))
        else:
            show(tabbed_plots)
