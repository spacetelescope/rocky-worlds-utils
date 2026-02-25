#! /usr/bin/env python
"""
Generate Targets Under Consideration interactive figure

Authors
-------
- Mees Fix <<mfix@stsci.edu>>
- Leonardo Ubeda <<lubeda@stsci.edu>>
"""

import os
import importlib.resources
from jinja2 import Template

from bokeh.embed import components
from bokeh.models import (
    ColumnDataSource,
    FixedTicker,
    Legend,
    LegendItem,
    Label,
    TapTool,
    OpenURL,
    DataTable,
    TableColumn,
    SelectEditor,
    IntEditor,
    CDSView,
    IndexFilter,
)
from bokeh.layouts import column
from bokeh.plotting import figure, show
import numpy as np
import pandas as pd
from PIL import Image


from rocky_worlds_utils.constants import SELECTED_TARGETS
from rocky_worlds_utils.figure_utils.write_figure import write_figure

DATA_DIR = (
    importlib.resources.files("rocky_worlds_utils")
    / "cross_mission"
    / "figures"
    / "targets_under_consideration"
    / "tuc_figure_data"
)


class make_tuc_figure:
    """Make TUC Interactive Figure"""

    def __init__(self, figure_out_path=None):
        self.figure_out_path = figure_out_path
        self.data = pd.read_csv(DATA_DIR / "interactive_tuc_plot_data.csv")
        self.build_figure()

    def build_figure(self, width=1150, height=650):
        """Incrementally build TUC figure using Bokeh."""

        # Define Source Data Object
        # -------------------------
        source = ColumnDataSource(self.data)

        # DataTable Settings
        # ------------------
        css = """
            .slick-header-columns {
                    background-color: #17648D !important;
                    font-family: Montserrat;
                    font-weight: bold;
                    font-size: 14pt;
                    color: #FFFFFF;
                    text-align: center;
                    }
                  .slick-row{
                    font-size: 10pt;
                    font-family: Montserrat;
                    text-align: center;
                    font-weight: normal;
                    }
                  .slick-cell.selected {
                    background-color: #9bdfe8;
                    }"""

        columns = [
            TableColumn(
                field="planet_name",
                title="Planet Name",
                editor=SelectEditor(options=self.data["planet_name"].tolist()),
            ),
            TableColumn(field="v_esc", title="Escape Velocity", editor=IntEditor()),
            TableColumn(field="I", title="Irradiance", editor=IntEditor()),
            TableColumn(
                field="priority_metric", title="Priority Metric", editor=IntEditor()
            ),
        ]
        self.data_table = DataTable(
            source=source,
            columns=columns,
            editable=True,
            width=width,
            height=400,
            index_position=None,
            stylesheets=[css],
        )

        # Generate figure object
        # ----------------------
        self.p = figure(
            width=width,
            height=height,
            x_axis_type="log",
            y_axis_type="log",
            sizing_mode="fixed",
            x_range=(4, 25),
            y_range=(0.1, 10000),
            tools=["lasso_select, box_zoom, tap, reset, hover, save"],
            tooltips=[
                ("Target Name", "@planet_name"),
                ("Escape Velocity", "@v_esc"),
                ("Irradiance", "@I"),
                ("Priority Metric", "@priority_metric"),
            ],
        )

        # Figure Configuration
        # --------------------
        # Axis Labels

        self.p.xaxis.axis_label = "Escape Velocity (km/s)"
        self.p.xaxis.axis_label_text_font_size = "15pt"
        self.p.xaxis.axis_label_text_font_style = "normal"

        self.p.yaxis.axis_label = "I (Relative Cummulative XUV Irradiation)"
        self.p.yaxis.axis_label_text_font_size = "15pt"
        self.p.yaxis.axis_label_text_font_style = "normal"

        # Axis Tick Formatting
        self.p.xaxis.ticker = FixedTicker(ticks=[4, 5, 6, 7, 8, 9, 10, 20])
        self.p.yaxis.ticker = FixedTicker(ticks=[0.1, 1, 10, 100, 1000, 10000])
        self.p.xaxis.major_label_text_font_size = "15pt"
        self.p.yaxis.major_label_text_font_size = "15pt"

        # Data Plotting
        # -------------
        selected_target_index = self.find_selected_targets()
        selected_targets_view = CDSView(filter=IndexFilter(selected_target_index))

        unselected_target_index = self.data.index.difference(
            selected_target_index
        ).tolist()
        unselected_targets_view = CDSView(filter=IndexFilter(unselected_target_index))

        selected_scatter = self.p.scatter(
            "v_esc",
            "I",
            color="darkorange",
            line_color="line_color",
            line_width=3,
            size=15,
            view=selected_targets_view,
            source=source,
        )

        unselected_scatter = self.p.scatter(
            "v_esc",
            "I",
            line_color="line_color",
            line_width=3,
            size=15,
            view=unselected_targets_view,
            source=source,
        )

        # define cosmic shoreline
        x_shore = [4, 25]
        y_shore = [0.2, 295]
        _ = self.p.line(x=x_shore, y=y_shore, line_color="black", line_width=1)

        # shade below cosmic shoreline
        _ = self.p.varea(
            x=x_shore,
            y1=self.p.y_range.start,
            y2=y_shore,
            alpha=0.05,
            color="navy",
        )

        # shade above cosmic shoreline
        _ = self.p.varea(
            x=x_shore,
            y1=self.p.y_range.end,
            y2=y_shore,
            alpha=0.05,
            color="darkred",
        )

        # Plot Tool Settings
        # ------------------
        # Tap tool settings
        url = f"https://exoplanetarchive.ipac.caltech.edu/overview/@planet_name#planet_{'@planet_name'.replace(' ', '-')}_collapsible"
        taptool = self.p.select(type=TapTool)
        taptool.callback = OpenURL(url=url)
        taptool.renderers = [selected_scatter, unselected_scatter]

        # hover only for scatters
        self.p.hover.renderers = [
            selected_scatter,
            unselected_scatter,
        ]

        # Legend Settings
        # ---------------
        # Defining Custom Legend Items
        legend_items = [
            LegendItem(
                label="Precise Mass Constraint",
                renderers=[
                    self.p.scatter(
                        1,
                        1,
                        fill_color=None,
                        line_color="mediumseagreen",
                        line_width=2,
                        size=10,
                    )
                ],
            ),
            LegendItem(
                label="No Mass Constraint",
                renderers=[
                    self.p.scatter(
                        1,
                        1,
                        fill_color=None,
                        line_color="darkred",
                        line_width=2,
                        size=10,
                    )
                ],
            ),
            LegendItem(
                label="Rocky Worlds DDT Targets",
                renderers=[
                    self.p.scatter(
                        1,
                        1,
                        color="darkorange",
                        line_width=2,
                        size=10,
                    )
                ],
            ),
        ]

        # Create a legend object
        legend = Legend(items=legend_items, location="bottom_right")
        self.p.add_layout(legend)

        # Configure legend
        self.p.legend.label_text_font_size = "15pt"
        self.p.legend.border_line_width = 3
        self.p.legend.border_line_color = "black"

        labels = [
            Label(
                x=4.5,
                y=6000.0,
                text="ATMOSPHERES LESS LIKELY",
                text_color="red",
                text_font_size="12pt",
            ),
            Label(
                x=4.5,
                y=0.2,
                text="ATMOSPHERES MORE LIKELY",
                text_color="darkblue",
                text_font_size="12pt",
            ),
            Label(
                x=6,
                y=1.1,
                text="COSMIC SHORELINE",
                text_color="black",
                text_font_size="12pt",
                angle=0.36,
            ),
            Label(
                x=11.6,
                y=0.85,
                text="EARTH",
                text_color="black",
                text_font_size="8pt",
            ),
            Label(
                x=10.7,
                y=3.3,
                text="VENUS",
                text_color="black",
                text_font_size="8pt",
            ),
            Label(
                x=5.5,
                y=0.55,
                text="MARS",
                text_color="black",
                text_font_size="8pt",
            ),
            Label(
                x=4.4,
                y=9.1,
                text="MERCURY",
                text_color="black",
                text_font_size="8pt",
            ),
        ]
        for label in labels:
            self.p.add_layout(label)

        planet_images = {
            "mars.png": [5.32, 0.6],
            "venus.png": [10.36, 3.6],
            "mercury.png": [4.25, 10],
            "earth.png": [11.2, 1.0],
        }
        for planet_img, coords in planet_images.items():
            coord_x = coords[0]
            coord_y = coords[1]
            pl_img_name = str(DATA_DIR / planet_img)
            self.plot_solar_system_images(pl_img_name, coord_x, coord_y)

        grid = column(self.p, self.data_table, sizing_mode="scale_both")

        if self.figure_out_path:
            filename = "targets_under_consideration.html"
            full_file_path = os.path.join(self.figure_out_path, filename)
            write_figure(grid, full_file_path)
        else:
            show(grid)

    def find_selected_targets(self):
        """Find indices where targets matched values in SELECTED_TARGETS"""
        indices = self.data[
            self.data["planet_name"].isin(SELECTED_TARGETS)
        ].index.tolist()

        return indices

    def plot_solar_system_images(self, png_path, x_coord, y_coord):
        """Plot images of solar system planets in bokeh plot"""

        log_width = 0.02
        log_height = 0.2

        with Image.open(png_path) as img:
            img = img.convert("RGBA")
            img = img.resize((200, 200), Image.Resampling.LANCZOS)
            image_data = np.array(img, dtype=np.uint8)

            r, g, b, a = (
                image_data[:, :, 0],
                image_data[:, :, 1],
                image_data[:, :, 2],
                image_data[:, :, 3],
            )
            image_data_2d = (
                (r.astype(np.uint32) << 24)
                | (g.astype(np.uint32) << 16)
                | (b.astype(np.uint32) << 8)
                | a.astype(np.uint32)
            )

            # Convert image to NumPy array
            image_data = np.array(img, dtype=np.uint8)
            r, g, b, a = (
                image_data[:, :, 0],
                image_data[:, :, 1],
                image_data[:, :, 2],
                image_data[:, :, 3],
            )
            image_data_2d = (
                (a.astype(np.uint32) << 24)
                | (b.astype(np.uint32) << 16)
                | (g.astype(np.uint32) << 8)
                | r.astype(np.uint32)
            )
            image_data_2d = np.flipud(image_data_2d)

            # Center coordinates in log-space
            x_center = x_coord
            y_center = y_coord

            x_start = 10 ** (np.log10(x_center) - log_width / 2)
            y_start = 10 ** (np.log10(y_center) - log_height / 2)

            dw_log = 10 ** (np.log10(x_center) + log_width / 2) - x_start
            dh_log = 10 ** (np.log10(y_center) + log_height / 2) - y_start

            # Place the image in the plot
            self.p.image_rgba(
                image=[image_data_2d],
                x=[x_start],
                y=[y_start],
                dw=[dw_log],
                dh=[dh_log],
                dilate=False,
            )
