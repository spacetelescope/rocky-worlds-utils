#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bokeh templating module.
Authors
-------
- Mees Fix <<mfix@stsci.edu>>
"""

from bokeh.embed import components
from bokeh.themes import Theme
from bokeh.resources import CDN

from jinja2 import Template


def write_figure(bokeh_figure, outfile):
    """Write out bokeh figures using template that will match to our website.
    Parameters
    ----------
    bokeh_figure: bokeh.plotting.figure
        Bokeh figure to convert text and name of plot
    """
    template = Template("""<!DOCTYPE html>
                                <html lang="en">
                                    <head>
                                        <meta charset="utf-8">
                                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
                                        {{ resources }}
                                        {{ script }}
                                    </head>
                                    <body>
                                        {{ div }}
                                    </body>
                                </html>
                                """)

    # Set theme for custom font type to match website.
    theme = Theme(
        json={
            "attrs": {
                "Axis": {
                    "axis_label_text_font": "Montserrat",
                    "major_label_text_font": "Montserrat",
                },
                "Title": {
                    "text_font": "Montserrat",
                },
                "Legend": {
                    "label_text_font": "Montserrat",
                },
                "Label": {
                    "text_font": "Montserrat",
                },
            }
        }
    )

    script, div = components(bokeh_figure, theme=theme)
    resources = CDN.render()
    html = template.render(resources=resources, script=script, div=div)

    with open(outfile, mode="w", encoding="utf-8") as f:
        f.write(html)
