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

    """
    template = Template("""<!DOCTYPE html>
                            <html lang="en">
                                <head>
                                    <meta charset="utf-8">
                                    <title>Bokeh Plot</title>
                                    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Oswald" />
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
                    "axis_label_text_font": "Oswald",
                    "major_label_text_font": "Oswald",
                },
                "Title": {
                    "text_font": "Oswald",
                },
                "Legend": {
                    "label_text_font": "Oswald",
                },
                "Label": {
                    "text_font": "Oswald",
                },
            }
        }
    )

    script, div = components(bokeh_figure, theme=theme)
    resources = CDN.render()
    html = template.render(resources=resources, script=script, div=div)

    with open(outfile, mode="w", encoding="utf-8") as f:
        f.write(html)
