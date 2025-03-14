import bokeh.plotting as bpl
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh import events
from bokeh.layouts import column, row
from bokeh.models import (
    CategoricalSlider,
    ColumnDataSource,
    CustomJS,
    DataTable,
    Div,
    FactorRange,
    HelpButton,
    MultiSelect,
    OpenURL,
    TabPanel,
    Select,
    Tabs,
    TableColumn,
    TextInput,
    TapTool,
    Tooltip,
    HelpTool,
)
from bokeh.models.dom import HTML
from bokeh.plotting import figure

from misc.data import ImageLabels
from .constants import generate_tooltips


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


def make_plot_slider(
    m_plots, dist_intervals, n_neighbors, metrics_table, distribution_plot
):
    d_intervals = list(map(str, sorted(dist_intervals)))
    umap_dist_slider = CategoricalSlider(
        categories=d_intervals, title="min_dist", value=d_intervals[0]
    )
    n_intervals = list(map(str, sorted(n_neighbors)))
    umap_neighbour_slider = CategoricalSlider(
        categories=n_intervals, title="neighbour", value=str(n_intervals[0])
    )

    for p in m_plots[1:]:
        p.visible = False

    """
    plots has a list like this
    [ d1_n1, d1_n2, d1_n3, d2_n1, d2_n2, d2_n3, d3_n1, d3_n2, d3_n3]
    """
    callback = CustomJS(
        args=dict(
            plots=m_plots,
            d_i=d_intervals,
            n_i=n_intervals,
            slider_dist=umap_dist_slider,
            slider_n=umap_neighbour_slider,
        ),
        code="""
        var dist_index = d_i.indexOf(slider_dist.value);
        var n_index = n_i.indexOf(slider_n.value);
        for (var i = 0; i < plots.length; i++) {
            plots[i].visible = false;
        }
        var target_idx = dist_index * (n_i.length) + n_index;
        plots[target_idx].visible = true;
    """,
    )
    umap_dist_slider.js_on_change("value", callback)
    umap_neighbour_slider.js_on_change("value", callback)

    min_dist_help_button = HelpButton(
        tooltip=Tooltip(
            content=HTML(
                """
    The minimum distance between points in the embedding. Smaller values (close to zero) are good for local patterns (also measured by KNN),
    larger values (close to 1) are good for global patterns (also measured by CPD)."""
            ),
            position="top_center",
        )
    )
    n_neighbors_help_button = HelpButton(
        tooltip=Tooltip(
            content=HTML(
                """
    How many neighbors to consider for each point in the embedding. Smaller values are good for local patterns (also measured by KNN),
    larger values are good for global patterns (also measured by CPD)."""
            ),
            position="top_center",
        )
    )

    return column(
        column(
            Div(
                text="<p style='margin-top: 0px;margin-bottom:0px'>UMAP parameters:</h4>"
            ),
            row(
                row(umap_dist_slider, min_dist_help_button),
                row(umap_neighbour_slider, n_neighbors_help_button),
            ),
        ),
        column(row(*m_plots), row(distribution_plot, metrics_table)),
    )


def make_plot_select(plots, titles):
    for p in plots[1:]:
        p.visible = False
    select = Select(
        title="Select variable to cluster by:", value=titles[0], options=titles
    )

    callback = CustomJS(
        args=dict(plots=plots, titles=titles),
        code="""
        var selected = cb_obj.value;
        for (var i = 0; i < plots.length; i++) {
            plots[i].visible = (titles[i] === selected);
        }
    """,
    )
    # Attach the CustomJS callback to the Select widget
    select.js_on_change("value", callback)

    return column(select, row(*plots))


"""
Inspired from the "interactive" method from https://github.com/lmcinnes/umap/blob/d4d4c4aeb96e0d2296b5098d9dc9736de79e4e96/umap/plot.py#L1220
"""


def make_umap_widget(
    umap_projection,
    raw_umap_projection,
    data,
    tooltips,
    unique_labels,
    interactive_text_search_columns,
    width=800,
    height=800,
    point_size=None,
    interactive_text_search_alpha_contrast=1.0,
    title=None,
):
    tabs = []
    data_sources = []
    for points, prefix in [(umap_projection, ""), (raw_umap_projection, "Raw ")]:
        if points is None:
            continue
        if points.shape[1] != 2:
            raise ValueError("Plotting is currently only implemented for 2D embeddings")

        if point_size is None:
            point_size = 100.0 / np.sqrt(points.shape[0])

        coordinates = pd.DataFrame(points, columns=("x", "y"))
        # merge coordinates with data
        data_source = bpl.ColumnDataSource(pd.concat([coordinates, data], axis=1))
        data_sources.append(data_source)

        plot = bpl.figure(
            width=width,
            height=height,
            tooltips=tooltips,
            title=title,
        )


        tt = TapTool()
        tt.callback = OpenURL(url="@image_url")
        # prevent selection after click on tt
        tt.gesture = "doubletap"

        plot.tools = [t for t in plot.tools if not isinstance(t, HelpTool)]
        plot.tools.append(tt)
        plot.scatter(
            x="x",
            y="y",
            source=data_source,
            legend_group="label",
            color="color",
            muted_color="color",
            size=point_size,
            alpha="alpha",
        )

        # Doesn't work, just hides all the points?
        # plot.legend.click_policy="mute"

        tooltips = generate_tooltips(
            data_source.column_names, interactive_text_search_columns[0]
        )
        plot.hover.tooltips = tooltips

        plot.grid.visible = False
        plot.axis.visible = False

        tab_scatter = TabPanel(child=plot, title=f"{prefix}scatter")
        tab_panes = [tab_scatter]

        if (
            ImageLabels.IMAGE_URL in data.columns
            and ImageLabels.FILENAME in data.columns
        ):
            plot_img = bpl.figure(
                width=width,
                height=height,
                tooltips=tooltips,
            )
            plot_img.rect(
                x="x",
                y="y",
                width="w",
                height="h",
                line_color="color",
                line_alpha="line_alpha",
                line_width=1,
                fill_alpha=0,
                source=data_source,
            )
            plot_img.image_url(
                url="image_url",
                source=data_source,
                x="x",
                y="y",
                w="w",
                h="h",
                anchor="center",
            )
            plot_img.grid.visible = False
            plot_img.axis.visible = False
            tab_image = TabPanel(child=plot_img, title=f"{prefix}images")
            tab_panes.append(tab_image)

        tabs.append(Tabs(tabs=tab_panes))

    callback_selector = CustomJS(
        args=dict(
            source=data_sources,
            matching_alpha=interactive_text_search_alpha_contrast,
            search_col=interactive_text_search_columns[0],
        ),
        code="""
    for (var i = 0; i < source[0].data.x.length; i++) {
        var label = String(source[0].data[search_col][i])
        for (var j in source) {
            if (cb_obj.value.includes(label)) {
                    source[j].data['alpha'][i] = matching_alpha;
                    source[j].data['line_alpha'][i] = 0.2;
            } else {
                source[j].data['alpha'][i] = 0;
                source[j].data['line_alpha'][i] = 0.0;
            }
        }
    }
    for (var j in source) {
        source[j].change.emit();
    }
    """,
    )

    text_input = TextInput(value="", title="Search:")
    interactive_text_search_columns.append("filename")

    callback_search = CustomJS(
        args=dict(
            source=data_sources[0],
            source_raw=data_sources[1] if len(data_sources) > 1 else data_sources[0],
            matching_alpha=interactive_text_search_alpha_contrast,
            search_columns=interactive_text_search_columns,
        ),
        code="""
        var text_search = cb_obj.value;

        var search_columns_dict = {}
        for (var col in search_columns){
            search_columns_dict[col] = search_columns[col]
        }
        // Loop over columns and values
        // If there is no match for any column for a given row, change the alpha value
        var string_match = false;
        for (var i = 0; i < source.data.x.length; i++) {
            string_match = false
            for (var j in search_columns_dict) {
                var possible_match = String(source.data[search_columns_dict[j]][i])
                if (possible_match.includes(text_search) ) {
                    string_match = true;
                }
            }
            if (string_match){
                source.data['alpha'][i] = matching_alpha;
                source_raw.data['alpha'][i] = matching_alpha;
                source.data['line_alpha'][i] = 0.2;
                source_raw.data['line_alpha'][i] = 0.2;
            }else{
                source.data['alpha'][i] = 0.0;
                source_raw.data['alpha'][i] = 0.0;
                source.data['line_alpha'][i] = 0.0;
                source_raw.data['line_alpha'][i] = 0.0;
            }
        }
        source.change.emit();
    """,
    )

    text_input.js_on_change("value", callback_search)
    multibox_input = MultiSelect(options=list(map(str, unique_labels)), size=40)
    multibox_input.js_on_change("value", callback_selector)

    # # make it possible to hide the legend with a doubletap
    # def show_hide_legend(legend=plot.legend[0]):
    #     legend.visible = not legend.visible

    callback_legend = CustomJS(args=dict(legend=plot.legend[0]), code="""
        legend.visible = !legend.visible;
    """)
    plot.js_on_event(events.DoubleTap, callback_legend)



    return row(row(*tabs), text_input, multibox_input)


def make_distribution_plot(datasource, unique_labels):
    distribution_plot = figure(
        toolbar_location=None,
        title="Distribution plot (number of points overall)",
        tools="hover",
        tooltips=[("Count", "@count")],
        width=500,
        height=300,
        x_range=FactorRange(factors=list(map(str, unique_labels))),
    )
    counts = datasource[["label", "color"]].value_counts().to_frame().reset_index()
    distribution_plot.vbar(
        source=ColumnDataSource(data=counts),
        x="label",
        top="count",
        width=0.9,
        color="color",
    )
    distribution_plot.y_range.start = 0
    if len(unique_labels) > 10:
        distribution_plot.xaxis.major_label_orientation = 1.2

    return distribution_plot


def make_metrics_table(metrics_data):
    source = ColumnDataSource(metrics_data)
    columns = [TableColumn(field=Ci, title=Ci) for Ci in metrics_data.columns]
    header = Div(
        text="<h4 style='margin-top: 0px;margin-bottom:0px'>Clustering metrics</h4>"
    )
    return column(
        [
            header,
            DataTable(
                source=source,
                columns=columns,
                index_position=None,
                width=600,
                height=400,
            ),
        ]
    )


def make_datasource(labels, label_key, img_preview_width=0.2, img_preview_height=0.2):
    data = labels.copy() # use integers for indexing
    data["label"] = labels[[label_key]]

    color_key_cmap = "tab20"
    unique_labels = sorted(data[label_key].unique())
    num_labels = len(unique_labels)
    if len(unique_labels) > 20:
        color_key_cmap = "Spectral"

    color_key = _to_hex(plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels)))
    new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
    data["color"] = labels[label_key].map(new_color_key)
    tooltip_dict = {}

    for col in labels.columns:
        tooltip_dict[col] = "@{" + str(col) + "}"

    # data["filename"] = labels['filename']
    # data["image_url"] = labels["image_url"]
    tooltips = list(tooltip_dict.items())

    data["alpha"] = 1
    data["line_alpha"] = 0.2
    data["w"] = img_preview_width
    data["h"] = img_preview_height

    return data, tooltips, unique_labels
