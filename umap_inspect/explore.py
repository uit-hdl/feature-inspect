#!/usr/bin/env python3
# coding: utf-8
from misc.global_util import logger

import os
import shutil
import time
from collections import defaultdict
from typing import List

from pandas import DataFrame
from tensorboardX import SummaryWriter


_CUML_IMPORTED = False
import math
import numpy as np
import pandas as pd
import umap
import zarr
import zarr.storage
from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models import Div
from bokeh.plotting import show
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

from misc.benchmark import track_method
from misc.data import ImageLabels
from misc.global_util import ensure_dir_exists

from umap_inspect.constants import METRIC_HELPTEXT, METRIC_CITETEXT, UmapMode
from umap_inspect.umap_metrics import compute_knn, compute_cpd
from umap_inspect.umap_utils import (
    make_umap_widget,
    make_plot_select,
    make_plot_slider,
    make_metrics_table,
    make_distribution_plot,
    make_datasource,
)


def get_umap_neighbors_intervals(num_values, writer, i):
    perplexity_score = math.floor(num_values / 100) if num_values > 200 else 2
    writer.add_scalar("ue_perplexity", perplexity_score, i)
    if num_values > 15:
        return sorted({perplexity_score, 15})
    return sorted({perplexity_score})


class UmapConfig:
    """Configuration object to store UMAP-related settings."""

    def __init__(self, values, dist, n, mode, knn=False, cpd=False):
        self.values = values
        self.min_dist = dist
        self.n = n
        self.mode = mode
        self.knn = knn
        self.cpd = cpd

    def __str__(self):
        return f"UmapConfig({self.mode}, {self.min_dist}, {self.n})"

    @property
    def lookup_key(self):
        return f"{self.mode}/d_{self.min_dist}n_{self.n}"


class UmapPlot:
    """Handles the generation and storage of UMAP plots."""

    def __init__(self, config, embedding, knn=None, cpd=None):
        self.config = config
        self.embedding = embedding
        self.knn = knn
        self.cpd = cpd

    def gen_df(self):
        """Generate a DataFrame with the UMAP configuration and metrics."""
        df = pd.DataFrame(
            {
                "mode": [self.config.mode],
                "d": [self.config.min_dist],
                "n": [self.config.n],
            }
        )
        if self.knn is not None:
            df["knn"] = [self.knn]
        if self.cpd is not None:
            df["cpd"] = [self.cpd]

        return df


class ClusteringMetrics:
    """Handles the generation and storage of UMAP clustering scores / metrics."""

    def __init__(self, key, do_ss=False):
        self.df = pd.DataFrame()
        self.key = key
        self.ss = do_ss

    def __call__(self, i, umap_runner, labels):
        lookup_key = umap_runner.config.lookup_key + "/metrics"
        unique_labels = sorted(labels[self.key].unique())
        if len(unique_labels) <= 1:
            return self.df

        if lookup_key in umap_runner.zroot:
            node = umap_runner.zroot[lookup_key]
            self.df = pd.DataFrame({})
            if "ss" in node.attrs:
                self.df["ss"] = [node.attrs["ss"]]
        else:
            self.df = self._gen_scores(
                umap_runner, labels[self.key].tolist(), unique_labels
            )
            self._cache_metrics(umap_runner.zroot, lookup_key)

        return self.df

    def _gen_scores(self, umap_runner, labels, unique_labels):
        ll_d = np.array([list(unique_labels).index(l) for l in labels])
        df = pd.DataFrame({})
        if self.ss:
            ss = track_method(
                silhouette_score,
                "ue_silhouette_score",
                umap_runner.writer,
                umap_runner.i,
            )(umap_runner.plot.embedding, labels)
            df["ss"] = [ss]
        return df

    def _cache_metrics(self, zroot: zarr.Group, lookup_key):
        node: zarr.Group = zroot.require_group(lookup_key)
        dst = node.create_array("metrics", shape=[1], dtype=np.float32)
        for key, value in self.df.items():
            dst.attrs[key] = value.values[0]


class UmapRunner:
    """Orchestrates UMAP creation and output generation."""

    def __init__(
        self,
        config: UmapConfig,
        zroot: zarr.Group,
        writer: SummaryWriter = None,
        use_cuml: bool = False,
    ):
        self.config = config
        self.writer = writer
        self.zroot = zroot
        self.i = -1
        self.plot: UmapPlot | None = None
        self.use_cuml = use_cuml

    def __str__(self):
        return f"UmapRunner({self.config})"

    def run(self, i):
        if self.writer:
            tracker = lambda l, s: track_method(l, s, self.writer, i)
            self.plot = self._generate_embedding(tracker)
        else:
            self.plot = self._generate_embedding(lambda l, s: l)
        self.i = i

    def _generate_embedding(self, tracker):
        # if cached, fetch and return
        if self.zroot is not None and self.config.lookup_key in self.zroot:
            node = self.zroot[self.config.lookup_key]["plot"]
            plot = UmapPlot(
                self.config,
                node[:],
                knn=node.attrs["knn"] if "knn" in node.attrs else None,
                cpd=node.attrs["cpd"] if "cpd" in node.attrs else None,
            )
            return plot

        if len(self.config.values) >= 10 and _CUML_IMPORTED:
            umap_conf = cumlUMAP(
                n_neighbors=self.config.n, min_dist=self.config.min_dist
            )
        else:
            umap_conf = umap.UMAP(
                n_neighbors=self.config.n, min_dist=self.config.min_dist
            )

        embedding = tracker(
            umap_conf.fit_transform, f"umap_{self.config.mode.split(' ')[0]}"
        )(np.array(self.config.values))

        ds = None
        # if not in cache, insert
        if self.zroot is not None:
            arr = self.zroot.create_array(
                name=os.path.join(self.config.lookup_key, "plot"),
                shape=embedding.shape,
                dtype=embedding.dtype,
            )
            arr[:] = embedding

        # We only compute knn and cpd if we are in plain mode, not raw
        knn = None
        cpd = None

        if self.config.mode == UmapMode.PLAIN:
            if self.config.knn:
                knn = np.mean(
                    tracker(compute_knn, f"knn")(self.config.values, embedding, k=10)
                )
                if ds:
                    ds.attrs.update({"knn": knn})
            if self.config.cpd:
                cpd = tracker(compute_cpd, f"cpd")(
                    self.config.values, embedding, sample_size=len(embedding) // 10
                )[0]
                if ds:
                    ds.attrs.update({"cpd": cpd})

        plot = UmapPlot(self.config, embedding, knn=knn, cpd=cpd)
        return plot


def compute_optimal_n_neighbors(num_values, num_unique_labels):
    n_neighbors_theoretical = math.floor(num_values / num_unique_labels)
    if n_neighbors_theoretical < 2:
        raise ValueError("Number of unique labels is too high for the number of values")
    return n_neighbors_theoretical


"""
    The function is used to generate a UMAP plot for a given set of values and annotations.
    
    :param values: The values to generate the UMAP plot for.
    :param raw_values: The raw values to generate the UMAP plot for.
    :param annotations: The annotations to use for the UMAP plot.
    :param min_dists: The minimum distances to use for the UMAP plot.
    :param n_intervals: The intervals to use for the UMAP plot.
    :param writer: The writer to use for logging.
    :param out_dir: The output directory to save the plots to.
    :param point_size: The size of the points in the plot.
    :param columns: The columns/labels to use for clustering the plot.
    :param include_images: Whether to include images from the embeddings to out_dir. This makes it easier to share the output from the plot
    :param knn: whether to do knn scoring
    :param cpd: whether to do cpd scoring
    :param ss: whether to do silhouette scoring
    
"""


def make_umap(
    values: List[List],
    raw_values: List[List] | None = None,
    labels: DataFrame | None = None,
    min_dists=None,
    n_intervals=None,
    writer=None,
    out_dir=None,
    point_size=3,
    include_images=False,
    knn=False,
    cpd=False,
    do_ss=False,
    render_html=False,
    logger=logger,
    show_plot=True,
    show_plot_title="fig",
    show_plot_step : int = -1,
    use_cuml=False,
):
    start = time.time()
    # check if cuml is imported
    if use_cuml:
        try:
            global _CUML_IMPORTED
            if not _CUML_IMPORTED:
                import cuml
                from cuml.manifold.umap import UMAP as cumlUMAP

                _CUML_IMPORTED = True
        except ImportError:
            _CUML_IMPORTED = False
            logger.warning(
                "Not using GPU for UMAP: unable to import cuml.manifold.umap"
            )

    if min_dists is None or len(min_dists) == 0:
        min_dists = [0.1]
    if n_intervals is None or len(n_intervals) == 0:
        n_intervals = [15]

    run_cfgs = []

    for dist in min_dists:
        for n in n_intervals:
            run_cfgs.append(
                UmapConfig(values, dist, n, UmapMode.PLAIN, knn=knn, cpd=cpd)
            )
            if raw_values is not None:
                run_cfgs.append(UmapConfig(raw_values, dist, n, UmapMode.RAW))

    runners: List[UmapRunner] = []
    zarr_store = zarr.storage.MemoryStore()

    root = zarr.create_group(store=zarr_store)
    for i, cfg in enumerate(run_cfgs):
        runner = UmapRunner(cfg, root, writer=writer, use_cuml=use_cuml)
        runner.run(i)
        runners.append(runner)

    metrics = defaultdict(pd.DataFrame)

    if labels is None or labels.empty:
        labels = pd.DataFrame({ImageLabels.DEFAULT: [0] * len(values)})
    else:
        # reset index to integers
        labels = labels.reset_index()

    columns = labels.columns.tolist()
    if ImageLabels.FILENAME in columns:
        columns.remove(ImageLabels.FILENAME)
    if ImageLabels.IMAGE_URL in columns:
        columns.remove(ImageLabels.IMAGE_URL)

    if labels.columns.isin([ImageLabels.FILENAME, ImageLabels.IMAGE_URL]).all():
        logger.info(
            "not computing cluster metrics: no cluster specific labels in annotations"
        )
    else:
        for i, runner in enumerate(runners):
            for key in columns:
                if isinstance(key, list):
                    raise ValueError("Pandas dataframe contains a list")
                plot_metrics = ClusteringMetrics(key, do_ss=do_ss)(
                    i, runner, labels,
                )
                metrics[key] = pd.concat(
                    [
                        metrics[key],
                        pd.concat([runner.plot.gen_df(), plot_metrics], axis=1),
                    ],
                )

    # now we need to create pairs of runners
    # a pair is formed if the runners have the same configuration except "mode"
    # we can then use these pairs to generate the widgets
    pairs = []
    for i, runner in enumerate(runners):
        # we want to pair a PLAIN umap embedding with others, so if the mode is not plain, skip
        if not runner.config.mode == UmapMode.PLAIN:
            continue
        pair_length_pre = len(pairs)
        for j, runner2 in enumerate(runners):
            if i == j:
                continue
            if (
                runner.config.n == runner2.config.n
                and runner.config.min_dist == runner2.config.min_dist
            ):
                pairs.append((runner, runner2))
                break
        # if there is no pair for the runner, go solo
        if pair_length_pre == len(pairs):
            pairs.append((runner, None))

    if ImageLabels.FILENAME in labels.columns:
        if include_images:
            image_urls = [
                f".{os.sep}" + x for x in labels[[ImageLabels.FILENAME]].values
            ]
        else:
            image_urls = list(labels[[ImageLabels.FILENAME]].values)

        labels = (
            labels.reset_index().assign(image_url=image_urls).reset_index(drop=True)
        )

    frames = []
    if render_html:
        for key in columns:
            metrics_table = make_metrics_table(metrics[key])
            data_source, tooltips, unique_labels = make_datasource(labels, key)
            distribution_plot = make_distribution_plot(data_source, unique_labels)
            widgets = []
            for runner, raw_runner in pairs:
                if writer is None:
                    tracker = lambda l: l
                else:
                    tracker = lambda l: track_method(
                        l, f"make_widget_for_{key}", writer, runner.i
                    )

                widget = tracker(make_umap_widget)(
                    runner.plot.embedding,
                    raw_runner.plot.embedding if raw_runner else None,
                    data_source,
                    tooltips,
                    unique_labels,
                    interactive_text_search_columns=[key],
                    width=800,
                    height=800,
                    point_size=point_size,
                    title=f"{key}: {len(unique_labels)} unique colors",
                )
                widgets.append(widget)

            frames.append(
                (
                    make_plot_slider(
                        widgets,
                        min_dists,
                        n_intervals,
                        metrics_table,
                        distribution_plot,
                    ),
                    key,
                )
            )

        col = make_plot_select([x[0] for x in frames], [x[1] for x in frames])
        explain_box = Div(text=METRIC_HELPTEXT)
        citation_box = Div(text=METRIC_CITETEXT)
        out_widget = column(col, column(explain_box, citation_box))

        if out_dir is None:
            logger.info("Showing UMAP plot in browser...")
            track_method(show, "ue_render_html", writer, 0)(out_widget)
        else:
            html_dst = os.path.join(out_dir, f"umap_inspect_{len(values)}.html")
            i = 1
            # append a number to the filename before the ".html"
            while os.path.exists(html_dst):
                html_dst = os.path.join(out_dir, f"umap_inspect_{len(values)}_{i}.html")
                i += 1
            output_file(html_dst, title="UMAP")
            ensure_dir_exists(os.path.dirname(html_dst))
            track_method(save, "ue_save_html", writer, 0)(out_widget)
            logger.info("Output written to {}".format(html_dst))

            if include_images and ImageLabels.FILENAME in labels.columns:
                logger.info("Copying images to output directory for easier exports...")
                for file in labels[ImageLabels.FILENAME].values:
                    dst = os.path.join(out_dir, file.lstrip(os.sep))
                    ensure_dir_exists(os.path.dirname(dst))
                    if os.path.exists(file) and not os.path.exists(dst):
                        shutil.copy2(file, dst)

    if writer is not None:
        writer.add_scalar("ue_total_sec", time.time() - start)

    if show_plot:
        for i, runner in enumerate(runners):
            embedding = runner.plot.embedding
            for c in columns:
                fig = plt.figure(figsize=(12, 12))
                ll = labels[c].tolist()
                unique_labels = np.unique(ll)
                class_map = {c: i for i, c in enumerate(unique_labels)}
                plt.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=[class_map[d] for d in ll],
                    cmap="Spectral",
                )
                fig_label = (
                    f"umap_{show_plot_title}_"
                    + "_".join(runner.config.lookup_key.split("/")[1:]).replace(
                        " ", "_"
                    )
                    + "_"
                    + c
                )
                if writer:
                    if show_plot_step > -1:
                        writer.add_figure(fig_label, fig, global_step=show_plot_step)
                    else:
                        writer.add_figure(fig_label, fig)
                    logger.debug(f"wrote figure '{fig_label}' to tensorboard")
                else:
                    plt.show()

    if raw_values is not None:
        umap_plots = [(runner.plot, raw_runner.plot) for (runner, raw_runner) in pairs]
    else:
        umap_plots = [runner.plot for runner, _ in pairs]
    logger.info("UMAP plot generation took {} seconds".format(time.time() - start))
    return umap_plots
