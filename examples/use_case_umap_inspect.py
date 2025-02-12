#!/usr/bin/env python3
# coding: utf-8
'''
This script will read embeddings and create UMAP plots.
It assumes that there is a label file with CSV data that has a format like this:
filename,label1,label2,...
...

It also assumes that the embeddings are stored in a zarr file, where the keys are the filenames and values are stored as arrays.

The output is a set of UMAP plots, with the images displayed on the plot.

Example:
  python examples/use_case_umap_inspect.py --embeddings-path out/phikon_TCGA_LUSC-tiles_embedding.zarr \
    --label-file out/top5TSSfiles.csv \
    --out-dir out \
    --tensorboard-name tb_out \
    --write-output
'''
import argparse
import logging
import os
import random
import shutil
import sys
from collections import defaultdict

import pandas as pd
import zarr.storage
from xdg_base_dirs import xdg_cache_home, xdg_data_home

from example_data import load_zarr_store
from misc.data import ImageLabels
from misc.global_util import ensure_dir_exists, init_tb_writer
from misc.global_util import logger
from umap_inspect.explore import get_umap_neighbors_intervals, make_umap
from umap_inspect.image_utils import get_raw_features


def parse_args():
    parser = argparse.ArgumentParser(description='Convert embeddings to UMAP plots')

    parser.add_argument('--embeddings-path', default=os.path.join('out', 'phikon_TCGA_LUSC-tiles_embedding.zarr'), type=str,
                        help="location of embeddings stored in a zarr file/directory, which should have data in zarr arrays")
    parser.add_argument('--number-of-images', default=10000, type=int, help="how many images to use for each UMAP plot")
    parser.add_argument('--dist-intervals', nargs='+', type=float, default=[0.1, 0.5, 0.9], help="interval of min_distance for UMAP, e.g. [0.1, 0.5, 0.9]")
    parser.add_argument('--label-file', default=None, type=str,
                        help='location of file containing meta-data for images, if available. Should be of format "filename,label1,label2,..."')
    parser.add_argument('--columns', nargs='+', default=[], type=str,
                        help='Columns to use from the label file. If not provided, all columns will be used')
    parser.add_argument('--randomized-select', default=True, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='R', help='Whether to sample images randomly or in order', dest='randomized_select')
    parser.add_argument( "--tensorboard-name", required=False, type=str, help="Name of the run, used for tracking stats. e.g. 'testing_no_cache', etc, or leave blank")
    parser.add_argument('--out-dir', default='out', type=str)
    parser.add_argument('--debug-mode', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='D', help='turn debugging on or off. Will limit amount of data used. Development only',
                        dest='debug_mode')
    parser.add_argument('--knn', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help='whether to do knn or not')
    parser.add_argument('--cpd', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help='whether to do cpd or not')
    parser.add_argument('--ss', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help='whether to do silhouette scoring or not')
    parser.add_argument('--dbcv', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help='whether to do dbcv scoring or not')
    parser.add_argument('--write-output', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help='whether to write a html file or not')
    return parser.parse_args()

def main():
    args = parse_args()

    ensure_dir_exists(args.out_dir)
    ensure_dir_exists(xdg_data_home())  # for tensorboard logs

    data = load_zarr_store(args.embeddings_path)

    if args.label_file:
        tile_annotations = pd.read_csv(args.label_file, dtype=defaultdict(lambda: str)) \
            .set_index(ImageLabels.FILENAME) \
            .sort_index()
        dlen = len(data)
        data = list(filter(lambda d: d[ImageLabels.FILENAME] in tile_annotations.index, data))
        if not data:
            logger.error(f"No data match between label file {args.label_file} and embeddings in {args.embeddings_path}: exiting")
            return
        if args.columns:
            tile_annotations = tile_annotations[args.columns]
        logger.info("Only keeping data that is also in the label file: pruned from {} to {} entries".format(dlen, len(data)))
    else:
        tile_annotations = pd.DataFrame({ImageLabels.FILENAME: [d[ImageLabels.FILENAME] for d in data], 'class_name': 'unknown'})
        tile_annotations.set_index(ImageLabels.FILENAME, inplace=True)
        tile_annotations = tile_annotations.sort_index()

    # limit to at most args.number_of_images samples at a time
    splits = list(range(0, len(data), args.number_of_images))
    splits.append(len(data))

    names = [d[ImageLabels.FILENAME] for d in data]
    chunks = [(data[i:j], tile_annotations.loc[names[i:j]]) for i, j in zip(splits, splits[1:])]
    tb_name = args.tensorboard_name or "default"

    for i, (data_chnk, annotations) in enumerate(chunks):
        number_of_images = len(data_chnk)
        logger.info("Making UMAP with {} data points..".format(number_of_images))
        filenames = [d[ImageLabels.FILENAME] for d in data_chnk]
        raw_values = get_raw_features(filenames)
        writer = init_tb_writer(os.path.join(args.out_dir, "tb_logs_umap"), f"{tb_name}_{i}", extra={})

        n_intervals = get_umap_neighbors_intervals(number_of_images, writer, i)

        image_urls = [f".{os.sep}" + x for x in filenames]

        annotations = annotations \
            .reset_index() \
            .rename(columns={'index': ImageLabels.FILENAME}) \
            .assign(image_url=image_urls) \
            .reset_index(drop=True)

        # for key in annotations.columns:
        #     if key in [ImageLabels.FILENAME, ImageLabels.IMAGE_URL]:
        #         continue
        #     unique_labels = annotations[key].unique()
        #
        #     #  n_neighbors should be a function of the number of images and the number of unique labels
        #     # ideally we would compute this only once per annotation key,
        #     # but for now we will compute it for each chunk in an effort to make code easier to read
        #     n_neighbors_theoretical = compute_optimal_n_neighbors(number_of_images, len(unique_labels))
        #     if n_neighbors_theoretical not in n_intervals and number_of_images > n_neighbors_theoretical > 1:
        #         n_intervals.append(n_neighbors_theoretical)

        make_umap([x["image"] for x in data_chnk],
                  raw_values,
                  annotations,
                  min_dists=args.dist_intervals,
                  n_intervals=n_intervals,
                  writer=writer,
                  knn=args.knn,
                  cpd=args.cpd,
                  do_ss=args.ss,
                  render_html=args.write_output,
                  out_dir=args.out_dir)

        # copy images to output directory, which makes it easier to export the output
        for file in filenames:
            dst = os.path.join(args.out_dir, file.lstrip(os.sep))
            ensure_dir_exists(os.path.dirname(dst))
            if os.path.exists(file) and not os.path.exists(dst):
                shutil.copy2(file, dst)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()
