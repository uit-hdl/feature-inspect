#!/usr/bin/env python3
# coding: utf-8
import logging
import tempfile
from time import sleep

import numpy as np
import pandas as pd

from examples.example_data import generate_random_embeddings
from umap_inspect import make_umap
from umap_inspect.constants import ImageLabels


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        names, features, values = generate_random_embeddings(temp_dir)

        #explore.make_umap(features)

        # if you want to see the images in the output, we need to add filenames in the annotations:
        annotations = pd.DataFrame(data=names, columns=[ImageLabels.FILENAME])
        #explore.make_umap(features, min_dists=[0.1], annotations=annotations)

        # (the images may not render unless you view it in a browser from the .html file):
        # explore.make_umap(features, annotations=annotations, render_html=True, out_dir=".")

        # if you want to side by side with the raw image data:
        # explore.make_umap(features, raw_values=values, annotations=annotations,
        #                   min_dists=[0.1, 0.9],
        #                   n_intervals=[2, 15, 25])#, out_dir="out")

        # if you want metrics:
        from misc.global_util import init_tb_writer
        writer = init_tb_writer(temp_dir, "tensorboard_stats")
        # generate 0 or 1 labels for the images
        annotations["label"] = np.random.randint(0, 2, size=(len(names))).tolist()
        make_umap(features,
                          raw_values=values,
                          labels=annotations,
                          min_dists=[0.1, 0.9],
                          n_intervals=[2, 15, 25],
                          knn=False,
                          cpd=False,
                          do_ss=False,
                          writer=writer,
                          )#, out_dir="out")

        # the temporary image files are deleted upon exiting the code, so we enter a loop here to keep the images around
        logging.info("While the example is running, you can view outputs in Tensorboard with the following command:"
                     f"\ntensorboard --logdir {temp_dir}")
        while True:
            sleep(100)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
