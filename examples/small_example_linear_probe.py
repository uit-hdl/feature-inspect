#!/usr/bin/env python
# coding: utf-8
import logging
import os
import sys
import tempfile
from time import sleep

import numpy as np
from monai.utils import CommonKeys

from examples.example_data import generate_random_embeddings
from lp_inspect import lp_eval
from fi_misc.global_util import init_tb_writer

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    with tempfile.TemporaryDirectory() as out_dir:
        names, features, values = generate_random_embeddings(out_dir, num_images=500)
        random_labels = np.random.randint(0, 2, size=(len(names))).tolist()

        # tensorboard will contain all outputs
        # tensorboards can be viewed using `tensorboard` command on the terminal
        # or easily loaded as a pandas dataframe
        tb_dir = os.path.join(out_dir, "tb_logs")
        writer = init_tb_writer(os.path.join(out_dir, "tb_logs"), "small_example_test")

        data = [{CommonKeys.IMAGE: f, CommonKeys.LABEL: l} for f, l in zip(features, random_labels)]
        lp_eval(
            data=data,
            out_dir=out_dir,
            writer=writer,
            epochs=10,
            batch_size=256,
            lr=1e-05)
        logging.info("Done. Entering sleep loop")
        logging.info("While the example is running, you can view outputs in Tensorboard with the following command:"
                     f"\ntensorboard --logdir {tb_dir}")
        while True:
            sleep(1)
