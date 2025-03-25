import glob
from typing import TypeAlias

from fi_misc.global_util import logger

import os
import random
from collections import defaultdict

import pandas as pd
from monai.utils import CommonKeys
import torch

from fi_misc.global_util import ensure_dir_exists

ImageSample: TypeAlias = dict[str, torch.Tensor | int]


def divide_data(
    files,
    balanced=True,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    writer = None,
):
    if not files:
        return {"train": [], "validation": [], "test": []}

    n = len(files)
    if balanced:
        # Group data by labels
        label_to_data = defaultdict(list)
        for item in files:
            label_to_data[item[CommonKeys.LABEL]].append(item)

        num_labels = len(label_to_data.keys())
        max_samples_per_label = min(len(items) for items in label_to_data.values())
        num_train = int(max_samples_per_label * train_ratio)
        num_val = int(max_samples_per_label * val_ratio)
        num_test = int(max_samples_per_label * test_ratio)
        if writer:
            writer.add_scalar("lp_samples_per_class_train", num_train)
            writer.add_scalar("lp_samples_per_class_val", num_val)
            writer.add_scalar("lp_samples_per_class_test", num_test)

        train_data, val_data, test_data = [], [], []

        for items in label_to_data.values():
            random.shuffle(items)

            train_data.extend(items[:num_train])
            val_data.extend(items[num_train : num_train + num_val])
            test_data.extend(
                items[num_train + num_val : num_train + num_val + num_test]
            )

        number_of_samples = len(train_data) - len(val_data) - len(test_data)
        pruned_entries = n - number_of_samples
        logger.info(
            f"Balanced data with {max_samples_per_label} samples per label - this prunes away {pruned_entries} of {n} entries for {len(label_to_data)} labels"
        )
        return {"train": train_data, "validation": val_data, "test": test_data}
    else:
        num_train = int(train_ratio * n)
        num_val = int(val_ratio * n)
        num_test = len(files) - num_train - num_val
        if writer:
            writer.add_scalar("lp_samples_per_class_train", num_train)
            writer.add_scalar("lp_samples_per_class_val", num_val)
            writer.add_scalar("lp_samples_per_class_test", num_test)
        return {
            "train": files[num_train:],
            "validation": files[num_train:num_train+num_val],
            "test": files[num_train+num_val:],
        }
