import glob
from typing import TypeAlias

from misc.global_util import logger

import os
import random
from collections import defaultdict

import pandas as pd
from monai.utils import CommonKeys
import torch

from misc.global_util import ensure_dir_exists

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

        max_samples_per_label = min(len(items) for items in label_to_data.values())
        num_train = int(max_samples_per_label * train_ratio)
        num_val = int(max_samples_per_label * val_ratio)
        num_test = int(max_samples_per_label * test_ratio)
        if writer:
            writer.add_text("lp_samples_per_class_train", str(num_train))
            writer.add_text("lp_samples_per_class_val", str(num_val))
            writer.add_text("lp_samples_per_class_test", str(num_test))

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
        return {
            "train": files[: int(train_ratio * n)],
            "validation": files[int(train_ratio * n) : int(1 - val_ratio * n)],
            "test": files[int(1 - test_ratio * n) :],
        }


def build_file_list(data_dir, file_list_path, labels, label_key=None, balanced=False):
    if not os.path.exists(file_list_path):
        logger.info(
            "File list not found. Creating file list in {}".format(file_list_path)
        )
        all_data = []
        for filename in glob.glob(f"{data_dir}{os.sep}**", recursive=True):
            if os.path.isfile(filename) and filename.endswith(
                (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            ):
                if labels is not None and not labels.empty and filename in labels.index:
                    all_data.append(
                        {"filename": filename, "label": labels.loc[filename][label_key]}
                    )
                else:
                    all_data.append({"filename": filename, "label": "unknown"})

        if labels is not None:
            splits = divide_data(all_data, balanced)
        else:
            splits = divide_data(all_data, balanced=False)

        if not splits["train"]:
            raise RuntimeError(f"Found no data in {data_dir}")

        train_data = pd.DataFrame(splits["train"])
        train_data["mode"] = "train"
        val_data = pd.DataFrame(splits["validation"])
        val_data["mode"] = "validation"
        test_data = pd.DataFrame(splits["test"])
        test_data["mode"] = "test"

        ensure_dir_exists(file_list_path)
        all_data = pd.concat([train_data, val_data, test_data])
        all_data.to_csv(file_list_path, index=False)

    all_data = pd.read_csv(file_list_path)
    train_data = all_data[all_data["mode"] == "train"].to_dict(orient="records")
    val_data = all_data[all_data["mode"] == "validation"].to_dict(orient="records")
    test_data = all_data[all_data["mode"] == "test"].to_dict(orient="records")
    for li in [train_data, val_data, test_data]:
        for item in li:
            item.pop("mode")
    logger.info(f"Loaded file list from {file_list_path}")

    train_data, val_data, test_data = [
        [
            {
                CommonKeys.IMAGE: entry["filename"],
                "filename": entry["filename"],
                CommonKeys.LABEL: entry[CommonKeys.LABEL],
            }
            for entry in data_list
        ]
        for data_list in [train_data, val_data, test_data]
    ]

    return train_data, val_data, test_data
