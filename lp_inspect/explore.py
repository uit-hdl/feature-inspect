import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader, Dataset
from monai.utils import CommonKeys
from tensorboardX import SummaryWriter

from .data import divide_data, ImageSample
from .model import LinearProbe, train, evaluate_model, plot_distributions
from misc.benchmark import track_method

"""
Train a small linear classifier on top of the embeddings to predict the labels
    :param data: list of dictionaries containing the embeddings and labels. Should look like this:
        [{"image": <1d tensor with data>, "label": <integer>}, ... ]
    :param out_dir: directory to save the model and tensorboard logs
    :param writer: tensorboard writer. Optional
    :param kwargs: additional arguments: batch_size, train_ratio, val_ratio, test_ratio, epochs, lr
"""


def lp_eval(
    data: List[ImageSample],
    out_dir: str = "./out",
    writer: SummaryWriter = None,
    balanced : bool = True,
    device = None,
    **kwargs
):
    batch_size = kwargs.get("batch_size", 64)

    if len(data) < batch_size:
        raise ValueError(
            f"Not enough data for LP training (expected at least {batch_size}, got {len(data)})"
        )

    labels = np.unique([item[CommonKeys.LABEL] for item in data])
    if len(labels) == 1:
        raise ValueError(
            "Only one unique label for predictions, expected > 1"
        )
    class_map = {c: i for i, c in enumerate(labels)}
    # pytorch requires labels to be integers
    for d in data:
        d[CommonKeys.LABEL] = class_map[d[CommonKeys.LABEL]]

    train_ratio=kwargs.get("train_ratio", 0.7)
    val_ratio=kwargs.get("val_ratio", 0.15)
    test_ratio=kwargs.get("test_ratio", 0.15)
    splits = divide_data(
        data,
        balanced=balanced,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        writer=writer,
    )
    balanced_text = "balanced (i.e. equal number of instances per class)" if balanced else ""
    if not splits["train"]:
        raise ValueError(f"After dividing data into {balanced_text} stratifications using {train_ratio*100}% of train data, no data remains")
    if not splits["validation"]:
        raise ValueError(f"After dividing data into {balanced_text} stratifications using {val_ratio*100}% of validation data, no data remains")
    if not splits["test"]:
        raise ValueError(f"After dividing data into {balanced_text} stratifications using {test_ratio*100}% of test data, no data remains")
    dl_train = DataLoader(Dataset(splits["train"]), batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(
        Dataset(splits["validation"]), batch_size=batch_size, shuffle=True
    )
    dl_test = DataLoader(Dataset(splits["test"]), batch_size=batch_size, shuffle=True)

    if writer:
        writer.add_text("lp_class_map", str(class_map))
        writer.add_text("lp_batch_size", str(batch_size))
        writer.add_text("lp_train_ratio", str(0.7))
        writer.add_text("lp_val_ratio", str(0.15))
        writer.add_text("lp_test_ratio", str(0.15))
        writer.add_text("lp_num_images_train", str(len(splits["train"])))
        writer.add_text("lp_num_images_val", str(len(splits["validation"])))
        writer.add_text("lp_num_images_test", str(len(splits["test"])))

        class_map_inv = {v: k for k, v in class_map.items()}
        plot_distributions(
            [x[CommonKeys.LABEL] for x in dl_train.dataset.data],
            "lp_train",
            class_map_inv,
            writer,
            step=0,
        )
        plot_distributions(
            [x[CommonKeys.LABEL] for x in dl_val.dataset.data],
            "lp_val",
            class_map_inv,
            writer,
            step=0,
        )

    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    number_features = len(data[0][CommonKeys.IMAGE])
    model = LinearProbe(number_features, len(class_map)).to(device)

    model_dir = os.path.join(out_dir, "lp_models")
    track_method(train, "lp_train", writer, track_gpu=torch.cuda.is_available())(
        model_dir,
        model,
        dl_train,
        dl_val,
        kwargs.get("epochs", 20),
        device,
        torch.optim.Adam(model.parameters(), lr=kwargs.get("lr", 1e-5)),
        nn.CrossEntropyLoss(),
        writer,
    )

    evaluate_models(dl_test, model_dir, number_features, class_map, device, writer)


def evaluate_models(dl, model_dir, number_features, class_map, device, writer):
    model_paths = [
        (p, int(p.split("=")[-1].split(".")[0]))
        for p in os.listdir(model_dir)
        if p.endswith(".pt")
    ]
    model_paths = sorted(model_paths, key=lambda e: e[1])

    for model_path, epoch_number in model_paths:
        model = LinearProbe(number_features, len(class_map))
        model.load_state_dict(
            torch.load(os.path.join(model_dir, model_path), map_location=device)
        )
        model = model.to(device)

        evaluate_model(
            model,
            dl,
            class_map=class_map,
            device=device,
            writer=writer,
            step=epoch_number,
        )
