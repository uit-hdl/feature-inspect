from fi_misc.global_util import logger
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    count,
    mean_prediction,
    true_positive_rate,
    true_negative_rate,
    false_positive_rate,
    false_negative_rate,
)
from ignite.metrics import Accuracy, Loss
from matplotlib import pyplot as plt
from monai.data import DataLoader
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    StatsHandler,
    from_engine,
    ValidationHandler,
    CheckpointSaver,
    TensorBoardStatsHandler,
)
from monai.inferers import SimpleInferer
from monai.networks import eval_mode
from monai.transforms import Compose, EnsureTyped
from monai.utils import CommonKeys
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from fi_misc.data import ImageLabels
from fi_misc.global_util import ensure_dir_exists, dataframe_to_image


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def train(out_dir, model, dl_train, dl_val, epochs, device, optimizer, loss, writer):
    ensure_dir_exists(out_dir)

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=dl_val,
        network=model,
        val_handlers=[
            CheckpointSaver(
                save_dir=out_dir,
                save_dict={"net": model},
                epoch_level=True,
                save_interval=2 if epochs > 1 else 1,
            ),
            TensorBoardStatsHandler(writer, output_transform=lambda x: x),
        ],
        key_val_metric={
            "lp_val_acc": Accuracy(
                output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL])
            )
        },
        postprocessing=Compose([EnsureTyped(keys=CommonKeys.PRED)]),
    )

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=epochs,
        train_data_loader=dl_train,
        network=model,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        key_train_metric={
            "lp_train_acc": Accuracy(
                output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL])
            )
        },
        additional_metrics={
            "lp_train_loss": Loss(
                output_transform=from_engine(
                    [CommonKeys.PRED, CommonKeys.LABEL], first=False
                ),
                loss_fn=loss,
            )
        },
        train_handlers=[
            StatsHandler(
                tag_name="lp_train_loss",
                output_transform=from_engine([CommonKeys.LOSS], first=True),
            ),
            ValidationHandler(1, evaluator),
            TensorBoardStatsHandler(writer, output_transform=lambda x: x),
        ],
    )

    trainer.run()


def evaluate_model(
    model, dl_test: DataLoader, class_map, device="cpu", writer=None, step=0
):
    predictions = np.array([])
    gts = np.array([])
    logger.info("Evaluating model")
    wrong_predictions = defaultdict(list)
    class_map_inv = {v: k for k, v in class_map.items()}
    with eval_mode(model):
        for item in tqdm(dl_test):
            y = model(item[CommonKeys.IMAGE].to(device))
            prob = F.softmax(y).detach().to("cpu")
            pred = torch.argmax(prob, dim=1).numpy()

            predictions = np.append(predictions, pred)

            gt = item[CommonKeys.LABEL].detach().cpu().numpy()
            gts = np.append(gts, gt)
            for i, (p, g) in enumerate(zip(pred, gt)):
                if p != g:
                    if ImageLabels.FILENAME in item:
                        wrong_predictions[class_map_inv[g]].append(
                            (item[ImageLabels.FILENAME][i], g, p)
                        )
                    else:
                        wrong_predictions[class_map_inv[g]].append((None, g, p))

    metrics_dict = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "count": count,
        "mean_pred": mean_prediction,
    }
    if len(np.unique(gts)) == 2:
        metrics_dict["tp_rate"] = true_positive_rate
        metrics_dict["tn_rate"] = true_negative_rate
        metrics_dict["fp_rate"] = false_positive_rate
        metrics_dict["fn_rate"] = false_negative_rate
    mf = MetricFrame(
        metrics=metrics_dict,
        y_true=gts,
        y_pred=predictions,
        sensitive_features=list(class_map_inv[x] for x in gts),
    )
    logger.info(mf.overall)
    if writer:
        writer.add_scalar("lp_test_acc", mf.overall["accuracy"], global_step=step)

    i = 1
    if wrong_predictions:
        if ImageLabels.FILENAME in item:
            grid_size = min(10, min(map(lambda l: len(l), wrong_predictions.values())))
            fig, axes = plt.subplots(
                nrows=len(wrong_predictions),
                ncols=grid_size,
                figsize=(16, 16),
                sharex=True,
                sharey=True,
            )
            plt.setp(axes, xticks=[], yticks=[])
            fontsize = 6
            for row, w in enumerate(wrong_predictions):
                if len(wrong_predictions) == 1 or grid_size == 1:
                    axes[row].set_ylabel(f"GT: {w}", fontsize=fontsize)
                else:
                    axes[row][0].set_ylabel(f"GT: {w}", fontsize=fontsize)
                for col in range(grid_size):
                    image_filename, g, p = wrong_predictions[w][col]
                    if len(wrong_predictions) == 1 or grid_size == 1:
                        axes[col].imshow(plt.imread(image_filename))
                        axes[col].set_title(
                            "pred: {}".format(class_map_inv[p]), fontsize=fontsize
                        )
                    else:
                        axes[row][col].imshow(plt.imread(image_filename))
                        axes[row][col].set_title("pred: {}".format(class_map_inv[p]))
                    i += 1
        else:
            fig, ax = plt.subplots()
            ax.text(
                0.05,
                0.5,
                "No image filenames available to show wrong predictions",
                fontsize=12,
            )
    else:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No wrong predictions", fontsize=12, ha="center")

    if writer:
        writer.add_figure("lp_wrong_predictions", fig, global_step=step)
        df = mf.by_group
        df = df.round(3)
        writer.add_image(
            "lp_fairness_metrics",
            dataframe_to_image(df),
            global_step=step,
            dataformats="HWC",
        )
    else:
        plt.show()

    labels = list(class_map_inv.values())
    plot_results(
        list(class_map_inv[x] for x in gts),
        list(class_map_inv[x] for x in predictions),
        labels,
        "test_acc",
        writer=writer,
        step=step,
    )

    plot_distributions(
        [x[CommonKeys.LABEL] for x in dl_test.dataset.data],
        "test",
        class_map_inv,
        writer,
        step=step,
    )


def plot_results(gts, predictions, labels, title, writer=None, step=0):
    cm = confusion_matrix(gts, predictions, labels=labels)
    correct_classifications = sum([cm[i][i] for i in range(len(labels))])
    wrong_classifications = len(gts) - correct_classifications
    total = correct_classifications + wrong_classifications
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cmd = disp.plot(ax=plt.subplots(1, 1, facecolor="white")[1])
    fig = cmd.ax_.get_figure()
    fig.set_figwidth(20)
    fig.set_figheight(20)

    if writer:
        writer.add_scalar(
            "lp_" + title, correct_classifications / total, global_step=step
        )
        writer.add_figure(f"lp_confusion_matrix_{title}", cmd.figure_, global_step=step)
    else:
        logger.info(f"lp_{title}: {correct_classifications / total}")
        plt.show()


def plot_distributions(data, mode, class_map, writer, step=0):
    count_per_label = pd.Series(data).value_counts()
    # convert the labels back to their original names (from integers)
    count_per_label.index = [class_map[x] for x in count_per_label.index]

    fig, ax = plt.subplots()
    ax.pie(count_per_label, labels=count_per_label.index, autopct="%1.1f%%")

    if writer:
        writer.add_figure(f"lp_label_distribution_{mode}", fig, global_step=step)
