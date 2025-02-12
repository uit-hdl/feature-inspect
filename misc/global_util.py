import logging
import os
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter


def __get_username():
    if os.environ.get("USER"):
        return os.environ.get("USER")

    login = None
    # this will typically fail in docker
    try:
        login = os.getlogin()
    except OSError:
        pass

    if not login or not login.isalnum():
        login = "unnamed"

    return login


def __get_logger():
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # Set to DEBUG to match config
    console_handler.setFormatter(formatter)

    feature_inspect_logger = logging.getLogger("feature_inspect")
    feature_inspect_logger.setLevel(logging.DEBUG)
    feature_inspect_logger.addHandler(console_handler)
    feature_inspect_logger.propagate = False  # To match propagate=0 in config
    return feature_inspect_logger


logger = __get_logger()


def init_tb_writer(tb_dir, tb_name, extra=None):
    # 1. get name for tensorboard dst dir. trying to include username since that will ensure
    # that multiple people on one server won't cause write errors
    user = __get_username()
    tb_dir = tb_dir or os.path.join(tempfile.gettempdir(), f"tb_{user}")
    tb_name = tb_name or str(time.time())
    tb_dst = os.path.join(tb_dir, tb_name)

    writer = SummaryWriter(log_dir=tb_dst)
    logger.info(
        f"Writing tensorboard stats to '{tb_dst}' (inspect with `tensorboard --logdir={tb_dst}`)"
    )

    if extra:
        for key, val in extra.items():
            writer.add_text(key, str(val))

    return writer


def ensure_dir_exists(path):
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"mkdir: '{path}'")


def dataframe_to_image(df):
    fig, ax = plt.subplots(figsize=(6, 2))  # Adjust the figure size as needed
    ax.axis("tight")
    ax.axis("off")
    # turn the index of df into a column
    df.reset_index(inplace=True)
    table = ax.table(
        cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))
