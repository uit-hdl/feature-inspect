# feature-inspect

The result of the following paper: ... _pending_ ...


This package is an open-source tool to explore high-level features from images with UMAPs and/or linear-probing.
This is becoming increasingly important as we're now seeing more large-scale models being made. 
How they perform for your task and dataset needs to be evaluated before use.
The main purpose of creating the package is:  
1. to make common guidelines for UMAPs parameters (e.g. from [Kobak and Berens](https://www.nature.com/articles/s41467-019-13056-x)) more accessible.
2. to provide objective metrics (to be used cautiously) for evaluating feature-spaces
3. to create a tool for exploring models that can scale for large inputs (e.g. whole-slide images)

# Installation
```bash
pip install feature_inspect
# optional if you want to use linear probing
pip install feature_inspect[lp_inspect]
```

## GPU acceleration for UMAP
To install the libraries needed for cuml, please use https://docs.rapids.ai/install/ and install the "cuml" and pytorch package using conda. Further, to use the GPU acceleration, pass `use_cuml=True` to `make_umap`

# Usage
Examples are given in the [examples](examples) folder. But a simple example is:

```python
import numpy as np
images = np.random.rand(100, 32, 32, 3)
# .. use a model or clustering method to extract features from the images
# which should be an array of shape (100, N), where N is the number of features
features = [[...]]
from umap_inspect import explore

explore.make_umap(features)

# if you install linear_probe
from lp_inspect import linear_probe

# labels should be a list of strings in the same order as the features
labels = [...]
data = [{"image": f, "label": l} for f, l in zip(features, labels)]
linear_probe.linear_probe(data=data)

```
Performance metrics and detailed results are written using [tensorboard](https://www.tensorflow.org/tensorboard).
you can initialise a writer like this: `from torch.utils.tensorboard import SummaryWriter; writer = SummaryWriter(log_dir="path/to/logdir")`
and pass it to the `make_umap` and `linear_probe` functions.

UMAPs can be rendered to html instead of the most common matplotlib solution.
The UI looks similar to this:
[./figures/umap.png](./figures/umap.png)

## Usage with MONAI
MONAI has some interfaces similar to pytorch-ignite that allows you to create a model with only a few lines of code.
I personally prefer this approach when training models. The following code snippet will attach handlers that evaluate the model using UMAPs and linear-probing on the validation set.
```
from monai_handlers.LinearProbeHandler import LinearProbeHandler
from monai_handlers.UmapHandler import UmapHandler
    val_postprocessing = Compose([EnsureTyped(keys=CommonKeys.PRED)])
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=dl_val,
        network=model,
        val_handlers=[
            UmapHandler(model=model, feature_layer_name=feature_layer_name, umap_dir=out_path, summary_writer=writer,
                        output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL])),
            LinearProbeHandler(model=model, feature_layer_name=feature_layer_name, out_dir=out_path, summary_writer=writer,
                output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL])),
        ],
        key_val_metric={
            "val_acc": Accuracy(output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL]))
        },
        postprocessing=val_postprocessing,
    )
```

# Recreating the results from the paper
First, follow the instructions at [https://github.com/uit-hdl/code-overfit-detection-framework](https://github.com/uit-hdl/code-overfit-detection-framework).
This will produce embeddings in the `out/` folder. Then you can run the following:

```bash
# Creating a fine-tuned phikon model to do disease-classification on TCGA-LUSC
ipython examples/use_case_linear_probe.py -- --embeddings-path out/phikon_TCGA_LUSC-tiles_embedding.zarr/ --label-file out/tcga-tile-annotations.csv --label-key disease --out-dir out_phikon_lp_disease --epochs 20 --batch-size 256

ipython examples/evaluate_lp.py -- --embeddings-path out/phikon_CPTAC-tiles_embedding.zarr/ --label-file out/cptac-tile-annotations.csv --label-key disease --out-dir out_phikon_lp_disease --model-dir out_phikon_lp_disease --tensorboard-name cptac 
```
