# umap-explore

A package to explore high-level features with UMAPs. 

The purpose is:
1) to help the community better understand the impact of different UMAP parameters such as `min_dist` and `n_neighbors`.
2) to help users understand whether their model/clustering is _actually_ clustering anything or just exploiting simple image properties like brightness or color.
3) To provide common utilities useful for UMAPs, such as caching, which makes it faster to explore different parameters.

This package relies on python only. It exports data to a web/html file that can be shared with others. You can download the plots as .png files from the web view.

## Installation
```bash
pip install git+https://github.com/uit-hdl/umap-explore.git
```

if you want to develop or modify the package:
```bash
git clone https://github.com/uit-hdl/umap-explore.git
cd umap-explore
pip install --editable .
```

# Usage

```python
images = np.random.rand(100, 32, 32, 3)
# .. use a model or clustering method to extract features from the images
# which should be an array of shape (100, N), where N is the number of features
features = [[...]]
from umap_inspect import explore

explore.make_umap(features)
```

See [examples/umap_explore.ipynb](examples/umap_explore.ipynb) for more use cases.  
See [examples/explore_tcga.ipynb](examples/explore_tcga.ipynb) for a more complex example with cancer data from TCGA, where we use a model to extract features.

### Benchmarking/Statistics
If you want to see stats from the tensorboards, you can attach a tensorboard-writer.
Initialize the tensorboard-writer however you like, or use the one provided in the package.

```python
from umap_inspect import image_utils

writer = utils.init_tb_writer("path/to/tensorboard/logs", "name_of_run")
explore.make_umap(features, writer=writer)
```
You may now view the stats in tensorboard by running `tensorboard --logdir=path/to/tensorboard/logs` 
(or ask chatGPT to convert the tensorboard to a pandas dataframe and do your own analysis)

# License
See [LICENSE](LICENSE).

# Future work
- [ ] Add support for GPU acceleration. See https://umap-learn.readthedocs.io/en/latest/faq.html#is-there-gpu-or-multicore-cpu-support