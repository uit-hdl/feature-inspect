# coding: utf-8
import logging
import os

import numpy as np
from PIL import Image
import zarr.storage
from tqdm import tqdm

from misc.data import ImageLabels
from misc.global_util import ensure_dir_exists
from misc.global_util import logger


def find_all_arrays(group):
    """
    Recursively iterate over a Zarr group to find all arrays.
    
    Parameters:
        group (zarr.Group): The current Zarr group.
        prefix (str): The prefix path for the current group.
    
    Yields:
        tuple: (full_path, array) for each Zarr array found.
    """
    groups_to_visit = [(group, "")]
    i = 0
    arrays = []
    names = []
    while True:
        if i >= len(groups_to_visit):
            break
        group,prefix = groups_to_visit[i]
        for name, member in group.groups():
            full_path = f"{prefix}/{name}"
            
            if isinstance(member, zarr.Group):
                groups_to_visit.append((member, full_path))
            for array_name, array in member.arrays():
                arrays.append(array)
                names.append(f"{full_path}/{array_name}")
        i += 1
    return (names, arrays)
            

def load_zarr_store(store_path):
    import zarr

    if not os.path.exists(store_path):
        raise ValueError(f"Store path {store_path} does not exist")

    with zarr.storage.LocalStore(store_path) as store:
        root = zarr.open_group(store)
        logger.info(f"Loading embeddings from {store_path}")
        names, features = find_all_arrays(root)
        if not names or not features:
            raise ValueError(f"No embeddings found in {store_path}")

        logger.info(f"Finished loading embeddings")
        logger.info(root.info)
        all_data = [{"image": f, ImageLabels.FILENAME: n} for f, n in zip(features, names)]
        return all_data

def extract_features(image):
    hist, bin_edges = np.histogram(np.array(image), bins=8, range=(0, 256))
    hist = hist.astype(np.float32) / hist.sum()
    return hist

def generate_random_embeddings(temp_dir, num_images=100):
    images, features, values = [], [], []
    ensure_dir_exists(os.path.join(temp_dir, "images"))

    # Generate `num_images` random images
    for i in range(num_images):
        # Create a random noise image (e.g., 64x64 pixels)
        random_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(random_image, 'RGB')

        image_path = os.path.join(temp_dir, "images", f"random_image_{i+1}.png")
        ensure_dir_exists(os.path.dirname(image_path))
        img.save(image_path)
        images.append(image_path)
        values.append(np.asarray(img).flatten().astype(np.float32))
        features.append(extract_features(img))
    logging.info(f"Generated {num_images} random embeddings")
    return images, features, values

