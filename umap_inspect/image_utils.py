import cv2
import numpy as np
from tqdm import tqdm


def load_and_preprocess_image(img_path, target_size=(64, 64)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image {img_path} not found.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return img


def get_raw_features(filenames):
    raw_features = []
    for file in tqdm(filenames, desc="Loading raw images"):
        img = load_and_preprocess_image(file)
        feature = img.flatten()
        raw_features.append(feature)
    return np.array(raw_features)
