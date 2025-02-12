import random
from collections import defaultdict

import numpy as np
import scipy
from scipy.stats import spearmanr, skew
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors


def compute_knn(high_dimensional_points, low_dimensional_points, k=10):
    if len(high_dimensional_points) < k:
        k = len(high_dimensional_points)
    nbrs_high = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(
        high_dimensional_points
    )
    nbrs_low = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(
        low_dimensional_points
    )

    distances_high, indices_high = nbrs_high.kneighbors(high_dimensional_points)
    distances_low, indices_low = nbrs_low.kneighbors(low_dimensional_points)
    # Calculate how many elements in b that is in a
    knn_frac = lambda s: len(set(s[0]).intersection(s[1])) / k
    knn_fractions = list(map(knn_frac, zip(indices_low, indices_high)))

    return knn_fractions


def compute_knc(high_dimensional_points, low_dimensional_points, cluster_labels, k=10):
    nbrs_high = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(
        high_dimensional_points
    )
    nbrs_low = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(
        low_dimensional_points
    )

    distances_high, indices_high = nbrs_high.kneighbors(high_dimensional_points)
    distances_low, indices_low = nbrs_low.kneighbors(low_dimensional_points)
    cluster_map = lambda x: cluster_labels[x]
    class_mapping_high = [list(map(cluster_map, li)) for li in indices_high]
    class_mapping_low = [list(map(cluster_map, li)) for li in indices_low]

    # Calculate how many elements in b that is in a
    knn_frac = lambda s: len(set(s[0]).intersection(s[1])) / k
    knn_fractions = list(map(knn_frac, zip(class_mapping_high, class_mapping_low)))

    return knn_fractions


def compute_cpd(high_dimensional_points, low_dimensional_points, sample_size=1000):
    points = random.sample(
        range(len(high_dimensional_points)),
        (
            sample_size
            if len(high_dimensional_points) > sample_size
            else len(high_dimensional_points)
        ),
    )
    dists_high = [0] * (len(points) * (len(points) - 1))
    dists_low = [0] * (len(points) * (len(points) - 1))
    index = 0
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i == j:
                continue
            dists_high[index] = distance.euclidean(
                high_dimensional_points[i], high_dimensional_points[j]
            )
            dists_low[index] = distance.euclidean(
                low_dimensional_points[i], low_dimensional_points[j]
            )
            index += 1
    return spearmanr(dists_high, dists_low)
