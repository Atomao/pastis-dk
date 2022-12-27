import os

import geopandas as gpd
import numpy as np
import pandas as pd
import tensorflow as tf


def get_files(path_to_dataset):
    metadata = gpd.read_file(os.path.join(path_to_dataset, "metadata.geojson"))
    norms = pd.read_json(os.path.join(path_to_dataset, "NORM_S2_patch.json")).to_dict()
    return metadata, norms


def get_slices(metadata, path_to_dataset, folds=[1, 2, 3]):
    s2_images = (
        metadata[metadata["Fold"].isin(folds)]["ID_PATCH"]
        .astype(str)
        .apply(lambda x: os.path.join(path_to_dataset, f"DATA_S2/S2_{x}.npy"))
        .values
    )
    target_mask = (
        metadata[metadata["Fold"].isin(folds)]["ID_PATCH"]
        .astype(str)
        .apply(lambda x: os.path.join(path_to_dataset, f"ANNOTATIONS/TARGET_{x}.npy"))
        .values
    )
    return s2_images, target_mask


def calculate_norm_mean(norms, train_folds=[1, 2, 3]):
    sum_mean = np.zeros(10)
    sum_std = np.zeros(10)
    for fold in norms:
        if fold in [f"Fold_{i}" for i in train_folds]:
            for stat in norms[fold]:
                if stat == "std":
                    sum_std += np.array(norms[fold][stat])
                else:
                    sum_mean += np.array(norms[fold][stat])

    normalize_mean = sum_mean[None, None, None, :] / len(train_folds)
    normalize_std = sum_std[None, None, None, :] / len(train_folds)
    return tf.convert_to_tensor(normalize_mean, dtype=tf.float32), tf.convert_to_tensor(
        normalize_std, dtype=tf.float32
    )


def normalize(x, mean, std):
    return (x - mean) / std


def load_npy(path):
    image = np.load(path)
    return image.astype(np.float32)


load_file = lambda path: tf.numpy_function(load_npy, [path], [tf.float32])


def preprocess(folder_path, train_folds=[1, 2, 3]):
    _, norms = get_files(folder_path)
    mean, std = calculate_norm_mean(norms=norms, train_folds=train_folds)

    def preprocess_(x, y):
        x = load_file(x)[0]
        y = load_file(y)[0][0]
        x = tf.transpose(x, perm=(0, 2, 3, 1))
        x = normalize(x, mean=mean, std=std)
        return x, y

    return preprocess_
