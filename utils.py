import os
import pathlib
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

CWD = Path(os.getcwd())


def save_hist(history, filename):
    pd.DataFrame.from_dict(history.history).to_csv(filename, index=False)


def scheduler(epoch, lr):
    if epoch == 15:

        return lr * 0.1
    return lr


class Utils:
    def __init__(self) -> None:
        self.mean = tf.convert_to_tensor(
            np.load(CWD / "static" / "utils" / "preprocessing" / "mean.npy")
        )
        self.std = tf.convert_to_tensor(
            np.load(CWD / "static" / "utils" / "preprocessing" / "std.npy")
        )
        self.cmap, self.norm = self.get_cmap_norm()

    def array_to_mask(self, array):
        image = np.argmax(array[0], axis=-1)
        image8 = ((image / 19) * 255).astype(np.uint8)
        image8 = Image.fromarray(image8)
        return image8, image

    def predict_mask(self, model, filepath):
        input = self.prepare_input(filepath)
        preds = model(input).numpy()
        gray, image = self.array_to_mask(preds)
        print(np.unique(image))
        image_path = CWD / "static" / "output" / "image" / (str(filepath.stem) + ".png")
        mask_path = CWD / "static" / "output" / "mask" / (str(filepath.stem) + ".png")
        if not image_path.parent.exists():
            image_path.parent.mkdir(parents=True)
        if not mask_path.parent.exists():
            mask_path.parent.mkdir(parents=True)
        gray.save(mask_path)
        plt.imsave(image_path, self.cmap(self.norm(image)))
        return image_path, mask_path

    def s2_image(self, filepath):
        image = np.load

        (filepath)
        image = np.transpose(image, axes=(0, 2, 3, 1))
        s2_path = CWD / "static" / "input" / "image" / (str(filepath.stem) + ".png")
        if not s2_path.parent.exists():
            s2_path.parent.mkdir(parents=True)
        plt.imsave(s2_path, self.get_rgb_np(image))
        return s2_path

    def get_rgb_np(self, x, t_show=0):
        """Utility function to get a displayable rgb image
        from a Sentinel-2 time series.
        """
        im = x[t_show, :, :, [2, 1, 0]]
        mx = im.max(axis=(1, 2))
        mi = im.min(axis=(1, 2))
        im = (im - mi[:, None, None]) / (mx - mi)[:, None, None]
        im = np.transpose(im, (1, 2, 0))
        im = np.clip(im, a_max=1, a_min=0)
        return im

    def get_cmap_norm(self):
        cm = matplotlib.cm.get_cmap("tab20")
        def_colors = cm.colors
        cus_colors = ["k"] + [def_colors[i] for i in range(1, 19)] + ["w"]
        cmap = matplotlib.colors.ListedColormap(colors=cus_colors, name="agri", N=20)
        return cmap, plt.Normalize(vmin=0, vmax=19)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def prepare_input(self, filepath):
        image = tf.convert_to_tensor(np.load(filepath), dtype=tf.float32)
        image = tf.transpose(image, perm=(0, 2, 3, 1))
        image = self.normalize(image)
        image = data = tf.pad(image, [[0, 61 - image.shape[0]], [0, 0], [0, 0], [0, 0]])
        return image[None]
