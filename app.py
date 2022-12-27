import datetime as dt
import os
from pathlib import Path

import numpy as np
from flask import Flask, redirect, render_template, request, url_for
from sentinelhub import SHConfig
from tifffile import imread as tiff_imread

from backbones.unet3d import get_model
from sentinel import Sentinel
from utils import Utils

CWD = Path(os.getcwd())


config = SHConfig()
config.instance_id = "<PUT YOUR CREDENTIALS HERE>"
config.sh_client_id = "<PUT YOUR CREDENTIALS HERE>"
config.sh_client_secret = "<PUT YOUR CREDENTIALS HERE>"
config.save()

app = Flask(__name__)


model = get_model(
    num_classes=20,
    feats=8,
    weights_path=CWD / "static" / "utils" / "weights" / "LastMRadam.h5",
)


satelite = Sentinel(config, CWD / "static" / "utils" / "sentinel" / "delta.csv")


utils = Utils()


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        files = request.files.getlist("my_image")

        if len(files) == 1:
            file = files[0]
            source_path = CWD / "static" / "input" / "numpy/" / file.filename
            if not source_path.parent.exists():
                source_path.parent.mkdir(parents=True)
            file.save(source_path)
        elif len(files) < 33:
            return render_template("index.html", exception="Not enogh Images")
        else:
            images = []
            for file in files:
                img = tiff_imread(file)[None]
                images.append(img)
            images = np.concatenate(images, axis=0)
            source_path = (
                CWD
                / "static"
                / "input"
                / "numpy/"
                / (Path(files[0].filename).stem + ".npy")
            )
            np.save(source_path, images)

        s2_path = utils.s2_image(source_path)
        mask_img_path, mask_gray_path = utils.predict_mask(model, source_path)
        return render_template(
            "index.html",
            img_path=s2_path,
            output_path=mask_img_path,
            mask_gray_path=mask_gray_path,
        )
    return render_template("index.html")


@app.route("/sentinel", methods=["GET", "POST"])
def sentinel():
    if request.method == "POST":
        try:
            length = float(request.form.get("length"))
            latitude = float(request.form.get("latitude"))
            timestamp = request.form.get("timestamp").split("T")[0]

            file = satelite.get_sentinel_images((length, latitude), timestamp=timestamp)

            numpy_path = CWD / "static" / "input" / "numpy/" / (timestamp + ".npy")
            if not numpy_path.parent.exists():
                numpy_path.parent.mkdir(parents=True)
            np.save(numpy_path, file)
            s2_path = utils.s2_image(numpy_path)
            mask_img_path, mask_gray_path = utils.predict_mask(model, numpy_path)
        except Exception as e:
            return render_template("index.html", exception=e)
        return render_template(
            "index.html",
            img_path=s2_path,
            output_path=mask_img_path,
            mask_gray_path=mask_gray_path,
        )
    return render_template("index.html")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    return redirect(url_for("main"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
