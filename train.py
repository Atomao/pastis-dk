import argparse
from pprint import pprint

import tensorflow as tf
import tensorflow_addons as tfa

from backbones.unet3d import get_model
from dataset.dataset import get_datasets
from metrics import LogitMeanIOU
from utils import save_hist, scheduler

parser = argparse.ArgumentParser(description="Model training script")
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--num_classes", type=int, default=20)
parser.add_argument("--feature_maps_factor", type=int, default=8)
parser.add_argument(
    "--weights_path",
    type=str,
    default=None,
)
parser.add_argument("--distribute", action="store_true")
parser.add_argument("--mixed_precision", action="store_true")


args = parser.parse_args()


if __name__ == "__main__":
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if args.mixed_precision else "float32"
    )
    train_ds, val_ds, _ = get_datasets(
        batch_size=args.batch_size, folder_path=args.dataset_path
    )

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./bestWeights.h5",
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
    ]

    if args.distribute:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = get_model(
                num_classes=args.num_classes,
                feats=args.feature_maps_factor,
                weights_path=args.weights_path,
            )
            opt = tfa.optimizers.RectifiedAdam(learning_rate=args.learning_rate)
            opt = tfa.optimizers.Lookahead(opt)

            model.compile(
                optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy", LogitMeanIOU(num_classes=args.num_classes)],
            )
    else:
        model = get_model(
            num_classes=args.num_classes,
            feats=args.feature_maps_factor,
            weights_path=args.weights_path,
        )
        opt = tfa.optimizers.RectifiedAdam(learning_rate=args.learning_rate)
        opt = tfa.optimizers.Lookahead(opt)

        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy", LogitMeanIOU(num_classes=args.num_classes)],
        )

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )
    try:
        model.save_weights("./weights/Weights3DUnet.h5")
        model.save("./weights/Full3DUnet.h5")
    except:
        model.save_weights("./Weights3DUnet.h5")
        model.save("./Full3DUnet.h5")
        model.save_conf
