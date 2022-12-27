import tensorflow as tf
from tensorflow.keras.activations import linear
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv3D,
    Conv3DTranspose,
    Dropout,
    Input,
    LeakyReLU,
    MaxPool3D,
)
from tensorflow_addons.activations import mish
from tensorflow_addons.layers import GroupNormalization


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, middle_channels, out_channels, activation="mish", norm="batch"):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv3D(middle_channels, kernel_size=3, padding="same", strides=1)
        self.conv2 = Conv3D(out_channels, kernel_size=3, padding="same", strides=1)

        if activation.lower() == "mish":
            self.activation = Activation(mish)
        elif activation.lower() == "lrelu":
            self.activation = LeakyReLU(alpha=0.3)

        if norm == "batch":
            self.norm1 = BatchNormalization(axis=-1)
            self.norm2 = BatchNormalization(axis=-1)
        elif norm == "group":
            self.norm1 = GroupNormalization(16)
            self.norm2 = GroupNormalization(16)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.activation(self.norm1(x))
        x = self.conv2(x)
        x = self.activation(self.norm2(x))
        return x


class DeConvBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, activation="mish", norm="batch"):
        super(DeConvBlock, self).__init__()
        self.deconv = Conv3DTranspose(
            out_channels, kernel_size=3, padding="same", strides=2, output_padding=None
        )

        if activation.lower() == "mish":
            self.activation = Activation(mish)
        elif activation.lower() == "lrelu":
            self.activation = LeakyReLU(alpha=0.01)
        else:
            self.activation = None

        if norm == "batch":
            self.norm = BatchNormalization(axis=-1)
        elif norm == "group":
            self.norm = GroupNormalization(16)
        else:
            self.norm = None

    def call(self, inputs):
        x = self.deconv(inputs)
        x = self.activation(self.norm(x)) if self.activation else x
        return x


class Unet(tf.keras.models.Model):
    def __init__(self, num_classes=20, feats=8):
        super(Unet, self).__init__()

        self.dropout = Dropout(0.2)
        self.maxpool = MaxPool3D(2, strides=2, padding="valid")

        self.encoder1 = ConvBlock(feats * 4, feats * 4)
        self.encoder2 = ConvBlock(feats * 8, feats * 8)

        self.bottleneck = ConvBlock(feats * 16, feats * 16)

        self.upsample1 = DeConvBlock(feats * 8, activation="None", norm="None")
        self.decoder1 = ConvBlock(feats * 8, feats * 8)
        self.upsample2 = DeConvBlock(feats * 4)
        self.decoder2 = ConvBlock(feats * 4, feats * 2)

        self.final = Conv3D(num_classes, kernel_size=3, padding="same", strides=1)

    def call(self, inputs):
        skip1 = self.encoder1(inputs)
        x = self.maxpool(skip1)
        skip2 = self.encoder2(x)
        x = self.maxpool(skip2)

        x = self.bottleneck(x)

        x = self.upsample1(x)
        w, h = skip2.shape[2:4]
        t = min(x.shape[1], skip2.shape[1])
        x = tf.concat([skip2[:, :t, :w, :h, :], x[:, :t, :w, :h, :]], axis=-1)
        x = self.dropout(x)

        x = self.decoder1(x)
        x = self.upsample2(x)

        w, h = skip1.shape[2:4]
        t = min(x.shape[1], skip1.shape[1])
        x = tf.concat([skip1[:, :t, :w, :h, :], x[:, :t, :w, :h, :]], axis=-1)
        x = self.dropout(x)
        x = self.decoder2(x)
        x = tf.reduce_mean(self.final(x), axis=1)

        return x


def get_model(
    num_classes,
    feats,
    weights_path="/Users/angel777/Desktop/segm/Flask/segm/static/utils/LastMRadam.h5",
):
    model = Unet()
    model(Input(shape=(61, 128, 128, 10)))
    if weights_path:
        model.load_weights(weights_path)
    return model
