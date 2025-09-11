# src/model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_cnn(input_shape=(48,48,1), num_classes=7):
    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomContrast(0.1),
    ])

    i = keras.Input(shape=input_shape)
    x = data_aug(i)                 # augment
    # NÃO use Rescaling aqui porque já normalizamos no pipeline do dataset.
    for f in [32, 64, 128]:
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    o = layers.Dense(num_classes, activation="softmax")(x)

    m = keras.Model(i, o)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m
