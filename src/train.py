import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np, os

# Carregue aqui os seus dados prontos em X_train, y_train, X_val, y_val
# (ou use tf.data a partir de arquivos)
# X_*: (N, 48, 48, 1); y_*: inteiros 0..num_classes-1

num_classes = 7
input_shape = (48, 48, 1)

data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomContrast(0.1),
])

def make_model():
    i = keras.Input(shape=input_shape)
    x = data_aug(i)
    x = layers.Rescaling(1./255)(x)
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

model = make_model()
callbacks = [
    keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
    keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    keras.callbacks.ModelCheckpoint("models/modelo.keras", save_best_only=True, monitor="val_accuracy"),
]

# history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
#                     epochs=60, batch_size=64, callbacks=callbacks)
# model.save("models/final.keras")
