# src/model_tl.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

def make_mobilenetv2(num_classes=7, input_shape=(96,96,3), unfreeze_top=None):
    """
    MobileNetV2 com data augmentation + preprocess_input dentro do modelo.
    Se unfreeze_top (int) for passado, faz fine-tuning liberando as últimas N camadas.
    """
    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_aug")

    inputs = keras.Input(shape=input_shape)
    x = data_aug(inputs)
    x = preprocess_input(x)  # [-1, 1] esperado pela MobileNetV2

    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False  # etapa 1: congelada

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mobilenetv2_emotions")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Configuração para fine-tuning (etapa 2)
    def enable_finetune(model, base_model, unfreeze_top_layers=30, lr=1e-4):
        base_model.trainable = True
        # congela tudo exceto as últimas N camadas
        for layer in base_model.layers[:-unfreeze_top_layers]:
            layer.trainable = False
        # recompila com LR menor
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    if unfreeze_top is not None:
        enable_finetune(model, base, unfreeze_top_layers=unfreeze_top)

    return model
