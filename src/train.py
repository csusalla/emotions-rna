# src/train.py
import os
import json

# Reprodutibilidade
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

from tensorflow import keras
from data import load_from_folders       # seu loader por pastas (train/test)
from model import make_cnn               # sua CNN 48x48x1

def main():
    # ==== 1) Dados ====
    ds_train, ds_val, ds_test, class_names = load_from_folders(
        root="data",
        img_size=48,
        batch_size=64,
        val_split=0.1,
        seed=42,
        color_mode="grayscale",
    )

    # ==== 2) Infra de saída ====
    os.makedirs("models", exist_ok=True)
    with open("models/class_names.json", "w") as f:
        json.dump(class_names, f)

    # ==== 3) Modelo ====
    model = make_cnn(input_shape=(48, 48, 1), num_classes=len(class_names))

    # ==== 4) Callbacks ====
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=8, restore_best_weights=True, monitor="val_accuracy"
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=4, factor=0.5, min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            "models/modelo.keras", save_best_only=True, monitor="val_accuracy"
        ),
    ]

    # ==== 5) Treino ====
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=50,
        callbacks=callbacks,
        verbose=2,
    )

    # ==== 6) Salvar histórico para gráficos ====
    with open("models/history.json", "w") as f:
        json.dump(history.history, f)

    # ==== 7) Avaliação rápida no teste ====
    loss, acc = model.evaluate(ds_test, verbose=0)
    print(f"[TEST] acc={acc:.4f} loss={loss:.4f}")

if __name__ == "__main__":
    main()
