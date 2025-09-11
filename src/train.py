# src/train.py
import os, json
import tensorflow as tf
from tensorflow import keras
from data import load_from_folders
from model import make_cnn

os.makedirs("models", exist_ok=True)

def main():
    ds_train, ds_val, ds_test, class_names = load_from_folders(
        root="data", img_size=48, batch_size=64, val_split=0.1, seed=42, color_mode="grayscale"
    )

    # salva as classes para usar na demo
    with open("models/class_names.json", "w") as f:
        json.dump(class_names, f)

    input_shape = (48, 48, 1)
    model = make_cnn(input_shape=input_shape, num_classes=len(class_names))

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint("models/modelo.keras", save_best_only=True, monitor="val_accuracy"),
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=50,
        callbacks=callbacks,
        verbose=2
    )

    # avaliação rápida
    loss, acc = model.evaluate(ds_test, verbose=0)
    print(f"[TEST] acc={acc:.4f} loss={loss:.4f}")

if __name__ == "__main__":
    main()
