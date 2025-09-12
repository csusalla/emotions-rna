# src/data_tl.py
from pathlib import Path
import tensorflow as tf

def load_from_folders_raw(
    root="data", img_size=96, batch_size=64, val_split=0.1, seed=42, color_mode="rgb"
):
    """
    Lê pastas data/train e data/test com subpastas por classe.
    NÃO normaliza (mantém 0..255) — a normalização é feita no modelo (preprocess_input).
    Retorna: ds_train, ds_val, ds_test, class_names
    """
    root = Path(root)
    train_dir = root / "train"
    test_dir  = root / "test"

    ds_train = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        color_mode=color_mode,          # "rgb"
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )

    ds_val = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        color_mode=color_mode,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )

    ds_test = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        color_mode=color_mode,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = ds_train.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    def prep(ds):
        # Sem normalização aqui
        return ds.cache().prefetch(AUTOTUNE)

    return prep(ds_train), prep(ds_val), prep(ds_test), class_names
