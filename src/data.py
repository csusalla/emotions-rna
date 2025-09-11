# src/data.py
from pathlib import Path
import tensorflow as tf

def load_from_folders(root="data", img_size=48, batch_size=64, val_split=0.1, seed=42, color_mode="grayscale"):
    """
    Lê pastas data/train e data/test com subpastas por classe.
    Retorna: ds_train, ds_val, ds_test, class_names
    Todos já normalizados para [0,1] e com cache/prefetch.
    """
    root = Path(root)
    train_dir = root / "train"
    test_dir  = root / "test"

    ds_train = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        color_mode=color_mode,      # "grayscale" => shape (H,W,1)
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
        shuffle=False,              # importante para avaliação
    )

    class_names = ds_train.class_names  # ex.: ["angry","disgust","fear","happy","neutral","sad","surprise"]

    # Normalização + performance
    AUTOTUNE = tf.data.AUTOTUNE
    def prep(ds, training=False):
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
        return ds.cache().prefetch(AUTOTUNE)

    return prep(ds_train, True), prep(ds_val), prep(ds_test), class_names
