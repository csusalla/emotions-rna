# src/train_tl.py
import os, json, argparse
import tensorflow as tf
from tensorflow import keras

# ===== Reprodutibilidade =====
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

from data_tl import load_from_folders_raw
from model_tl import make_mobilenetv2

STAGE1_WEIGHTS = "models/model_tl_stage1.weights.h5"   # salva só pesos
FINAL_MODEL    = "models/model_tl.keras"               # modelo final (SavedModel/Keras)
HIST_JSON      = "models/history_tl.json"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs1", type=int, default=12, help="épocas etapa 1 (base congelada)")
    p.add_argument("--epochs2", type=int, default=8,  help="épocas etapa 2 (fine-tuning)")
    p.add_argument("--unfreeze-top", type=int, default=30, help="liberar últimas N camadas para fine-tuning")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--fast", action="store_true",
                   help="modo rápido: usa poucos batches (para testar o pipeline)")
    return p.parse_args()

def maybe_shrink(ds_train, ds_val, enable=False):
    if not enable:
        return ds_train, ds_val
    # reduz para ~poucos passos por época (rápido para debug)
    ds_train_small = ds_train.take(50)  # ~50 batches
    ds_val_small   = ds_val.take(10)    # ~10 batches
    return ds_train_small, ds_val_small

def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)

    # ===== Dados (RGB, sem normalizar) =====
    ds_train, ds_val, ds_test, class_names = load_from_folders_raw(
        root="data",
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=0.1,
        seed=42,
        color_mode="rgb",
    )
    # Modo rápido (debug)
    ds_train, ds_val = maybe_shrink(ds_train, ds_val, enable=args.fast)

    with open("models/class_names.json", "w") as f:
        json.dump(class_names, f)

    # ===== ETAPA 1: base congelada =====
    print("\n==============================")
    print(" ETAPA 1 — BASE CONGELADA")
    print("==============================\n")

    model_stage1 = make_mobilenetv2(
        num_classes=len(class_names),
        input_shape=(args.img_size, args.img_size, 3),
    )
    callbacks1 = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(STAGE1_WEIGHTS, save_best_only=True,
                                        save_weights_only=True, monitor="val_accuracy"),
        keras.callbacks.CSVLogger("models/log_stage1.csv", append=False),
    ]
    hist1 = model_stage1.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs1,
        callbacks=callbacks1,
        verbose=2,
    )
    # Garante melhores pesos da etapa 1
    model_stage1.load_weights(STAGE1_WEIGHTS)

    # ===== ETAPA 2: fine-tuning =====
    print("\n==============================")
    print(f" ETAPA 2 — FINE-TUNING (últimas {args.unfreeze_top} camadas)")
    print("==============================\n")

    # cria novo modelo com FT ligado e reaproveita pesos
    model_ft = make_mobilenetv2(
        num_classes=len(class_names),
        input_shape=(args.img_size, args.img_size, 3),
        unfreeze_top=args.unfreeze_top,
    )
    model_ft.set_weights(model_stage1.get_weights())

    callbacks2 = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=5e-6),
        keras.callbacks.ModelCheckpoint(FINAL_MODEL, save_best_only=True, monitor="val_accuracy"),
        keras.callbacks.CSVLogger("models/log_stage2.csv", append=False),
    ]
    hist2 = model_ft.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs2,
        callbacks=callbacks2,
        verbose=2,
    )

    # ===== Histórico combinado =====
    hist_comb = {}
    for k, v in hist1.history.items():
        hist_comb[k] = list(v) + list(hist2.history.get(k, []))
    with open(HIST_JSON, "w") as f:
        json.dump(hist_comb, f)

    # ===== Avaliação rápida =====
    loss, acc = model_ft.evaluate(ds_test, verbose=0)
    print(f"\n[TEST] TL acc={acc:.4f} loss={loss:.4f}")
    print(f"✔ Modelo final salvo em: {FINAL_MODEL}")
    print("✔ Históricos em: models/log_stage1.csv, models/log_stage2.csv e models/history_tl.json")

if __name__ == "__main__":
    main()
