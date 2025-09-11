# src/eval.py
import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from data import load_from_folders

def main():
    # ==== 1) Carregar modelo e classes ====
    model_path = Path("models/modelo.keras")
    if not model_path.exists():
        raise FileNotFoundError("models/modelo.keras não encontrado. Rode antes: python src/train.py")
    model = tf.keras.models.load_model(model_path)

    class_file = Path("models/class_names.json")
    if class_file.exists():
        with open(class_file) as f:
            class_names = json.load(f)
    else:
        class_names = ["angry","disgust","fear","happy","neutral","sad","surprise"]

    # ==== 2) Dados de teste (sem shuffle) ====
    _, _, ds_test, _ = load_from_folders(
        root="data",
        img_size=48,
        batch_size=64,
        val_split=0.1,
        seed=42,
        color_mode="grayscale",
    )

    # ==== 3) Predições ====
    y_true, y_pred = [], []
    for xb, yb in ds_test:
        probs = model.predict(xb, verbose=0)
        y_true.append(yb.numpy())
        y_pred.append(np.argmax(probs, axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # ==== 4) Relatório + matriz ====
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    os.makedirs("models", exist_ok=True)
    with open("models/report.txt", "w") as f:
        f.write(report)
    print(report)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predito"); plt.ylabel("Verdadeiro"); plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    plt.close()
    print("✔ Salvos: models/report.txt e models/confusion_matrix.png")

if __name__ == "__main__":
    main()
