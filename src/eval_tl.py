# src/eval_tl.py
import os, json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_tl import load_from_folders_raw

def main():
    model_path = Path("models/model_tl.keras")
    if not model_path.exists():
        raise FileNotFoundError("models/model_tl.keras não encontrado. Rode antes: python src/train_tl.py")
    model = tf.keras.models.load_model(model_path)

    class_file = Path("models/class_names.json")
    if class_file.exists():
        with open(class_file) as f:
            class_names = json.load(f)
    else:
        class_names = ["angry","disgust","fear","happy","neutral","sad","surprise"]

    _, _, ds_test, _ = load_from_folders_raw(
        root="data", img_size=96, batch_size=64, val_split=0.1, seed=42, color_mode="rgb"
    )

    y_true, y_pred = [], []
    for xb, yb in ds_test:
        probs = model.predict(xb, verbose=0)
        y_true.append(yb.numpy())
        y_pred.append(np.argmax(probs, axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    os.makedirs("models", exist_ok=True)
    with open("models/report_tl.txt", "w") as f:
        f.write(report)
    print(report)

    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predito"); plt.ylabel("Verdadeiro"); plt.title("Matriz de Confusão (TL)")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix_tl.png", dpi=150)
    plt.close()
    print("✔ Salvos: models/report_tl.txt e models/confusion_matrix_tl.png")

if __name__ == "__main__":
    main()
