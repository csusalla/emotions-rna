# src/eval.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from data import load_from_folders

def main():
    model = tf.keras.models.load_model("models/modelo.keras")
    _, _, ds_test, class_names = load_from_folders(root="data", img_size=48, batch_size=64, val_split=0.1, seed=42)

    # coletar y_true e y_pred
    y_true = []
    y_pred = []
    for xb, yb in ds_test:
        probs = model.predict(xb, verbose=0)
        y_true.append(yb.numpy())
        y_pred.append(np.argmax(probs, axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
