# src/plot_curves.py
import json
import matplotlib.pyplot as plt
from pathlib import Path

hist_path = Path("models/history.json")
if not hist_path.exists():
    raise FileNotFoundError("models/history.json não encontrado. Rode antes: python src/train.py")

with open(hist_path) as f:
    h = json.load(f)

def plot_one(y_key: str, title: str, out_png: str):
    plt.figure()
    plt.plot(h.get(y_key, []), label=y_key)
    val_key = f"val_{y_key}"
    if val_key in h:
        plt.plot(h[val_key], label=val_key)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(y_key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

Path("models").mkdir(exist_ok=True)
plot_one("loss", "Treino vs Val - Loss", "models/curve_loss.png")
plot_one("accuracy", "Treino vs Val - Accuracy", "models/curve_acc.png")
print("✔ Curvas salvas em: models/curve_loss.png e models/curve_acc.png")
