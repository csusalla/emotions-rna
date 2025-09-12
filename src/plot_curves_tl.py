# src/plot_curves_tl.py
import json
from pathlib import Path
import matplotlib.pyplot as plt

hist_path = Path("models/history_tl.json")
if not hist_path.exists():
    raise FileNotFoundError("models/history_tl.json não encontrado. Rode antes: python src/train_tl.py")

with open(hist_path) as f:
    h = json.load(f)

def plot_one(y_key: str, title: str, out_png: str):
    plt.figure()
    if y_key in h:
        plt.plot(h[y_key], label=y_key)
    val_key = f"val_{y_key}"
    if val_key in h:
        plt.plot(h[val_key], label=val_key)
    plt.title(title)
    plt.xlabel("epoch (etapa1 + etapa2)")
    plt.ylabel(y_key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

plot_one("loss", "TL - Treino vs Val - Loss", "models/tl_curve_loss.png")
plot_one("accuracy", "TL - Treino vs Val - Accuracy", "models/tl_curve_acc.png")
print("✔ Curvas TL salvas em models/tl_curve_loss.png e models/tl_curve_acc.png")
