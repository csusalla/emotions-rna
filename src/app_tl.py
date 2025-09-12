# src/app_tl.py
import json
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
import cv2

# Carrega modelo TL e classes
model = tf.keras.models.load_model("models/model_tl.keras")
try:
    with open("models/class_names.json") as f:
        CLASSES = json.load(f)
except:
    CLASSES = ["angry","disgust","fear","happy","neutral","sad","surprise"]

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_rgb(img_pil: Image.Image, size=96):
    # Converte para RGB numpy
    rgb = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Detecta rosto no cinza (mais robusto), recorta no RGB
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        pad_x, pad_y = int(0.1*w), int(0.1*h)
        x0 = max(x - pad_x, 0); y0 = max(y - pad_y, 0)
        x1 = min(x + w + pad_x, rgb.shape[1]); y1 = min(y + h + pad_y, rgb.shape[0])
        rgb = rgb[y0:y1, x0:x1]

    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32)[None, ...]  # (1, H, W, 3), valores 0..255
    return x

def predict(img: Image.Image):
    x = preprocess_rgb(img, size=96)
    # O modelo já tem preprocess_input dentro, então não dividir por 255 aqui.
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {c: float(probs[i]) for i, c in enumerate(CLASSES)}, CLASSES[idx]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=len(CLASSES)), gr.Textbox(label="Predição")],
    title="Reconhecimento de Emoções (MobileNetV2 · 96×96 RGB)"
)

if __name__ == "__main__":
    demo.launch()
