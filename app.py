import json
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
import cv2

# Carrega modelo e classes
model = tf.keras.models.load_model("models/modelo.keras")
try:
    with open("models/class_names.json") as f:
        CLASSES = json.load(f)
except:
    CLASSES = ["angry","disgust","fear","happy","neutral","sad","surprise"]

# Detector de faces (Haar Cascade já vem com o OpenCV)
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess(img_pil, size=48, equalize=False):
    # Converte para escala de cinza (np.array)
    gray = np.array(img_pil.convert("L"))

    # Detecta rostos
    faces = CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Se detectar, recorta o maior rosto e dá uma pequena folga (padding)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        pad_x, pad_y = int(0.1 * w), int(0.1 * h)
        x0 = max(x - pad_x, 0); y0 = max(y - pad_y, 0)
        x1 = min(x + w + pad_x, gray.shape[1]); y1 = min(y + h + pad_y, gray.shape[0])
        gray = gray[y0:y1, x0:x1]

    # Redimensiona
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    # Equalização de histograma (opcional) — pode ajudar com iluminação
    if equalize:
        gray = cv2.equalizeHist(gray)

    # Normaliza e coloca no formato (1, H, W, 1)
    x = gray.astype(np.float32)[None, ..., None] / 255.0
    return x

def predict(img: Image.Image):
    x = preprocess(img, size=48, equalize=False)  # tente equalize=True se quiser comparar
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return {c: float(probs[i]) for i, c in enumerate(CLASSES)}, CLASSES[pred_idx]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=len(CLASSES)), gr.Textbox(label="Predição")],
    title="Reconhecimento de Emoções (com detecção de face)"
)

if __name__ == "__main__":
    demo.launch()
