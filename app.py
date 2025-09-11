import json
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("models/modelo.keras")
try:
    with open("models/class_names.json") as f:
        classes = json.load(f)
except:
    classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]

def predict(img: Image.Image):
    img = img.convert("L").resize((48,48))        # grayscale 48x48
    x = np.array(img, dtype=np.float32)[None, ..., None] / 255.0
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {c: float(probs[i]) for i,c in enumerate(classes)}, classes[idx]

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=len(classes)), gr.Textbox(label="Predição")],
                    title="Reconhecimento de Emoções")
if __name__ == "__main__":
    demo.launch()
