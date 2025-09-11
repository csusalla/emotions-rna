import gradio as gr
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/modelo.keras")
classes = ["raiva","nojo","medo","feliz","neutro","triste","surpresa"]

def predict(img):
    # img chega como PIL RGB; converta p/ 48x48 cinza
    img = img.convert("L").resize((48,48))
    x = np.array(img)[None, ..., None] / 255.0
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {c: float(probs[i]) for i,c in enumerate(classes)}, classes[idx]

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=7), gr.Textbox(label="Predição")],
                    title="Reconhecimento de Emoções")
if __name__ == "__main__":
    demo.launch()
