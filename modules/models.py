import onnxruntime as ort
import numpy as np
import requests
from PIL import Image
import streamlit as st

@st.cache_resource
def load_and_check():
    MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx"
    MODEL_PATH = "mnist-8.onnx"

    print(f"Install Model: {MODEL_URL}")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Complete")

    ort_session = ort.InferenceSession(MODEL_PATH)
    print("Load Completed")

    return ort_session

# Image preprocessing
def preprocess(img: Image.Image):
    img = img.convert("L").resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array  # invert
    img_array = img_array / 255.0  # normalize
    img_array = img_array.astype(np.float32).reshape(1, 1, 28, 28)
    return img_array

# Inference
def predict(img_array):
    model = load_and_check()
    inputs = {model.get_inputs()[0].name: img_array}
    outputs = model.run(None, inputs)
    return outputs[0][0]  # shape (10,)


def softmax(logits):
    exps = np.exp(logits - np.max(logits))  # overflow 방지
    return exps / np.sum(exps)