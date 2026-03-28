import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np

st.title("Cloud Identifier")
st.write("""
This app identifies whether the photo contains cumulus (fluffy), stratus (layer), cirrus (thin), clear skies, or mixed clouds.
""")

@st.cache_resource
def load_model():
    return keras.layers.TFSMLayer("keras_model", call_endpoint="serving_default")

with st.spinner("Loading model, please wait..."):
    model = load_model()

try:
    with open("labels.txt") as f:
        class_names = [line.strip() for line in f if line.strip()]
except Exception:
    class_names = None

def extract_label(label_str):
    # Assumes format like "0 clear"
    parts = label_str.strip().split(" ", 1)
    return parts[1] if len(parts) > 1 else label_str.strip()

uploaded_file = st.file_uploader("Drag and drop a cloud photo here", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=320)
    size = (224, 224)
    img_resized = ImageOps.fit(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img_resized).astype(np.float32)
    img_normalized = (img_array / 127.5) - 1.0
    data = np.expand_dims(img_normalized, axis=0)

    preds = model(data)
    if isinstance(preds, dict):
        preds = list(preds.values())[0]
    preds = preds.numpy()
    index = int(np.argmax(preds[0]))
    confidence = float(preds[0][index])

    if class_names and 0 <= index < len(class_names):
        class_name = extract_label(class_names[index])
    else:
        class_name = f"{index}"

    st.markdown(f"### This is a: {class_name}")
    st.markdown(f"Confidence: {confidence*100:.2f}%")