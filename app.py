#Updated traceback
import os, traceback
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort


IMAGENET_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype="float32")

def preprocess_vgg16_caffe(img: Image.Image, size=(224, 224), nhwc=True):
    img = img.convert("RGB").resize(size)
    x = np.asarray(img, dtype="float32")      # RGB
    x = x[..., ::-1]                          # BGR
    x -= IMAGENET_MEAN_BGR
    if nhwc:
        x = np.expand_dims(x, 0)              # [1,H,W,3]
    else:
        x = np.transpose(x, (2, 0, 1))        # [3,H,W]
        x = np.expand_dims(x, 0)              # [1,3,H,W]
    return x

BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "exported_savedmodel" / "model.onnx"
model_path = str(MODEL_PATH)

st.set_page_config(page_title="Pneumonia Detection (ONNX)", layout="centered")
st.title("Pneumonia Detection from Chest X-ray (ONNX)")

if not MODEL_PATH.exists():
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

with st.sidebar:
    st.subheader("Diagnostics")
    st.write("Script dir:", BASE_DIR)
    try:
        st.write("models/ contents:", os.listdir(BASE_DIR / "exported_savedmodel"))
    except Exception:
        st.write("No 'models' directory or cannot list.")
@st.cache_resource
def load_session_and_layout(path: str):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    nhwc = (len(inp.shape) == 4 and inp.shape[-1] == 3)
    return session, inp.name, nhwc, inp.shape

try:
    session, input_name, expect_nhwc, input_shape = load_session_and_layout(model_path)
    with st.sidebar:
        st.write("ONNX input name:", input_name)
        st.write("ONNX input shape:", input_shape)
        st.write("Layout:", "NHWC" if expect_nhwc else "NCHW")
except Exception as e:
    st.error("Failed to load ONNX model.")
    st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
    st.stop()

threshold = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)

uploaded = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])
if uploaded:
    try:
        img = Image.open(uploaded)
        x = preprocess_vgg16_caffe(img, size=(224, 224), nhwc=expect_nhwc)
        y = session.run(None, {input_name: x})[0]      # [1,1]
        prob = float(y.squeeze())                      # already sigmoid from Keras
        pred = "Pneumonia" if prob >= threshold else "Normal"

        st.image(img, caption=f"Prediction: {pred} (Confidence: {prob:.2f})", use_column_width=True)
        st.write(f"**Raw probability:** {prob:.6f}")
        st.write(f"**Threshold:** {threshold:.2f} → **Decision:** {pred}")
        if abs(prob - threshold) < 0.02:
            st.info("Prediction is close to the threshold. Consider context and more tests.")
    except Exception as e:
        st.error("Inference failed.")
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
