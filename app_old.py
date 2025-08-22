# Streamlit inference app for VGG16 (Keras → ONNX) pneumonia model
# - Expects models/model.onnx (exported from Keras model)
# - Uses VGG16 "caffe" preprocessing: RGB→BGR, subtract ImageNet means, NO /255
# - Final Keras layer was Dense(1, activation='sigmoid') → ONNX already outputs probability

import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# ----- Preprocessing (matches tf.keras.applications.vgg16.preprocess_input in caffe mode) -----
IMAGENET_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype="float32")  # per-channel B,G,R means

def preprocess_vgg16_caffe(img: Image.Image, size=(224, 224), nhwc: bool = True) -> np.ndarray:
    """
    Convert PIL image -> model input tensor.
    Steps:
      - Convert to RGB
      - Resize to 224x224
      - Convert to float32 in 0..255 (NO /255.0)
      - RGB -> BGR
      - Subtract ImageNet mean per channel
      - Return [1,H,W,3] if nhwc=True, else [1,3,H,W]
    """
    img = img.convert("RGB").resize(size)
    x = np.asarray(img, dtype="float32")     # [H,W,3] RGB
    x = x[..., ::-1]                         # BGR
    x -= IMAGENET_MEAN_BGR                   # mean subtraction

    if nhwc:
        x = np.expand_dims(x, axis=0)        # [1,H,W,3]
    else:
        x = np.transpose(x, (2, 0, 1))       # [3,H,W]
        x = np.expand_dims(x, axis=0)        # [1,3,H,W]
    return x

# ----- Streamlit UI -----
st.set_page_config(page_title="Pneumonia Detection (ONNX)", layout="centered")
st.title("Pneumonia Detection from Chest X-ray (ONNX)")

st.write(
    "Upload a chest X-ray image. The model will predict whether pneumonia is present. "
    "Preprocessing matches VGG16 (caffe): RGB→BGR, ImageNet mean subtraction, no scaling."
)

# Threshold UI
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)

@st.cache_resource
def load_session():
    # Load ONNX model and infer input layout
    session = ort.InferenceSession("models/vgg16_pytorch.onnx", providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    input_name = inp.name
    # NHWC if last dim is 3; otherwise assume NCHW
    expect_nhwc = (len(inp.shape) == 4 and inp.shape[-1] == 3)
    return session, input_name, expect_nhwc

session, input_name, expect_nhwc = load_session()

uploaded = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded)

    # Preprocess to match training
    x = preprocess_vgg16_caffe(img, size=(224, 224), nhwc=expect_nhwc)

    # Run inference
    y = session.run(None, {input_name: x})[0]  # expected shape [1,1]
    prob = float(y.squeeze())                  # already sigmoid output from Keras final layer

    pred = "Pneumonia" if prob >= threshold else "Normal"

    # Display results
    st.image(img, caption=f"Prediction: {pred} (Confidence: {prob:.2f})", use_column_width=True)

    st.write(f"**Raw probability:** {prob:.6f}")
    st.write(f"**Threshold:** {threshold:.2f} → **Decision:** {pred}")

    # Tiny safety note if prob is extremely close to threshold
    if abs(prob - threshold) < 0.02:
        st.info("Prediction is close to the threshold. Consider clinical context and additional tests.")
