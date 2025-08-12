# diag_app.py — quick Streamlit Cloud diagnostics
import os, sys, json
from pathlib import Path
import streamlit as st

st.title("Diag")

st.write("Python:", sys.version)
try:
    import numpy as np
    st.write("numpy:", np.__version__)
except Exception as e:
    st.error(f"numpy import failed: {e}")

try:
    import onnxruntime as ort
    st.write("onnxruntime:", ort.__version__)
    st.write("providers:", ort.get_available_providers())
except Exception as e:
    st.error(f"onnxruntime import failed: {e}")

BASE = Path(__file__).parent.resolve()
st.write("Script dir:", BASE)
st.write("CWD:", Path.cwd())

model_path = BASE / "models" / "model.onnx"
st.write("Model path:", str(model_path))
st.write("Model exists:", model_path.exists())

# List repo tree lightly
try:
    st.write("Root files:", [p.name for p in BASE.iterdir()][:50])
    models_dir = BASE / "models"
    if models_dir.exists():
        st.write("models/ files:", [p.name for p in models_dir.iterdir()])
except Exception as e:
    st.error(f"listing failed: {e}")

# Try to open the model with ORT
try:
    import onnxruntime as ort
    if model_path.exists():
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        i = sess.get_inputs()[0]
        st.success(f"ONNX loaded. Input: name={i.name}, shape={i.shape}")
except Exception as e:
    st.error(f"ONNX load failed: {e}")
