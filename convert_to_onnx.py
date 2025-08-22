# convert_h5_with_patch.py
import os
os.environ["TF_KERAS"] = "1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tf_keras as keras
from tf_keras.layers import InputLayer
import tf2onnx, onnxruntime as ort, numpy as np
from pathlib import Path
import inspect

# --- paths ---
H5 = Path(r"models/vgg16_finetuned_model.h5")   # adjust if yours is elsewhere
OUT = Path("models"); OUT.mkdir(exist_ok=True)
SM = Path("exported_savedmodel"); SM.mkdir(exist_ok=True)
ONNX = OUT / "model_final.onnx"

# --- robust classmethod patch for legacy 'batch_shape' ---
orig_from_config_attr = InputLayer.from_config
# get underlying function if it's already a classmethod
orig_from_config_fn = (orig_from_config_attr.__func__ 
                       if isinstance(orig_from_config_attr, classmethod) 
                       else orig_from_config_attr)

def patched_from_config(cls, config):
    cfg = dict(config) if isinstance(config, dict) else config
    if isinstance(cfg, dict) and "batch_shape" in cfg and "batch_input_shape" not in cfg:
        cfg["batch_input_shape"] = cfg.pop("batch_shape")
    return orig_from_config_fn(cls, cfg)

InputLayer.from_config = classmethod(patched_from_config)

# --- load legacy h5 with tf_keras ---
model = keras.models.load_model(H5, compile=False)
print("Loaded model OK.")
try:
    last = model.layers[-1]
    act = getattr(getattr(last, "activation", None), "__name__", str(getattr(last,"activation",None)))
    print("Last layer:", type(last).__name__, "| activation:", act)
except Exception as e:
    print("Could not inspect last layer:", e)

# --- export SavedModel with the exact NHWC signature your app uses ---
@tf.function(input_signature=[tf.TensorSpec([1,224,224,3], tf.float32, name="input")])
def serving(x):
    return {"prob": model(x, training=False)}

tf.saved_model.save(model, str(SM), signatures={"serving_default": serving})
print("SavedModel exported to:", SM.resolve())

# --- convert SavedModel -> ONNX ---
tf2onnx.convert.from_saved_model(str(SM), opset=13, output_path=str(ONNX))
print("ONNX written to:", ONNX.resolve())

# --- runtime smoke test ---
sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
print("ONNX input:", iname, sess.get_inputs()[0].shape)
y = sess.run(None, {iname: np.random.randn(1,224,224,3).astype("float32")})[0]
print("Output shape:", y.shape, "range:", float(y.min()), float(y.max()))
