import tensorflow as tf
import tf2onnx
import numpy as np

H5 = "models/vgg16_finetuned_model.h5"
ONNX_OUT = "models/vgg16_pytorch.onnx"
INPUT_SHAPE = (224, 224, 3)

# Build model with EXACT names seen in the H5:
# Groups present: global_average_pooling2d, dropout, dense/sequential/dense
inp = tf.keras.Input(shape=INPUT_SHAPE, name="input_1")
base = tf.keras.applications.VGG16(include_top=False, weights=None, input_tensor=inp)

x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(base.output)

# IMPORTANT: replicate the Sequential container named "sequential"
head = tf.keras.Sequential(
    name="sequential",
    layers=[
        tf.keras.layers.Dropout(0.5, name="dropout"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="dense"),  # final layer
    ],
)

out = head(x)
model = tf.keras.Model(inp, out, name="vgg16_head_exact")

# Load weights BY NAME (matches vgg16 blocks + our head container/leaf names)
model.load_weights(H5, by_name=True, skip_mismatch=True)

# Diagnostics: confirm the final Dense has weights
w = model.get_layer("sequential").get_layer("dense").get_weights()
print("[diag] sequential/dense weights:",
      "loaded" if w else "EMPTY",
      ([a.shape for a in w] if w else ""))

# Export to ONNX
sig = (tf.TensorSpec((None,) + INPUT_SHAPE, tf.float32, name="input_1"),)
tf2onnx.convert.from_keras(model, input_signature=sig, opset=13, output_path=ONNX_OUT)
print("Saved ONNX to:", ONNX_OUT)
