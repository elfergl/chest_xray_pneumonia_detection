# sanity_onnx.py
import onnxruntime as ort
import numpy as np
from PIL import Image

ONNX = r"models\vgg16_pytorch.onnx"
IMG  = r"path\to\some\xray.jpg"  # use a Normal and a Pneumonia example if you have both

def preprocess(img: Image.Image, layout="NHWC"):
    # VGG16 "caffe" preproc (RGB->BGR + mean subtract), no /255
    img = img.convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype(np.float32)
    arr = arr[:, :, ::-1]  # RGB->BGR
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    arr = arr - mean
    if layout == "NHWC":
        x = np.expand_dims(arr, 0)
    else:
        x = np.transpose(arr, (2,0,1))[None, ...]
    return x

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
input_name = inp.name
shape = inp.shape
layout = "NHWC" if (len(shape)==4 and shape[-1]==3) else "NCHW"

img = Image.open(IMG)
x = preprocess(img, layout)
y = sess.run(None, {input_name: x})[0]  # shape (1,1) expected
logit_or_prob = float(y[0,0])

# Some exports keep sigmoid; some export logits. Just compute prob both ways:
prob_from_raw = logit_or_prob
prob_from_sigmoid = 1.0 / (1.0 + np.exp(-logit_or_prob))

print("Raw output:", logit_or_prob)
print("As prob (assuming already sigmoid):", prob_from_raw)
print("As prob (apply sigmoid):           ", prob_from_sigmoid)
