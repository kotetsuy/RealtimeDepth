import numpy as np
import onnxruntime as ort
import cv2
import time

MODEL_PATH = 'depth_anything_v2_vits_518.onnx'
INPUT_SIZE = 518

providers = [
    ('MIGraphXExecutionProvider', {'device_id': 0}),
    'CPUExecutionProvider',
]
session = ort.InferenceSession(MODEL_PATH, providers=providers)
print('Active provider:', session.get_providers())

img = cv2.imread('test.jpg')
if img is None:
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError('No image source available')

def preprocess(bgr, size):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    return np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)

inp = preprocess(img, INPUT_SIZE)

for _ in range(3):
    _ = session.run(None, {'input': inp})

N = 30
t0 = time.time()
for _ in range(N):
    out = session.run(None, {'input': inp})
elapsed = time.time() - t0
print(f'Avg inference: {elapsed/N*1000:.1f} ms ({N/elapsed:.1f} FPS)')

depth = out[0][0]
print('Depth shape:', depth.shape, 'min:', depth.min(), 'max:', depth.max())
