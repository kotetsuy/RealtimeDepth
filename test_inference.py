"""推論単体のベンチマーク。app.py の設定 (config.yaml) をそのまま使う。

    .venv-torch/bin/python test_inference.py
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get('CONFIG_PATH', os.path.join(BASE_DIR, 'config.yaml'))
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

INPUT_SIZE = CONFIG['model']['input_size']
RUNTIME = CONFIG.get('runtime', {})
DEVICE = RUNTIME.get('device', 'cuda')
DTYPE = torch.float16 if RUNTIME.get('precision', 'fp16') == 'fp16' else torch.float32

sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, CONFIG['model']['repo'])))
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

ENCODER_CONFIGS = {
    'vits': {'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}
ENCODER = CONFIG['model']['encoder']

model = DepthAnythingV2(encoder=ENCODER, **ENCODER_CONFIGS[ENCODER])
model.load_state_dict(torch.load(
    os.path.abspath(os.path.join(BASE_DIR, CONFIG['model']['checkpoint'])), map_location='cpu'))
model = model.to(device=DEVICE, dtype=DTYPE).eval()

if DEVICE.startswith('cuda'):
    print('Device:', torch.cuda.get_device_name(0),
          f'({torch.cuda.get_device_properties(0).gcnArchName})')
print('Precision:', str(DTYPE).replace('torch.', ''))

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE, dtype=DTYPE).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE, dtype=DTYPE).view(1, 3, 1, 1)

img = cv2.imread(os.path.join(BASE_DIR, 'test.jpg'))
if img is None:
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    cap.release()
    if not ret:
        # カメラも画像も無ければ合成画像でベンチだけ回す。
        print('No image source available; using a synthetic frame.')
        img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)


def preprocess(bgr, size):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    t = torch.from_numpy(np.ascontiguousarray(resized)).to(DEVICE)
    t = t.permute(2, 0, 1).unsqueeze(0).to(DTYPE).div_(255.0)
    return (t - MEAN) / STD


inp = preprocess(img, INPUT_SIZE)

with torch.inference_mode():
    for _ in range(3):
        out = model(inp)
    if DEVICE.startswith('cuda'):
        torch.cuda.synchronize()

    N = 30
    t0 = time.time()
    for _ in range(N):
        out = model(inp)
    if DEVICE.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.time() - t0

print(f'Avg inference: {elapsed/N*1000:.1f} ms ({N/elapsed:.1f} FPS)')

depth = out[0].float().cpu().numpy()
print('Depth shape:', depth.shape, 'min:', depth.min(), 'max:', depth.max())
