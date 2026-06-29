"""
Depth Anything V2 リアルタイム深度推定 - ブラウザ表示サーバー

http://<NucBoxのIP>:8000/  にChromeでアクセス
"""
import os
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from flask import Flask, Response, render_template_string

# === 設定の読み込み ===
CONFIG_PATH = os.environ.get('CONFIG_PATH', os.path.join(os.path.dirname(__file__), 'config.yaml'))
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

MODEL_PATH = CONFIG['model']['path']
INPUT_SIZE = CONFIG['model']['input_size']

CAMERA_DEVICE = CONFIG['camera']['device']  # int (V4L2 index) または str (デバイスパス)
CAM_WIDTH = CONFIG['camera']['width']
CAM_HEIGHT = CONFIG['camera']['height']
CAM_FPS = CONFIG['camera']['fps']

JPEG_QUALITY = CONFIG['server']['jpeg_quality']
HOST = CONFIG['server']['host']
PORT = CONFIG['server']['port']

COMPILE_CACHE_DIR = CONFIG.get('runtime', {}).get('compile_cache_dir')
if COMPILE_CACHE_DIR:
    COMPILE_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(CONFIG_PATH), COMPILE_CACHE_DIR))
    os.makedirs(COMPILE_CACHE_DIR, exist_ok=True)

# === ONNX Runtime セットアップ ===
# ROCm 7.1+ では ROCMExecutionProvider が廃止され MIGraphXExecutionProvider に統合
migraphx_opts = {'device_id': 0}
if COMPILE_CACHE_DIR:
    migraphx_opts['migraphx_model_cache_dir'] = COMPILE_CACHE_DIR
providers = [
    ('MIGraphXExecutionProvider', migraphx_opts),
    'CPUExecutionProvider',
]
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=providers)
print('Active provider:', session.get_providers())
INPUT_NAME = session.get_inputs()[0].name

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    return np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)


def depth_to_colormap(depth, target_shape):
    """Depth Anything V2 の出力は disparity 風 (近=大)。COLORMAP_INFERNO で 近=明 / 遠=暗 になる。"""
    h, w = target_shape[:2]
    depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    d_min = np.percentile(depth_resized, 2)
    d_max = np.percentile(depth_resized, 98)
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(depth_resized, dtype=np.uint8)
    else:
        norm = np.clip((depth_resized - d_min) / (d_max - d_min), 0, 1)
        norm = (norm * 255).astype(np.uint8)

    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


class DepthWorker:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open camera {CAMERA_DEVICE!r}')

        self.lock = threading.Lock()
        self.frame_event = threading.Event()
        self.latest_jpeg = None
        self.fps = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        fps_alpha = 0.9
        fps_avg = 0.0
        frame_idx = 0
        while self.running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            inp = preprocess(frame)
            out = session.run(None, {INPUT_NAME: inp})
            depth = out[0][0]
            colored = depth_to_colormap(depth, frame.shape)

            display = np.hstack([frame, colored])

            dt = time.time() - t0
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_avg = fps_alpha * fps_avg + (1 - fps_alpha) * inst_fps if frame_idx > 0 else inst_fps
            cv2.putText(display, f'{fps_avg:.1f} FPS', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            ok, jpg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with self.lock:
                    self.latest_jpeg = jpg.tobytes()
                    self.fps = fps_avg
                self.frame_event.set()
            frame_idx += 1

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


worker = DepthWorker()

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>Depth Anything V2 - ROCm Realtime Demo</title>
  <style>
    body {
      background: #111;
      color: #eee;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 0;
      padding: 16px;
      text-align: center;
    }
    h1 { font-size: 18px; font-weight: 500; margin: 8px 0 16px; color: #aaa; }
    .stream {
      max-width: 100%;
      border-radius: 8px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    }
    .legend {
      margin-top: 12px;
      font-size: 13px;
      color: #888;
    }
  </style>
</head>
<body>
  <h1>Depth Anything V2 Small / onnxruntime-migraphx / gfx1150</h1>
  <img class="stream" src="/stream" alt="depth stream">
  <div class="legend">
    左: 元映像 / 右: 深度マップ (明=近い, 暗=遠い, 想定レンジ ~10m)
  </div>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


def mjpeg_generator():
    boundary = b'--frame'
    while True:
        worker.frame_event.wait(timeout=1.0)
        worker.frame_event.clear()
        jpg = worker.get_jpeg()
        if jpg is None:
            continue
        yield (boundary + b'\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n'
               + jpg + b'\r\n')


@app.route('/stream')
def stream():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    return {'fps': round(worker.fps, 2)}


if __name__ == '__main__':
    try:
        app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)
    finally:
        worker.stop()
