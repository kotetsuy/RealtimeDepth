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


class CameraConfig:
    """1 台分のカメラ設定。"""

    def __init__(self, device, width, height, fps, name=None):
        self.device = device  # int (V4L2 index) または str (デバイスパス)
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name or str(device)

    def device_path(self):
        """存在確認に使う実ファイルパス。整数指定は /dev/video{N} に対応付ける。"""
        if isinstance(self.device, int):
            return f'/dev/video{self.device}'
        return str(self.device)

    def __repr__(self):
        return f'<CameraConfig {self.name!r} device={self.device!r}>'


def load_camera_configs(config):
    """config['camera'] を CameraConfig のリストへ正規化する。

    新形式 (camera.devices リスト) と旧形式 (camera.device 直書き) の両方に対応。
    """
    cam = config['camera']
    defaults = cam.get('defaults', {})
    default_w = defaults.get('width', 640)
    default_h = defaults.get('height', 480)
    default_fps = defaults.get('fps', 30)

    if 'devices' in cam:
        entries = cam['devices']
    else:
        # 旧形式: 単一指定を 1 要素リストへ正規化。
        entries = [{
            'device': cam['device'],
            'width': cam.get('width', default_w),
            'height': cam.get('height', default_h),
            'fps': cam.get('fps', default_fps),
        }]

    configs = []
    for entry in entries:
        configs.append(CameraConfig(
            device=entry['device'],
            width=entry.get('width', default_w),
            height=entry.get('height', default_h),
            fps=entry.get('fps', default_fps),
            name=entry.get('name'),
        ))
    return configs


CAMERA_CONFIGS = load_camera_configs(CONFIG)

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


def open_camera(cfg):
    """CameraConfig を開いて検証する。成功時 VideoCapture、失敗時 None。"""
    cap = cv2.VideoCapture(cfg.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    if not cap.isOpened():
        cap.release()
        return None
    # 「開けたが読めない」ケースを除外するため試し読みする。
    ret, _ = cap.read()
    if not ret:
        cap.release()
        return None
    return cap


def find_connected_camera(configs):
    """登録カメラのうち接続されているものを先頭優先で 1 台開いて返す。

    複数刺さっていてもリスト先頭に近いものだけを選ぶ。戻り値 (cfg, cap) / None。
    """
    for cfg in configs:
        # まずパス存在を確認(存在しなければ open を試さず次へ)。
        if not os.path.exists(cfg.device_path()):
            continue
        cap = open_camera(cfg)
        if cap is not None:
            return cfg, cap
    return None


def make_placeholder(message):
    """カメラ未接続時に配信するプレースホルダ JPEG を生成する。"""
    w = max(CAMERA_CONFIGS[0].width if CAMERA_CONFIGS else 640, 640)
    h = CAMERA_CONFIGS[0].height if CAMERA_CONFIGS else 480
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, 'NO CAMERA', (20, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 220), 3)
    cv2.putText(img, message, (20, h // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpg.tobytes() if ok else None


class DepthWorker:
    """カメラ接続をステートマシンで管理し、ホットプラグに追従する。

    DISCONNECTED <-> STREAMING を遷移。未接続中はプレースホルダを配信し続けるので
    ブラウザの MJPEG ストリームは切れず、カメラを刺した瞬間に映像へ復帰する。
    """

    SCAN_INTERVAL = 1.0   # 未接続時の再スキャン間隔 [s]
    READ_FAIL_LIMIT = 10  # 連続 read 失敗が続いたら切断とみなす

    def __init__(self):
        self.lock = threading.Lock()
        self.frame_event = threading.Event()
        self.latest_jpeg = None
        self.fps = 0.0
        self.current_name = None  # 配信中のカメラ名 (未接続時 None)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _publish(self, jpg, fps=0.0, name=None):
        with self.lock:
            self.latest_jpeg = jpg
            self.fps = fps
            self.current_name = name
        self.frame_event.set()

    def _loop(self):
        cap = None
        cfg = None
        fps_alpha = 0.9
        fps_avg = 0.0
        frame_idx = 0
        read_fails = 0
        last_scan = 0.0

        while self.running:
            # === DISCONNECTED: 接続を待つ ===
            if cap is None:
                now = time.time()
                if now - last_scan >= self.SCAN_INTERVAL:
                    last_scan = now
                    found = find_connected_camera(CAMERA_CONFIGS)
                    if found is not None:
                        cfg, cap = found
                        fps_avg = 0.0
                        frame_idx = 0
                        read_fails = 0
                        print(f'[camera] connected: {cfg.name} ({cfg.device!r})', flush=True)
                        continue
                self._publish(make_placeholder('Connect a camera registered in config.yaml'),
                              fps=0.0, name=None)
                time.sleep(0.2)
                continue

            # === STREAMING: 通常処理 ===
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                read_fails += 1
                if read_fails >= self.READ_FAIL_LIMIT or not os.path.exists(cfg.device_path()):
                    print(f'[camera] disconnected: {cfg.name}, waiting...', flush=True)
                    cap.release()
                    cap = None
                    continue
                time.sleep(0.01)
                continue
            read_fails = 0

            inp = preprocess(frame)
            out = session.run(None, {INPUT_NAME: inp})
            depth = out[0][0]
            colored = depth_to_colormap(depth, frame.shape)

            display = np.hstack([frame, colored])

            dt = time.time() - t0
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_avg = fps_alpha * fps_avg + (1 - fps_alpha) * inst_fps if frame_idx > 0 else inst_fps
            cv2.putText(display, f'{fps_avg:.1f} FPS  {cfg.name}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            ok, jpg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                self._publish(jpg.tobytes(), fps=fps_avg, name=cfg.name)
            frame_idx += 1

        if cap is not None:
            cap.release()

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_status(self):
        with self.lock:
            return {'fps': round(self.fps, 2), 'camera': self.current_name}

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


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
    return worker.get_status()


if __name__ == '__main__':
    try:
        app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)
    finally:
        worker.stop()
