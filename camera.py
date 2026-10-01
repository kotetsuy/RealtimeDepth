"""Platform-specific camera I/O; independent of model initialization."""
import os

import cv2


class CameraConfig:
    """1 台分のカメラ設定。"""

    def __init__(self, device, width, height, fps, name=None, backend=None):
        self.device = device  # int (V4L2 index) または str (デバイスパス)
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name or str(device)
        self.backend = (backend or ('MSMF' if os.name == 'nt' else 'V4L2')).upper()
        allowed = ('MSMF', 'DSHOW') if os.name == 'nt' else ('V4L2',)
        if self.backend not in allowed:
            raise ValueError(f'camera backend must be one of {allowed}: {self.backend}')
        if os.name == 'nt' and (type(device) is not int or device < 0):
            raise ValueError('Windows camera.device must be a non-negative integer index')

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
            'backend': cam.get('backend', defaults.get('backend')),
        }]

    configs = []
    for entry in entries:
        configs.append(CameraConfig(
            device=entry['device'],
            width=entry.get('width', default_w),
            height=entry.get('height', default_h),
            fps=entry.get('fps', default_fps),
            name=entry.get('name'),
            backend=entry.get('backend', defaults.get('backend')),
        ))
    return configs


def device_present(cfg):
    """Windows indices are probed by open/read, never as filesystem paths."""
    return os.name == 'nt' or os.path.exists(cfg.device_path())


def open_camera(cfg):
    """CameraConfig を開いて検証する。成功時 VideoCapture、失敗時 None。"""
    cap = cv2.VideoCapture(cfg.device, getattr(cv2, f'CAP_{cfg.backend}'))
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
        if not device_present(cfg):
            continue
        cap = open_camera(cfg)
        if cap is not None:
            return cfg, cap
    return None
