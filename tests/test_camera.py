"""OS branches and reconnect behavior, with no GPU or USB dependency."""
import ast
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

import camera


class CameraTests(unittest.TestCase):
    def test_linux_legacy_and_paths(self):
        with patch.object(camera.os, 'name', 'posix'):
            cfg = camera.load_camera_configs({'camera': {'device': 2, 'width': 800}})[0]
            self.assertEqual((cfg.device_path(), cfg.width, cfg.backend), ('/dev/video2', 800, 'V4L2'))
            for path in ['/dev/video3', '/dev/v4l/by-id/usb-camera']:
                cfg = camera.CameraConfig(path, 640, 480, 30)
                with patch.object(camera.os.path, 'exists', return_value=False) as exists, patch.object(camera, 'open_camera') as opened:
                    self.assertIsNone(camera.find_connected_camera([cfg]))
                    exists.assert_called_once_with(path)
                    opened.assert_not_called()

    def test_windows_probe_and_priority(self):
        with patch.object(camera.os, 'name', 'nt'):
            configs = camera.load_camera_configs({'camera': {'devices': [{'device': 0}, {'device': 1}], 'defaults': {'backend': 'dshow'}}})
            self.assertEqual(configs[0].backend, 'DSHOW')
            cap = Mock()
            with patch.object(camera.os.path, 'exists') as exists, patch.object(camera, 'open_camera', side_effect=[None, cap]) as opened:
                self.assertEqual(camera.find_connected_camera(configs), (configs[1], cap))
                self.assertEqual(opened.call_count, 2)
                exists.assert_not_called()

    def test_backend_validation(self):
        for system, invalid in [('nt', 'V4L2'), ('posix', 'MSMF')]:
            with patch.object(camera.os, 'name', system), self.assertRaises(ValueError):
                camera.CameraConfig(0, 640, 480, 30, backend=invalid)
        with patch.object(camera.os, 'name', 'nt'), self.assertRaises(ValueError):
            camera.CameraConfig('/dev/video0', 640, 480, 30)

    def test_open_failure_releases_capture(self):
        with patch.object(camera.os, 'name', 'nt'):
            cfg = camera.CameraConfig(0, 640, 480, 30)
        for opened in [False, True]:
            cap = Mock()
            cap.isOpened.return_value = opened
            cap.read.return_value = (False, None)
            with patch.object(camera.cv2, 'VideoCapture', return_value=cap) as factory:
                self.assertIsNone(camera.open_camera(cfg))
                factory.assert_called_once_with(0, camera.cv2.CAP_MSMF)
                cap.release.assert_called_once()

    def test_worker_disconnects_then_retries(self):
        # Execute the actual worker class without app.py's model-loading side effects.
        source = Path('app.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'DepthWorker')
        cfg = Mock(name='camera-config')
        cap = Mock()
        cap.read.return_value = (False, None)
        clock = Mock()
        clock.time.side_effect = range(100)
        scope = {'threading': threading, 'time': clock, 'CAMERA_CONFIGS': [cfg],
                 'device_present': lambda cfg: True, 'make_placeholder': lambda _: b'jpeg'}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'app.py', 'exec'), scope)
        worker = scope['DepthWorker'].__new__(scope['DepthWorker'])
        worker.running = True
        worker._publish = Mock()
        def scan(configs):
            if not cap.read.call_count:
                return cfg, cap
            worker.running = False
            return None
        scope['find_connected_camera'] = Mock(side_effect=scan)
        worker._loop()
        self.assertEqual(cap.read.call_count, 10)
        cap.release.assert_called_once()
        self.assertEqual(scope['find_connected_camera'].call_count, 2)
        worker._publish.assert_called_with(b'jpeg', fps=0.0, name=None)


if __name__ == '__main__':
    unittest.main()
