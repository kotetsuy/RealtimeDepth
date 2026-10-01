"""Exercise real route functions with a placeholder worker, without GPU startup."""
import ast
from pathlib import Path
import threading
import unittest

import cv2
from flask import Flask, Response, render_template_string
import numpy as np

from camera import CameraConfig


class HttpTests(unittest.TestCase):
    def test_placeholder_routes(self):
        source = ast.parse(Path('app.py').read_text(encoding='utf-8'))
        names = {'make_placeholder', 'index', 'mjpeg_generator', 'stream', 'stats'}
        nodes = [node for node in source.body if
                 (isinstance(node, ast.FunctionDef) and node.name in names) or
                 (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'INDEX_HTML' for t in node.targets))]
        app = Flask(__name__)
        scope = dict(app=app, Response=Response, render_template_string=render_template_string,
                     cv2=cv2, np=np, CAMERA_CONFIGS=[CameraConfig(0, 640, 480, 30)], JPEG_QUALITY=80)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'app.py', 'exec'), scope)
        jpeg = scope['make_placeholder']('No camera')
        class Worker:
            frame_event = threading.Event()
            def get_jpeg(self):
                return jpeg
            def get_status(self):
                return {'fps': 0.0, 'camera': None}
        scope['worker'] = Worker()
        client = app.test_client()
        self.assertEqual(client.get('/').status_code, 200)
        self.assertEqual(client.get('/stats').json, {'fps': 0.0, 'camera': None})
        scope['worker'].frame_event.set()
        response = client.get('/stream', buffered=False)
        try:
            chunk = next(response.response)
            self.assertTrue(chunk.startswith(b'--frame\r\nContent-Type: image/jpeg'))
            self.assertIn(jpeg, chunk)
            decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, (480, 640, 3))
        finally:
            response.close()
