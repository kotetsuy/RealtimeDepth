"""Windows setup diagnostics without loading the depth model."""
import argparse
import json
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', action='store_true')
    parser.add_argument('--camera', choices=['MSMF', 'DSHOW'])
    args = parser.parse_args()
    if args.port:
        import yaml
        with open(os.environ['CONFIG_PATH'], encoding='utf-8') as stream:
            print(yaml.safe_load(stream)['server']['port'])
    elif args.camera:
        import cv2
        from camera import CameraConfig, open_camera
        cap = open_camera(CameraConfig(0, 640, 480, 30, backend=args.camera))
        if cap is None:
            print(json.dumps({'backend': args.camera, 'connected': False}))
            return
        try:
            started = time.perf_counter()
            frames = sum(bool(cap.read()[0]) for _ in range(30))
            print(json.dumps({'backend': args.camera, 'connected': True,
                              'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                              'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                              'reported_fps': cap.get(cv2.CAP_PROP_FPS),
                              'frames': frames,
                              'measured_fps': frames / (time.perf_counter() - started)}))
        finally:
            cap.release()
    else:
        from runtime import configure_runtime
        import torch
        configure_runtime({})
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
        if arch != 'gfx1151':
            raise RuntimeError(f'Expected gfx1151, found {arch}')
        value = torch.ones(16, device='cuda').sum().item()
        assert value == 16
        print('gfx1151 GPU computation passed')


if __name__ == '__main__':
    main()
