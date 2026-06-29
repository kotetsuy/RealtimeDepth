# RealtimeDepth — realtime depth estimation demo (setup guide)

A demo that runs Depth Anything V2 Small on an AMD Ryzen AI MAX+ 395
(gfx1150 / Strix Halo) via ROCm + MIGraphX, and streams depth-mapped
USB camera footage to Chrome over MJPEG.

This document walks you from **`git clone` to `./start_all.sh` showing the
stream in your browser**. For architecture and design rationale, see
[TECHNICAL.md](./TECHNICAL.md).

A Japanese version of this guide is available as
[READMEJ.md](./READMEJ.md).

---

## Prerequisites

| Item | Required state |
| --- | --- |
| Machine | GMKtec NucBox EVO X2 or similar (Ryzen AI MAX+ 395, gfx1150) |
| OS | Ubuntu 24.04 |
| ROCm | 7.2.x already installed (reachable via `/opt/rocm`) |
| Python | 3.10 (Ubuntu 24.04 ships 3.12, so install 3.10 separately) |
| USB camera | A monocular V4L2 camera (`/dev/video0`, etc.) |
| Browser | Google Chrome (on the NucBox itself or another LAN host) |

This guide assumes **ROCm 7.2.x is already installed under `/opt/rocm`**
and **the camera is visible via `ls /dev/video*`**.

---

## 1. Clone the repository

```bash
cd ~
git clone <this repo URL> RealtimeDepth
cd RealtimeDepth
```

> All commands below assume `~/RealtimeDepth` as the current working
> directory.

---

## 2. Install Python 3.10

Ubuntu 24.04 ships Python 3.12. We need **3.10** because the
`onnxruntime-migraphx` cp310 wheel is the one that lines up with our
ROCm 7.2.1 stack.

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
python3.10 --version    # expect 3.10.x
```

---

## 3. Create the venv and install Python dependencies

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# PyTorch is only used for ONNX export, so the CPU build is fine.
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# ONNX exporter + ROCm 7.2.1-built onnxruntime
pip install onnx onnxscript
pip install -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ onnxruntime-migraphx

# Image processing / web server / misc
pip install opencv-python flask pyyaml huggingface_hub
```

Verify:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expect: ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
```

> **Note**: do not install `onnxruntime-rocm` from PyPI — it is built
> against ROCm 6.x and fails to load on ROCm 7.x. Always use
> `onnxruntime-migraphx` from the AMD repository above. See
> [TECHNICAL.md](./TECHNICAL.md) for the full story.

---

## 4. Clone Depth-Anything-V2 and download the checkpoint

The official repo is large, so we clone it under `$HOME` and reference it
from this project via a symlink (this repo's `Depth-Anything-V2` already
points at `~/Depth-Anything-V2`).

```bash
cd ~
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
mkdir -p checkpoints
wget -O checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

# Confirm the symlink resolves
ls -l ~/RealtimeDepth/Depth-Anything-V2
# Should point to ~/Depth-Anything-V2. If it doesn't:
#   ln -s ~/Depth-Anything-V2 ~/RealtimeDepth/Depth-Anything-V2
```

---

## 5. Export to ONNX

Create `~/Depth-Anything-V2/export_onnx.py` with the following content:

```python
import torch
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64,
             'out_channels': [48, 96, 192, 384]},
}

encoder = 'vits'
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(
    f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model.eval()

INPUT_SIZE = 518
dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

torch.onnx.export(
    model, dummy_input,
    f'../RealtimeDepth/depth_anything_v2_vits_{INPUT_SIZE}.onnx',
    input_names=['input'], output_names=['depth'],
    opset_version=17,
    dynamic_axes={'input': {0: 'batch'}, 'depth': {0: 'batch'}},
    dynamo=False,    # required: MIGraphX cannot parse opset-18 Resize attributes
)
```

Run it:

```bash
cd ~/Depth-Anything-V2
source ~/RealtimeDepth/.venv/bin/activate
python export_onnx.py
ls -lh ~/RealtimeDepth/depth_anything_v2_vits_518.onnx   # ~95 MB
```

> Without `dynamo=False`, MIGraphX cannot handle the `keep_aspect_ratio_policy`
> attribute on the Resize op and will fail at run time. See
> [TECHNICAL.md](./TECHNICAL.md) for details.

---

## 6. Adjust `config.yaml` (camera devices)

Edit `~/RealtimeDepth/config.yaml` to register your camera(s). You can
list **several cameras in priority order** under `camera.devices`; at
startup the app automatically selects the first one that is actually
connected, and follows USB hot-plug events at run time (see below).

To survive USB-port changes, **prefer the stable path under
`/dev/v4l/by-id/`**:

```bash
ls /dev/v4l/by-id/
# usb-XXXX_..._camera-video-index0   ← use this (index1 is metadata)
```

`config.yaml`:

```yaml
camera:
  # Listed in priority order. The first connected one is picked.
  devices:
    - name: 2K USB Camera        # free-form label, shown in logs / overlay
      device: /dev/v4l/by-id/usb-..._camera-video-index0
      width: 640                 # optional; falls back to defaults below
      height: 480
      fps: 30
    - name: Spare Camera
      device: /dev/v4l/by-id/usb-..._other-video-index0
  # Used when a device entry omits width/height/fps.
  defaults:
    width: 640
    height: 480
    fps: 30
```

Per device, both an integer (`device: 0`) and a string path
(`device: /dev/video0`) are accepted, but the `by-id` form is
robust to plugging the camera into a different port — and is
recommended for reliable hot-plug detection.

> **Backward compatible**: the old single-camera form is still accepted:
>
> ```yaml
> camera:
>   device: /dev/v4l/by-id/usb-..._camera-video-index0
>   width: 640
>   height: 480
>   fps: 30
> ```
>
> It is normalized internally to a one-entry `devices` list.

### Hot-plug behaviour

- **Auto-select on start**: the first connected camera from the list is
  opened. If none is connected, the app still starts and serves a
  "NO CAMERA" placeholder instead of crashing.
- **Unplug / replug while running**: if the active camera is removed,
  the stream switches to the placeholder and the app keeps polling;
  when a registered camera is plugged back in it reconnects
  automatically — the browser MJPEG stream never drops.
- **Switching cameras**: to move to a different registered camera,
  unplug the current one; on the next scan the app re-selects the
  highest-priority connected camera.
- **Multiple cameras connected**: only **one** camera (the
  highest-priority connected entry) is streamed at a time.

---

## 7. Start the demo

```bash
cd ~/RealtimeDepth
./start_all.sh
```

What this does:

1. activates the venv and exports `HSA_OVERRIDE_GFX_VERSION=11.5.0`
2. starts `app.py` in the background (PID written to `.depth_app.pid`)
3. waits for MIGraphX compile (~110 s on first run, ~3 s when cached)
4. once ready, opens Chrome to `http://localhost:8000/` in a new window

Expected output:

```
started (pid 12345), log: /home/test/RealtimeDepth/depth_app.log
waiting for ready (cold start ~110s, cached ~3s)...
ready (camera: 2K USB Camera). open http://localhost:8000/ or http://172.23.0.7:8000/
launching Chrome...
```

If no registered camera is connected, the app still starts and the
message reads `ready (no camera connected; serving placeholder)` — plug
a camera in and the stream appears automatically.

You should see the original camera feed and the depth map (bright = near,
dark = far) side-by-side in Chrome, with an FPS counter and the selected
camera name overlaid in the top-left.

---

## 8. Stop the demo

```bash
./stop_all.sh
```

This sends `SIGTERM` to the PID file process, waits 10 s, and falls back
to `SIGKILL` if needed.

---

## 9. Accessing from another LAN host (Mac, etc.)

Find the NucBox's IP:

```bash
ip route get 1.1.1.1 | awk '/src/ {print $7}'
```

If `ufw` is active, open the port:

```bash
sudo ufw allow 8000/tcp
```

Then from your Mac's Chrome: `http://<NucBox-IP>:8000/`.

---

## Troubleshooting (quick)

| Symptom | What to check |
| --- | --- |
| `ROCMExecutionProvider` not in providers list | Expected — this project uses `MIGraphXExecutionProvider`. |
| MIGraphX compile runs every time | Confirm `runtime.compile_cache_dir` in `config.yaml` is set and writable. |
| Stream stuck on the "NO CAMERA" placeholder | No registered camera is connected. Run `ls /dev/v4l/by-id/` and confirm a path matching a `camera.devices` entry exists and no other app is holding the device. |
| Chrome doesn't auto-launch | `DISPLAY` / `WAYLAND_DISPLAY` is missing (e.g., over SSH). Open the printed URL manually. |
| Low FPS | `./stop_all.sh && rm -rf .migraphx_cache && ./start_all.sh` to rebuild the cache; check `rocm-smi` for GPU utilization. |

For deeper diagnostics, see [TECHNICAL.md](./TECHNICAL.md).
