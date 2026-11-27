# RealtimeDepth — realtime depth estimation demo (setup guide)

A PyTorch ROCm demo for Depth Anything V2 and Depth Pro 2 on AMD Ryzen AI MAX+ 395 (gfx1151). It streams a USB camera or playlist video alongside its depth map to a browser.

This document walks you from **`git clone` to `./start_all.sh` showing the
stream in your browser**. For architecture and design rationale, see
[TECHNICAL.md](./TECHNICAL.md).

A Japanese version of this guide is available as
[READMEJ.md](./READMEJ.md).

> **Note (2026-07)**: this project used to run inference through
> ONNX Runtime + MIGraphX. That path is retired — no MIGraphX build
> exists for gfx1151 on ROCm 7.14, and PyTorch's ROCm wheels ship their
> own ROCm runtime, so they don't care what version lives in
> `/opt/rocm`. See [TECHNICAL.md §2](./TECHNICAL.md) for the full story.

---

## Prerequisites

| Item | Required state |
| --- | --- |
| Machine | GMKtec NucBox EVO X2 or similar (Ryzen AI MAX+ 395, gfx1151) |
| OS | Ubuntu 26.04 (24.04 also works) |
| GPU driver | In-tree `amdgpu`/KFD from the distro kernel — that's all you need |
| ROCm | **Not required as a system install.** The PyTorch wheel bundles its own runtime |
| Python | 3.14 (Ubuntu 26.04 ships it). 3.12 / 3.13 also have wheels |
| USB camera | A monocular V4L2 camera (`/dev/video0`, etc.) |
| Browser | Google Chrome (on the NucBox itself or another LAN host) |

This guide assumes **the GPU is visible to the kernel** (`ls /dev/kfd
/dev/dri` succeeds) and **the camera is visible via `ls /dev/video*`**.
A system-wide `/opt/rocm` may be present but is neither used nor required
by this project.

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

## 2. Check your Python

Ubuntu 26.04 ships Python 3.14, which is what this project uses. The
gfx1151 wheels exist for **cp312 / cp313 / cp314** — anything in that
range is fine, but there is **no cp310 wheel**, so a 3.10 interpreter
will not work.

```bash
python3.14 --version    # expect 3.14.x
```

---

## 3. Python environment for ROCm 10

### Environment for both models

The AMD stable wheels for gfx1151 can be installed in a separate environment. The existing 10.0 environment remains available.

```bash
bash setup_depthpro2.sh
.venv-depthpro2/bin/python test_inference.py
./start_all.sh
```

Python 3.14 and `uv` are required. `setup_depthpro2.sh` installs the pinned ROCm 10.2 wheel pair and the source under `vendor/depth-pro-2`. The instructions below describe the older Depth Anything V2 environment.

Use AMD ROCm 10.0.0 builds of PyTorch 2.13.0 and torchvision 0.28.0.
The `device-gfx1151` extra installs Radeon 8060S kernels.

```bash
bash setup_rocm10.sh
.venv-rocm10/bin/python test_inference.py
VENV_DIR="$PWD/.venv-rocm10" ./start_all.sh
```

Requires `uv` and Python 3.14. Setup creates `.venv-rocm10` and leaves
the old `.venv-torch` unchanged. Set `VENV_DIR` for a different environment
(use the same value during setup and launch).

Versions and the AMD index are pinned in `requirements-rocm10.txt`.
Use `https://stable.repo.amd.com/rocm/whl-next/` instead of the old ROCm 7 index.
Leave `HSA_OVERRIDE_GFX_VERSION` unset and use `runtime.device: cuda`.
Updating `/opt/rocm` alone does not update Python's HIP runtime.
GPU execution requires access to `/dev/kfd` and `/dev/dri`.

[AMD PyTorch installation](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)

---

## 4. Clone Depth-Anything-V2 and download the checkpoint

The official repo is large, so clone it outside this project and point at
it with a symlink. `app.py` puts that path on `sys.path` to import the
`depth_anything_v2` package, and loads the `.pth` from underneath it.

```bash
cd ~
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
mkdir -p checkpoints
wget -O checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

**Create the symlink yourself.** It is deliberately *not* tracked by git —
its target is an absolute path that differs per machine, so committing it
would break every other checkout.

```bash
ln -s ~/Depth-Anything-V2 ~/RealtimeDepth/Depth-Anything-V2

# Verify it resolves and the checkpoint is reachable
ls -l ~/RealtimeDepth/Depth-Anything-V2
ls ~/RealtimeDepth/Depth-Anything-V2/checkpoints/
```

If you'd rather clone the official repo straight into `~/RealtimeDepth/`
instead of symlinking, that works too — `model.repo` in `config.yaml` is
just a path relative to the config file. Either way the name
`Depth-Anything-V2` is gitignored.

---

## 5. Check the model settings in `config.yaml`

There is **no ONNX export step** — `app.py` imports the official
`depth_anything_v2` package straight from the symlinked repo and loads
the `.pth` checkpoint. The defaults should already match what you
downloaded in step 4:

```yaml
model:
  repo: Depth-Anything-V2                # the symlink from step 4
  encoder: vits                          # vits / vitb / vitl / vitg
  checkpoint: Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth
  input_size: 518                        # must be a multiple of 14

runtime:
  device: cuda        # ROCm's HIP layer answers to the CUDA API, so "cuda" is correct on gfx1151
  precision: fp16     # fp16 / fp32
  compile: false      # torch.compile; see below
```

Paths are resolved relative to `config.yaml`. If you use a larger
encoder, change **both** `encoder` and `checkpoint` and download the
matching `.pth`.

`compile: true` enables `torch.compile(mode="reduce-overhead")`. It costs
1–2 minutes of compilation on every start and is unnecessary at the
default settings (fp16 vits already runs at ~77 FPS), so it ships off.

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
  opened. If none is connected, the app loops through `videos/playlist.m3u8`.
  It serves a "NO CAMERA" placeholder only when no video is playable.
- **Unplug / replug while running**: if the active camera is removed,
  the stream switches to the playlist and the app keeps polling;
  when a registered camera is plugged back in it reconnects
  automatically — the browser MJPEG stream never drops.
- **Video credits**: the current entry's `#CREDIT:` appears in small text
  below the stream.
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

1. activates `.venv-depthpro2` when present, otherwise `.venv-rocm10.2` and `unset`s `HSA_OVERRIDE_GFX_VERSION` (set `VENV_DIR` to use another environment)
2. starts `app.py` in the background (PID written to `.depth_app.pid`)
3. waits for the model load + GPU warmup (~7 s; 1–2 min if
   `runtime.compile` is on)
4. once ready, opens Chrome to `http://localhost:8000/` in a new window

Expected output:

```
started (pid 12345), log: /home/test/RealtimeDepth/depth_app.log
waiting for ready (model load + warmup ~10s; 1-2 min with torch.compile)...
ready (camera: 2K USB Camera). open http://localhost:8000/ or http://172.23.0.7:8000/
launching Chrome...
```

The log also carries a few harmless warnings on every start:
`xFormers not available` (DINOv2 imports it optionally),
`xnack 'Off' was requested for a processor that does not support it`, and
a MIOpen note about a missing `gfx1151_20.HIP.fdb.txt` tuning database.
None of them affect the run.

If no registered camera is connected, the app still starts and the
message reads `ready (no camera connected; serving video playlist if available)`.
Plug in a camera to switch back automatically.

You should see the original camera feed and the depth map (bright = near,
dark = far) side-by-side in Chrome, with an FPS counter and the selected
camera name overlaid in the top-left.

---

## Switching models

Use the slider above the video to select Depth Anything V2 or Depth Pro 2. Depth Anything V2 displays relative depth; Depth Pro 2 estimates metric depth. Depth Pro 2 loads on first selection. Its color scale starts at 0.3–10 m and expands its upper limit for distant scenes. The color bar shows the current range; set `depth_pro_2.auto_range: false` in `config.yaml` for a fixed range. Estimated distances are not guaranteed to be accurate. Model size, precision, input size, window and range are configured under `depth_pro_2`.

`start_all.sh` prefers `.venv-depthpro2`, falling back to `.venv-rocm10.2` if absent. The fallback must have Depth Pro 2 installed to use that model. `VENV_DIR` and `TORCH_HOME` override the environment and cache paths; the default cache is `.cache/torch`. Source and license files are in `vendor/depth-pro-2`. The Python interpreter backing the venv and the Depth Anything V2 repository may be outside this directory. `~/RealtimeDepth2` is not needed by the default setup. Model weights not already cached are downloaded on first use.

Depth Pro 2 model weights are limited to noncommercial scientific research and academic development. Read `vendor/depth-pro-2/LICENSE_MODEL`, `LICENSE`, and `ACKNOWLEDGEMENTS` before use.

Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.

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
| `hipErrorInvalidImage` / `kpack_load_code_object failed with error: 13` | The wheel came from `download.pytorch.org` instead of `stable.repo.amd.com/rocm/whl-next/`. Reinstall per step 3. |
| `RuntimeError: operator torchvision::nms does not exist` | torch/torchvision version mismatch. Pin them as a pair (torch 2.13.0 ↔ torchvision 0.28.0). |
| `torch.cuda.is_available()` is `False` | Check `ls /dev/kfd /dev/dri` and that your user is in the `render` / `video` groups. Also make sure `HSA_OVERRIDE_GFX_VERSION` is **not** exported. |
| Stream stuck on the "NO CAMERA" placeholder | No registered camera is connected. Run `ls /dev/v4l/by-id/` and confirm a path matching a `camera.devices` entry exists and no other app is holding the device. |
| Chrome doesn't auto-launch | `DISPLAY` / `WAYLAND_DISPLAY` is missing (e.g., over SSH). Open the printed URL manually. |
| Low FPS | Run `.venv-depthpro2/bin/python test_inference.py` to isolate inference from camera I/O. Watch `rocm-smi` for GPU utilization, and confirm `runtime.precision` is `fp16`, not `fp32`. |

For deeper diagnostics, see [TECHNICAL.md](./TECHNICAL.md).
