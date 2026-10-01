# RealtimeDepth — Windows Technical Guide

[日本語](TECHNICALJ_WIN.md) | [User guide](README_WIN.md) | [Validation log (Japanese)](WINDOWS.md)

Updated: 2026-10-01. This document describes the `windows-support` implementation and tests on Windows 11 / Ryzen AI MAX+ 395.

## 1. Architecture and responsibilities

```text
OpenCV MSMF / DSHOW → BGR frame
  → CPU: RGB conversion and resize to 518×518
  → GPU: normalization and Depth Anything V2 Small inference
  → CPU: depth resize, color mapping, side-by-side composition, JPEG
  → shared latest JPEG → Flask /stream → browser
```

| File | Responsibility |
| --- | --- |
| `app.py` | Configuration, model initialization, inference, camera worker, Flask routes |
| `camera.py` | Current/legacy camera configuration, OS backend and existence checks, open/probe |
| `runtime.py` | HIP/GPU validation and device/dtype selection |
| `check_windows.py` | gfx1151 computation check, backend camera probes, startup port lookup |
| `test_inference.py` | Single-input inference benchmark and shape/finite-value checks |
| `config.windows.yaml` | Complete Windows configuration |
| `requirements-rocm72-windows.txt` | Fixed official AMD URLs for the 7.2 SDK and matching torch/torchvision |
| `setup_windows.ps1` / `start_windows.ps1` / `stop_windows.ps1` | Environment and process management |
| `tests/test_camera.py` / `tests/test_http.py` | Regression tests without GPU or USB dependencies |

The model runs directly in PyTorch. ONNX Runtime / MIGraphX is not part of the current inference path. Flask runs its threaded development server with debugging and the reloader disabled.

## 2. Windows runtime

The environment is `.venv-rocm72-windows`, using Python 3.12 x64. ROCm packages are `7.2.0.dev0`, torch is `2.9.1+rocmsdk20260116`, and torchvision is `0.24.1+rocmsdk20260116`. The `dev0` suffix is the distribution version in [AMD's official 7.2 instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installryz/windows/install-pytorch.html).

`configure_runtime()` validates precision as fp16 or fp32. For a cuda device it checks the HIP build and GPU availability, resolves the device index, calls `set_device()`, and prints the GPU name and `gcnArchName`. ROCm uses the `torch.cuda` API. Strict gfx1151 validation and a small GPU computation belong to `check_windows.py`; the shared runtime does not reject other GPU architectures itself.

`HSA_OVERRIDE_GFX_VERSION` is removed before importing torch so the actual architecture is used. The driver and SDK are separate: installing Python packages does not install or update the GPU driver.

The original ROCm 10 environment failed with WinError 4551 when Smart App Control rejected `hipfft.dll`. The 7.2 DLL also reports `NotSigned`, but GPU inference succeeded on this machine. This is not evidence of improved signing. Smart App Control and DLL contents were unchanged, and the old environment was retained. The later user change to PowerShell RemoteSigned is a separate execution policy.

## 3. Model initialization and image processing

Configuration is read as UTF-8. Without `CONFIG_PATH`, Python uses `config.yaml`; the start script defaults to `config.windows.yaml`. Relative model.repo/checkpoint paths resolve against the configuration directory.

Small (`vits`) uses features=64 and out_channels=[48, 96, 192, 384]. The checkpoint is loaded onto CPU with `weights_only=True`; after loading the state dict, the model moves to the selected device/dtype and enters `eval()` mode. Input size must be a multiple of 14.

Startup runs three warmup inferences on a zero image, followed by CUDA synchronization when applicable. Loading and warmup occur at module scope, so a working model/GPU is required even for camera-free startup. `compile: true` selects `torch.compile(mode='reduce-overhead')`, which has not been validated on Windows here.

Preprocessing:

1. Convert BGR to RGB and resize directly to the square input size using INTER_CUBIC.
2. Transfer contiguous uint8 HWC data to the GPU, then rearrange it to NCHW.
3. Cast to the selected dtype, divide by 255, and normalize with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
4. Infer under `torch.inference_mode()` and return the first batch's depth as a float32 CPU NumPy array.

This is a direct square resize, not aspect-preserving letterboxing. Model output is `(1, S, S)` and the application uses `(S, S)`.

Postprocessing resizes depth to the camera resolution using INTER_LINEAR, normalizes between the frame's second and 98th percentiles, converts to 0–255, and applies INFERNO. A range below 1e-6 produces zeros. The original frame and depth visualization are concatenated horizontally: 640×480 input normally produces a 1280×480 JPEG. The default disconnected placeholder is 640×480.

Depth is relative, with larger values representing nearer regions. Colors are normalized per frame. Neither the values nor colors are calibrated distances in meters; the viewer's range wording does not establish measurement accuracy.

## 4. Camera states and probing

`CameraConfig` stores device, width, height, fps, name, and backend. Windows requires a non-negative integer index and MSMF or DSHOW, normalized to uppercase. Linux uses V4L2. Per-device values override defaults, and legacy `camera.device` configuration remains supported.

On Windows, `device_present()` does not inspect the filesystem; actual open/read determines availability. Linux maps integer indices to `/dev/video{N}` and checks path existence. Both pre-probe filtering and read-failure handling use this OS branch.

`open_camera()` constructs VideoCapture with the selected backend and requests width, height, and FPS. It validates isOpened and one trial read, releasing the capture on failure. Requested properties are not guaranteed to be accepted by hardware; the diagnostic reports actual values.

```text
DISCONNECTED
  ├─ probe in listed order every second → successful open/trial read → STREAMING
  └─ none found → publish placeholder, sleep 0.2 seconds
STREAMING
  ├─ successful read → reset failures, infer and publish JPEG
  └─ 10 consecutive read failures → release → DISCONNECTED
```

Linux also disconnects when the device path disappears. Intermediate Windows read failures sleep 0.01 seconds before retrying. Priority applies during probing; connecting a higher-priority device does not preempt a working camera.

If open/read never returns, timers and failure counters cannot advance. Camera I/O is not isolated in a separate process with a hang watchdog. Indices are not persistent hardware identifiers, and names are display labels only.

Validated behavior includes standalone MSMF/DSHOW capture and camera-free startup followed by attachment and video recovery with default MSMF. A full unplug/replug cycle during streaming, hardware priority with multiple cameras, and hangs remain untested.

## 5. Threads, HTTP, and shutdown

A daemon `DepthWorker` sequentially performs capture through JPEG encoding. A Lock protects the latest JPEG, FPS, and camera label. There is no FIFO frame queue; slow clients can skip intermediate frames.

`_publish()` updates shared state and sets an Event. Each MJPEG generator waits for at most one second, clears the Event, and reads the latest JPEG. The shared Event is not a per-client delivery guarantee; multi-client fairness and performance have not been evaluated.

| Route | Response |
| --- | --- |
| `/` | HTML with an img referencing `/stream` |
| `/stats` | `fps` rounded to two decimals and `camera`; disconnected: 0 / null |
| `/stream` | `multipart/x-mixed-replace; boundary=frame`, each part includes image/jpeg and Content-Length |

For direct startup, Flask's surrounding finally calls `worker.stop()`, sets running=False, and joins for up to two seconds. Camera release occurs after the worker loop exits. A hung read or forced termination does not guarantee completion of this cleanup.

The start script launches the dedicated Python with a hidden window and redirects stdout/stderr to `.windows-state`. It checks loopback port availability, records PID, start time, Python path, and configuration in server.json, then watches process exit while waiting for `/stats`. The default timeout is 180 seconds. Camera connection is not a readiness requirement.

Stop checks PID, start time, and executable path before forcefully terminating the process with `Stop-Process`. It is not an API requesting graceful worker shutdown. The user verified start/stop, but stop/restart during disconnection remains untested. The setup script has only passed syntax checks; its individual installation commands were validated.

## 6. Configuration reference

| Setting | Windows default / meaning |
| --- | --- |
| camera.devices | `name: USB Camera`, `device: 0`; probe in listed order |
| camera.defaults | backend=MSMF, width=640, height=480, fps=30 |
| model.repo | `Depth-Anything-V2` |
| model.encoder | `vits`; code also defines vitb/vitl/vitg, not tested here |
| model.checkpoint | `Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth` |
| model.input_size | 518, must be a multiple of 14 |
| server.host / port | `127.0.0.1` / 8000 |
| server.jpeg_quality | 80 |
| runtime.device / precision / compile | cuda / fp16 / false |

Windows and Linux use separate configuration files. Linux shell scripts, requirements, and configuration are preserved. Shared OS branches have mock tests, but physical Linux regression testing remains outstanding.

## 7. Performance and reproducibility

| Measurement | Result / scope |
| --- | --- |
| Inference only | 12.6 ms, 79.4 FPS; same preprocessed input, three warmups then 30 iterations, GPU synchronization |
| Application `/stats` | Approximately 36–36.4 FPS; smoothed inverse time from capture through inference, coloring, and composition |
| Camera-only MSMF / DSHOW | 640×480, 30 successful frames; short samples around 32.0 / 25.0 FPS |

The `/stats` timer ends before text overlay and JPEG encoding. Its exponential average uses 0.9 for the previous value and 0.1 for the new inverse processing time. It is not a full pipeline FPS measurement including JPEG, HTTP, and browser rendering.

`test_inference.py` chooses test.jpg in the configuration directory, then camera index 0 with an automatic backend, then a synthetic image with seed=0. It does not use the application's camera configuration for this selection. Preprocessing happens once before timing model calls; the final output is checked for shape and finite values. The recorded run used camera input, not a fixed Linux comparison image.

For reproduction, match test.jpg, checkpoint, configuration, precision, and source commit. Separate before/after Linux comparisons from cross-OS comparisons.

- Model source commit: `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`
- Checkpoint SHA256: `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`
- Observed HIP: `7.2.26024-f6f897bd3d`

## 8. Tests and remaining validation

```powershell
.\.venv-rocm72-windows\Scripts\python.exe -m unittest discover -s tests -v
```

Six tests passed, covering OS branches, legacy configuration/paths, probing priority, release after open failure, rescan after ten read failures, and placeholder HTTP behavior. Worker and route definitions are extracted from app.py using AST to avoid model initialization, so these are not complete application startup tests.

Hardware validation covers GPU computation, inference, HTTP/MJPEG, user-confirmed display, start/stop, and attachment after camera-free startup. Remaining work includes full setup-script execution, a full streaming unplug/replug cycle, disconnected shutdown/restart, multiple cameras and hangs, long-duration operation, torch.compile, and physical Linux regression tests.
