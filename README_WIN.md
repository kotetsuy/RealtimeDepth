# RealtimeDepth — Windows Setup and User Guide

[日本語](READMEJ_WIN.md) | [Technical guide](TECHNICAL_WIN.md) | [Validation log (Japanese)](WINDOWS.md) | [Linux guide](README.md)

Updated: 2026-10-01

This demo runs Depth Anything V2 Small on an AMD Ryzen AI MAX+ 395 GPU and streams camera images alongside relative depth maps to a browser over MJPEG. Windows uses **ROCm 7.2 / PyTorch 2.9.1 / Python 3.12**. Depth values are relative, not calibrated distances in meters.

## 1. Prerequisites

| Item | Validated environment |
| --- | --- |
| OS / CPU | Windows 11 / AMD Ryzen AI MAX+ 395 |
| GPU | Radeon 8060S Graphics, `gfx1151` |
| Python | 3.12.10 x64; the wheels target Python 3.12 |
| Driver | `32.0.31041.1004` |
| Camera | A camera accessible through OpenCV MSMF / DSHOW |
| Tools | Git, PowerShell, browser |

AMD's 7.2 instructions specify Adrenalin 26.1.1. The mapping between that release and the observed driver number above has not been verified. Check the [AMD installation instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installryz/windows/install-pytorch.html) and [support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/windows/windows_compatibility.html) for other machines.

ROCm SDK packages are installed into a dedicated Python environment. The validated setup did not use the system HIP SDK installer or an additional SDK tarball.

## 2. Obtain the repository

Use a checkout containing the Windows files. Development was performed on `windows-support`; this document does not guarantee that the branch has been published remotely.

```powershell
git clone https://github.com/kotetsuy/RealtimeDepth.git RealtimeDepth
Set-Location RealtimeDepth
```

On the existing test machine, the checkout is already available:

```powershell
Set-Location C:\temp\RealtimeDepth
```

Run all subsequent commands from the repository root.

## 3. Set up the environment

Install Python 3.12 x64 and Git first. Where PowerShell script execution is allowed, use:

```powershell
.\setup_windows.ps1 -DownloadModel
```

To specify the base Python executable:

```powershell
.\setup_windows.ps1 -Python 'C:\path\to\Python312\python.exe' -DownloadModel
```

Setup creates `.venv-rocm72-windows`, installs dependencies, runs `pip check` and GPU diagnostics, and downloads/checks the model. Existing model files are preserved. Without `-Python`, it uses `.python\python.exe` if present, otherwise `py -3.12`. The `.python` directory is local to the test machine and is not included in the repository.

**Validation scope:** the setup script passed syntax checks but has not been tested end to end. Environment installation succeeded using the individual commands below.

### Manual installation

```powershell
py -3.12 -m venv .venv-rocm72-windows
.\.venv-rocm72-windows\Scripts\python.exe -m pip install --upgrade pip
.\.venv-rocm72-windows\Scripts\python.exe -m pip install -r requirements-rocm72-windows.txt
.\.venv-rocm72-windows\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-rocm72-windows\Scripts\python.exe -m pip check
```

The test machine used `.\.python\python.exe -m venv .venv-rocm72-windows` for the first command. Do not recreate an environment that is already prepared.

Download the model only if it is missing:

```powershell
git clone https://github.com/DepthAnything/Depth-Anything-V2.git Depth-Anything-V2
New-Item -ItemType Directory -Force Depth-Anything-V2/checkpoints
Invoke-WebRequest -Uri 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth' -OutFile 'Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth'
```

Windows uses a direct clone without symlinks. Skip cloning if the source is already present.

### If PowerShell refuses to run scripts

The user enabled script execution on the test machine with:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

This changes the current user's execution policy. Follow administrator policy on managed devices. The direct Python method below is available without changing this setting.

## 4. Verify the GPU and inference

```powershell
$env:CONFIG_PATH = Join-Path $PWD 'config.windows.yaml'
.\.venv-rocm72-windows\Scripts\python.exe check_windows.py
.\.venv-rocm72-windows\Scripts\python.exe test_inference.py
```

Diagnostics should identify HIP and the GPU and print `gfx1151 GPU computation passed`. The inference benchmark checks for a 518×518 output with no NaN or infinity values.

## 5. Start, view, and stop

```powershell
.\start_windows.ps1
```

Open **http://127.0.0.1:8000/**. The camera image appears on the left and relative depth on the right. To stop:

```powershell
.\stop_windows.ps1
```

Both scripts have been verified on the machine. Starting without a camera produces a placeholder stream. Connecting the camera after that startup restored live video in the user's test.

| URL / file | Purpose |
| --- | --- |
| `/` | Viewer page |
| `/stats` | Camera label and FPS; disconnected: `camera: null`, `fps: 0` |
| `/stream` | MJPEG stream |
| `.windows-state/stdout.log` / `stderr.log` | Startup and runtime logs |
| `.windows-state/server.json` | Managed process identity |

To select a configuration and readiness timeout:

```powershell
.\start_windows.ps1 -Config .\config.windows.yaml -TimeoutSeconds 180
```

### Direct Python startup

```powershell
$env:CONFIG_PATH = Join-Path $PWD 'config.windows.yaml'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv-rocm72-windows\Scripts\python.exe app.py
```

Stop this process with **Ctrl+C** in the launching terminal. `stop_windows.ps1` does not manage a directly launched process. Virtual environment activation is unnecessary.

## 6. Camera and inference settings

Edit `config.windows.yaml`. Use a non-negative integer such as `device: 0` for a camera. Set `camera.defaults.backend` to `MSMF` or `DSHOW`. Per-device backend, width, height, and fps override defaults. Multiple devices are probed in listed order.

An index is not a permanent hardware identifier, and `name` is only a display label. Check device mapping after unplugging or connecting multiple cameras.

Defaults are `vits`, input size 518, `fp16`, and `compile: false`. Input size must be a multiple of 14. Model paths are relative to the configuration file's directory. Direct Python startup without `CONFIG_PATH` selects the Linux `config.yaml`.

Run standalone camera tests while the application is stopped:

```powershell
.\.venv-rocm72-windows\Scripts\python.exe check_windows.py --camera MSMF
.\.venv-rocm72-windows\Scripts\python.exe check_windows.py --camera DSHOW
```

## 7. Results and troubleshooting

On 2026-10-01, Small / 518 / fp16 / eager inference averaged **12.6 ms (79.4 FPS)**. Application `/stats` reported approximately **36–36.4 FPS**. Inference-only timing excludes capture, rendering, and streaming; `/stats` is also not a browser display FPS measurement.

| Symptom | Checks |
| --- | --- |
| Startup failure | `.windows-state/stderr.log` and GPU diagnostics with the dedicated Python |
| `NO CAMERA` | Connection, Windows camera access settings, index, MSMF / DSHOW, other camera applications |
| Existing server state | Stop a managed server before restarting; investigate process identity mismatches before modifying state |
| Port already in use | Check the application using port 8000; avoid duplicate servers |
| Model loading failure | Source/checkpoint paths and encoder/checkpoint compatibility |
| `xFormers not available` | Inference succeeded with this message in the tested configuration |
| WinError 4551 | Inspect Windows Code Integrity logs; DLL control is separate from PowerShell execution policy |

ROCm 10's `hipfft.dll` was blocked on this machine, so Windows now defaults to 7.2. Its DLL also reports `NotSigned`, but ran successfully. This is not a guarantee of a signed distribution. Smart App Control was not changed.

Not yet validated: full setup script, disconnect/reconnect while streaming, stop/restart during disconnection, multiple-camera identity, long-term stability, torch.compile, and physical Linux regression tests. See [WINDOWS.md](WINDOWS.md) for the detailed Japanese validation log.
