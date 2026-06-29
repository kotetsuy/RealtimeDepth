# RealtimeDepth — リアルタイム深度推定デモ (セットアップ手順書)

Depth Anything V2 Small を AMD Ryzen AI MAX+ 395 (gfx1150 / Strix Halo) 上で
ROCm + MIGraphX で動かし、USB カメラ映像を Chrome に MJPEG ストリーミング
するデモです。

このドキュメントは **`git clone` から `./start_all.sh` でブラウザ表示が
始まるところまで** の手順書です。技術的な内容は [TECHNICALJ.md](./TECHNICALJ.md)
を参照してください。

---

## 前提条件

| 項目 | バージョン / 状態 |
| --- | --- |
| マシン | GMKtec NucBox EVO X2 等 (Ryzen AI MAX+ 395, gfx1150) |
| OS | Ubuntu 24.04 |
| ROCm | 7.2.x がインストール済み (`/opt/rocm` から参照可能) |
| Python | 3.10 (Ubuntu 24.04 標準は 3.12 なので別途インストール) |
| USB カメラ | V4L2 で認識される単眼カメラ (`/dev/video0` 等) |
| ブラウザ | Google Chrome (NucBox 本体 or LAN 内別マシン) |

**ROCm 7.2.x が `/opt/rocm` 配下にインストールされていること**, および
**カメラが `ls /dev/video*` で見えること**が前提です。

---

## 1. リポジトリ取得

```bash
cd ~
git clone <このリポジトリの URL> RealtimeDepth
cd RealtimeDepth
```

> 以下の手順はすべて `~/RealtimeDepth` をカレントディレクトリとして実行
> する前提で書いています。

---

## 2. Python 3.10 を入れる

Ubuntu 24.04 の標準は Python 3.12 です。本プロジェクトは
onnxruntime-migraphx の cp310 wheel を使う都合で **3.10 が必須** です。

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
python3.10 --version    # 3.10.x が出ること
```

---

## 3. venv と Python 依存パッケージ

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# PyTorch (ONNX エクスポート用なので CPU 版で OK)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# ONNX 変換 + ROCm 7.2.1 向け onnxruntime
pip install onnx onnxscript
pip install -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ onnxruntime-migraphx

# 画像処理 / Web サーバー / その他
pip install opencv-python flask pyyaml huggingface_hub
```

確認:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# → ['MIGraphXExecutionProvider', 'CPUExecutionProvider'] が出れば OK
```

> **注意**: `onnxruntime-rocm` (PyPI 版) は ROCm 6.x ビルドのため ROCm 7.x
> では動きません。必ず上記の AMD 公式リポジトリから
> `onnxruntime-migraphx` を入れてください。詳細は
> [TECHNICALJ.md](./TECHNICALJ.md) の該当節を参照。

---

## 4. Depth-Anything-V2 のクローンとモデル取得

公式リポジトリは大きいので `$HOME` 直下にクローンし、本プロジェクトには
シンボリックリンクで参照します (本リポジトリの `Depth-Anything-V2` は
`~/Depth-Anything-V2` を指す symlink です)。

```bash
cd ~
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
mkdir -p checkpoints
wget -O checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

# 本プロジェクトから参照できるよう symlink を確認
ls -l ~/RealtimeDepth/Depth-Anything-V2
# → ~/Depth-Anything-V2 を指している symlink ならOK
# 無ければ:
#   ln -s ~/Depth-Anything-V2 ~/RealtimeDepth/Depth-Anything-V2
```

---

## 5. ONNX エクスポート

`~/Depth-Anything-V2/export_onnx.py` を以下の内容で作成:

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
    dynamo=False,    # ← MIGraphX が opset 18 の Resize 属性を未対応のため必須
)
```

実行:

```bash
cd ~/Depth-Anything-V2
source ~/RealtimeDepth/.venv/bin/activate
python export_onnx.py
ls -lh ~/RealtimeDepth/depth_anything_v2_vits_518.onnx   # ~95 MB
```

> `dynamo=False` を入れずに export すると MIGraphX が Resize op の
> `keep_aspect_ratio_policy` 属性を解釈できず実行時に落ちます。詳細は
> [TECHNICALJ.md](./TECHNICALJ.md) を参照。

---

## 6. config.yaml の調整 (カメラデバイス)

`~/RealtimeDepth/config.yaml` の `camera.device` をお手元のカメラに合わせて
書き換えます。

USB ポートが変わってもデバイス名が変わらないよう、**`/dev/v4l/by-id/`
配下の固定パス**を推奨します:

```bash
ls /dev/v4l/by-id/
# usb-XXXX_..._camera-video-index0   ← これを使う (index1 はメタ)
```

`config.yaml`:

```yaml
camera:
  device: /dev/v4l/by-id/usb-..._camera-video-index0
  width: 640
  height: 480
  fps: 30
```

整数 (`device: 0`) でもパス (`device: /dev/video0`) でも動きますが、
ポート差し替えに強いのは `by-id` 形式です。

---

## 7. 起動

```bash
cd ~/RealtimeDepth
./start_all.sh
```

実行されること:

1. venv activation + `HSA_OVERRIDE_GFX_VERSION=11.5.0` を内部設定
2. `app.py` をバックグラウンド起動 (PID は `.depth_app.pid` に保存)
3. 初回は MIGraphX のコンパイル待ち (~110 秒)、2回目以降は ~3 秒
4. 起動完了後、Chrome を新規ウィンドウで `http://localhost:8000/` に開く

期待出力:

```
started (pid 12345), log: /home/test/RealtimeDepth/depth_app.log
waiting for ready (cold start ~110s, cached ~3s)...
ready. open http://localhost:8000/ or http://172.23.0.7:8000/
launching Chrome...
```

Chrome に元映像と深度マップ (近=明 / 遠=暗) が横並びで表示され、
左上に FPS が出れば成功です。

---

## 8. 停止

```bash
./stop_all.sh
```

PID ファイル経由で `SIGTERM`、10 秒待って残ったら `SIGKILL`。

---

## 9. LAN 内別マシン (Mac 等) からアクセスする場合

NucBox の IP を確認:

```bash
ip route get 1.1.1.1 | awk '/src/ {print $7}'
```

ufw が active なら穴開け:

```bash
sudo ufw allow 8000/tcp
```

Mac の Chrome から `http://<NucBoxのIP>:8000/` でアクセス。

---

## トラブルシューティング (簡易)

| 症状 | 確認・対処 |
| --- | --- |
| `ROCMExecutionProvider` が出ない | 想定通り。本プロジェクトは `MIGraphXExecutionProvider` を使います |
| MIGraphX のコンパイルが毎回走る | `config.yaml` の `runtime.compile_cache_dir` が書き込み可能か確認 |
| `Cannot open camera ...` | `ls /dev/video*` でデバイス確認、別アプリが占有していないか確認 |
| Chrome が自動起動しない | `DISPLAY` / `WAYLAND_DISPLAY` 不在 (SSH 等)。表示された URL を手動で開く |
| FPS が出ない | `./stop_all.sh && rm -rf .migraphx_cache && ./start_all.sh` でキャッシュ再生成、`rocm-smi` で GPU 使用率確認 |

より詳細なトラブルシュートは [TECHNICALJ.md](./TECHNICALJ.md) を参照。
