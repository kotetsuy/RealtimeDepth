# RealtimeDepth — リアルタイム深度推定デモ (セットアップ手順書)

Depth Anything V2 Small を AMD Ryzen AI MAX+ 395 (gfx1151 / Strix Halo) 上で
**PyTorch (ROCm)** で動かし、USB カメラ映像を Chrome に MJPEG ストリーミング
するデモです。

このドキュメントは **`git clone` から `./start_all.sh` でブラウザ表示が
始まるところまで** の手順書です。技術的な内容は [TECHNICALJ.md](./TECHNICALJ.md)
を参照してください。

> **注意 (2026-07)**: 以前は ONNX Runtime + MIGraphX で推論していましたが、
> この経路は廃止しました。ROCm 7.14 / gfx1151 向けの MIGraphX が存在せず、
> 一方 PyTorch の ROCm wheel は自前の ROCm ランタイムを同梱するため
> `/opt/rocm` のバージョンに依存しないためです。詳細は
> [TECHNICALJ.md §2](./TECHNICALJ.md) を参照。

---

## 前提条件

| 項目 | バージョン / 状態 |
| --- | --- |
| マシン | GMKtec NucBox EVO X2 等 (Ryzen AI MAX+ 395, gfx1151) |
| OS | Ubuntu 26.04 (24.04 でも可) |
| GPU ドライバ | ディストロカーネルの in-tree `amdgpu`/KFD だけでよい |
| ROCm | **システムへのインストールは不要**。PyTorch wheel が自前のランタイムを同梱する |
| Python | 3.14 (Ubuntu 26.04 標準)。3.12 / 3.13 の wheel もある |
| USB カメラ | V4L2 で認識される単眼カメラ (`/dev/video0` 等) |
| ブラウザ | Google Chrome (NucBox 本体 or LAN 内別マシン) |

**GPU がカーネルから見えていること** (`ls /dev/kfd /dev/dri` が通る) と
**カメラが `ls /dev/video*` で見えること**が前提です。システム全体の
`/opt/rocm` があっても構いませんが、本プロジェクトは使いません。

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

## 2. Python の確認

Ubuntu 26.04 の標準は Python 3.14 で、本プロジェクトもこれを使います。
gfx1151 の wheel は **cp312 / cp313 / cp314** 向けが存在します。この範囲なら
どれでも構いませんが、**cp310 の wheel は存在しない**ので 3.10 では動きません。

```bash
python3.14 --version    # 3.14.x が出ること
```

---

## 3. venv と Python 依存パッケージ

最重要なのは **wheel の入手元** です。必ず AMD の gfx1151 専用 index を使います:

```
https://repo.amd.com/rocm/whl/gfx1151/
```

`download.pytorch.org` の ROCm index は使わないでください。あちらは
マルチアーキの kpack ビルドで、gfx1151 では実行時に
`hipErrorInvalidImage` / `kpack_load_code_object failed with error: 13`
になります。

```bash
# ここでは uv を使用。python3.14 -m venv + pip でも構いません。
uv venv --python 3.14 .venv-torch

VIRTUAL_ENV=$PWD/.venv-torch uv pip install \
  --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match --prerelease allow \
  torch==2.9.1+rocm7.13.0 torchvision==0.24.0+rocm7.13.0

VIRTUAL_ENV=$PWD/.venv-torch uv pip install flask opencv-python pyyaml
```

> **torch と torchvision はバージョン組で固定すること。** index には
> torchvision 0.24.0 / 0.25.0 / 0.26.0 が並んでいますが、torch 2.9.1 に
> 対応するのは **0.24.0** だけです。ズレると import 時に
> `RuntimeError: operator torchvision::nms does not exist` で落ちます。
>
> | torch | torchvision |
> | --- | --- |
> | 2.9.1 | 0.24.0 |
> | 2.10.0 | 0.25.0 |
> | 2.11.0 | 0.26.0 |
>
> torchvision は省略できません。`depth_anything_v2/dpt.py` が
> `from torchvision.transforms import Compose` を import しています。

GPU 認識の確認:

```bash
.venv-torch/bin/python -c "
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_properties(0).gcnArchName)"
# → True / gfx1151 が出れば OK
```

> **`HSA_OVERRIDE_GFX_VERSION` は設定しないこと。** この wheel は gfx1151
> ネイティブビルドなので、アーキを override すると壊れます。シェルの
> プロファイルで export されている場合に備え、`start_all.sh` は明示的に
> `unset` しています。

torch を入れると依存として `rocm-sdk-libraries-gfx1151` (wheel 専用の ROCm
ランタイム) も入ります。システムの `/opt/rocm` のバージョンが無関係なのは
このためです。

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

## 5. config.yaml のモデル設定を確認

**ONNX エクスポートの手順はありません。** `app.py` が symlink 経由で公式の
`depth_anything_v2` パッケージを直接 import し、`.pth` チェックポイントを
読み込みます。既定値は手順 4 でダウンロードしたものと一致しているはずです:

```yaml
model:
  repo: Depth-Anything-V2                # 手順 4 の symlink
  encoder: vits                          # vits / vitb / vitl / vitg
  checkpoint: Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth
  input_size: 518                        # 14 の倍数であること

runtime:
  device: cuda        # ROCm の HIP 層が CUDA API を受けるので gfx1151 でも "cuda" が正しい
  precision: fp16     # fp16 / fp32
  compile: false      # torch.compile。下記参照
```

パスは `config.yaml` からの相対で解決されます。より大きな encoder を使う
場合は `encoder` と `checkpoint` の **両方** を変更し、対応する `.pth` を
ダウンロードしてください。

`compile: true` にすると `torch.compile(mode="reduce-overhead")` が有効に
なります。起動のたびに 1〜2 分のコンパイルが走る一方、既定設定
(fp16 vits) では既に約 77 FPS 出ているため不要と判断し、既定は off です。

---

## 6. config.yaml の調整 (カメラデバイス)

`~/RealtimeDepth/config.yaml` にお手元のカメラを登録します。`camera.devices`
配下に **複数のカメラを優先順位付きで** 列挙でき、起動時には実際に
接続されているものが先頭優先で自動選択されます。さらに動作中の USB
抜き差しにも追従します (下記参照)。

USB ポートが変わってもデバイス名が変わらないよう、**`/dev/v4l/by-id/`
配下の固定パス**を推奨します:

```bash
ls /dev/v4l/by-id/
# usb-XXXX_..._camera-video-index0   ← これを使う (index1 はメタ)
```

`config.yaml`:

```yaml
camera:
  # 上から優先。接続されているものが選ばれる。
  devices:
    - name: 2K USB Camera        # ログ/画面表示用の任意ラベル
      device: /dev/v4l/by-id/usb-..._camera-video-index0
      width: 640                 # 省略時は下の defaults を使用
      height: 480
      fps: 30
    - name: 予備カメラ
      device: /dev/v4l/by-id/usb-..._other-video-index0
  # devices で width/height/fps を省略したときの既定値
  defaults:
    width: 640
    height: 480
    fps: 30
```

デバイスごとに整数 (`device: 0`) でもパス (`device: /dev/video0`) でも
動きますが、ポート差し替えに強く、ホットプラグ検出も確実なのは `by-id`
形式です。

> **後方互換**: 旧形式の単一指定もそのまま使えます:
>
> ```yaml
> camera:
>   device: /dev/v4l/by-id/usb-..._camera-video-index0
>   width: 640
>   height: 480
>   fps: 30
> ```
>
> 内部で 1 要素の `devices` リストに正規化されます。

### ホットプラグの挙動

- **起動時の自動選択**: リストの中で最初に接続されているカメラを開きます。
  1台も接続されていなくてもエラーで落ちず、「NO CAMERA」プレースホルダを
  配信して待機します。
- **動作中の抜き差し**: 配信中のカメラが抜かれるとプレースホルダ表示に
  切り替わり、ポーリングを続けます。登録済みカメラが再び挿されると自動的に
  再接続します (ブラウザの MJPEG ストリームは切れません)。
- **カメラの切り替え**: 別の登録済みカメラに移したい場合は、現在のカメラを
  抜いてください。次のスキャンで先頭優先のカメラが再選択されます。
- **複数同時接続**: 同時に複数刺さっている場合は、優先順位が最も高い1台
  だけを配信します。

---

## 7. 起動

```bash
cd ~/RealtimeDepth
./start_all.sh
```

実行されること:

1. `.venv-torch` を activate し、`HSA_OVERRIDE_GFX_VERSION` を `unset`
2. `app.py` をバックグラウンド起動 (PID は `.depth_app.pid` に保存)
3. モデルロード + GPU ウォームアップ待ち (約 7 秒。`runtime.compile` が
   有効なら 1〜2 分)
4. 起動完了後、Chrome を新規ウィンドウで `http://localhost:8000/` に開く

期待出力:

```
started (pid 12345), log: /home/test/RealtimeDepth/depth_app.log
waiting for ready (model load + warmup ~10s; torch.compile 有効時は 1〜2 分)...
ready (camera: 2K USB Camera). open http://localhost:8000/ or http://172.23.0.7:8000/
launching Chrome...
```

なお起動のたびに以下の無害な警告がログに出ますが、動作に影響はありません:
`xFormers not available` (DINOv2 の optional import)、
`xnack 'Off' was requested for a processor that does not support it`、
MIOpen の `gfx1151_20.HIP.fdb.txt` が読めないという警告。

登録済みカメラが1台も接続されていない場合もアプリは起動し、メッセージは
`ready (no camera connected; serving placeholder)` になります。カメラを
挿せば自動的にストリームが現れます。

Chrome に元映像と深度マップ (近=明 / 遠=暗) が横並びで表示され、
左上に FPS と選択中のカメラ名が出れば成功です。

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
| `hipErrorInvalidImage` / `kpack_load_code_object failed with error: 13` | wheel を `repo.amd.com/rocm/whl/gfx1151/` ではなく `download.pytorch.org` から入れている。手順 3 で入れ直す |
| `RuntimeError: operator torchvision::nms does not exist` | torch と torchvision のバージョン不一致。組で固定する (torch 2.9.1 ↔ torchvision 0.24.0) |
| `torch.cuda.is_available()` が `False` | `ls /dev/kfd /dev/dri` の確認と、ユーザーが `render` / `video` グループに入っているか確認。`HSA_OVERRIDE_GFX_VERSION` が export されて**いない**ことも確認 |
| 「NO CAMERA」プレースホルダから変わらない | 登録済みカメラが未接続。`ls /dev/v4l/by-id/` で `camera.devices` のいずれかに一致するパスがあるか、別アプリが占有していないか確認 |
| Chrome が自動起動しない | `DISPLAY` / `WAYLAND_DISPLAY` 不在 (SSH 等)。表示された URL を手動で開く |
| FPS が出ない | `.venv-torch/bin/python test_inference.py` で推論単体を切り分ける (fp16 vits 518² で約 12 ms / 78 FPS が目安)。`rocm-smi` で GPU 使用率を確認し、`runtime.precision` が `fp32` になっていないか確認 |

より詳細なトラブルシュートは [TECHNICALJ.md](./TECHNICALJ.md) を参照。
