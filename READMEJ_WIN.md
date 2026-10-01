# RealtimeDepth — Windows セットアップ・利用ガイド

[English](README_WIN.md) | [技術解説](TECHNICALJ_WIN.md) | [検証記録](WINDOWS.md) | [Linux版](READMEJ.md)

更新日: 2026-10-01

Depth Anything V2 SmallをAMD Ryzen AI MAX+ 395のGPUで実行し、カメラ映像と相対深度マップをブラウザへMJPEG配信するデモです。Windows版は **ROCm 7.2 / PyTorch 2.9.1 / Python 3.12** を使用します。深度は相対値であり、メートル単位の距離測定ではありません。

## 1. 前提条件

| 項目 | 検証した環境 |
| --- | --- |
| OS / CPU | Windows 11 / AMD Ryzen AI MAX+ 395 |
| GPU | Radeon 8060S Graphics、`gfx1151` |
| Python | 3.12.10 x64（wheelはPython 3.12用） |
| ドライバー | `32.0.31041.1004` |
| カメラ | OpenCVのMSMF / DSHOWで取得可能なカメラ |
| ツール | Git、PowerShell、ブラウザ |

AMD公式7.2手順ではAdrenalin 26.1.1が前提です。上記の実測ドライバー番号との対応は未確認です。導入先の対応条件は[AMDの導入手順](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installryz/windows/install-pytorch.html)と[対応表](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/windows/windows_compatibility.html)を参照してください。

ROCm SDKは専用仮想環境へPythonパッケージとして導入します。今回の検証ではシステム用HIP SDKインストーラーや追加のSDK tarballは使用していません。

## 2. リポジトリを用意する

Windows対応ファイルを含むチェックアウトを使ってください。作業ブランチは `windows-support` です。リモートへの公開状況はこの資料では保証していません。

```powershell
git clone https://github.com/kotetsuy/RealtimeDepth.git RealtimeDepth
Set-Location RealtimeDepth
```

既存の検証端末では取得済みです。

```powershell
Set-Location C:\temp\RealtimeDepth
```

以下のコマンドはすべてリポジトリ直下で実行します。

## 3. 環境をセットアップする

Python 3.12 x64とGitを先にインストールしてください。スクリプトを実行できる環境では次を使えます。

```powershell
.\setup_windows.ps1 -DownloadModel
```

Pythonの場所を指定する場合:

```powershell
.\setup_windows.ps1 -Python 'C:\path\to\Python312\python.exe' -DownloadModel
```

setupは `.venv-rocm72-windows` を作成し、依存導入、`pip check`、GPU確認、モデル取得・存在確認を行います。既存モデルは保持します。Python未指定時は、このリポジトリに `.python\python.exe` があればそれを使い、なければ `py -3.12` を使います。`.python` はこの検証端末のローカル配置であり、リポジトリには含まれません。

**検証範囲:** setupスクリプトは構文確認済みですが、一括実行は未検証です。環境構築は次の個別コマンドで成功しています。

### 個別コマンドで構築する場合

```powershell
py -3.12 -m venv .venv-rocm72-windows
.\.venv-rocm72-windows\Scripts\python.exe -m pip install --upgrade pip
.\.venv-rocm72-windows\Scripts\python.exe -m pip install -r requirements-rocm72-windows.txt
.\.venv-rocm72-windows\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-rocm72-windows\Scripts\python.exe -m pip check
```

検証端末では最初のコマンドの代わりに `.\.python\python.exe -m venv .venv-rocm72-windows` を使用しました。既存の仮想環境は作成し直す必要はありません。

モデルがまだない場合のみ取得します。

```powershell
git clone https://github.com/DepthAnything/Depth-Anything-V2.git Depth-Anything-V2
New-Item -ItemType Directory -Force Depth-Anything-V2/checkpoints
Invoke-WebRequest -Uri 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth' -OutFile 'Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth'
```

Windowsでは直接cloneし、symlinkは不要です。取得済みなら再cloneしません。

### PowerShellの実行が拒否される場合

この端末ではユーザーが次を実行した後、start / stopの動作を確認しました。

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

これは現在のユーザーの実行ポリシー変更です。管理対象の端末では管理者の方針に従ってください。ポリシーを変更しない場合は後述のPython直接起動を使えます。

## 4. GPU・推論を確認する

```powershell
$env:CONFIG_PATH = Join-Path $PWD 'config.windows.yaml'
.\.venv-rocm72-windows\Scripts\python.exe check_windows.py
.\.venv-rocm72-windows\Scripts\python.exe test_inference.py
```

診断ではHIPとGPUが利用でき、`gfx1151 GPU computation passed` が表示されることを確認します。単体推論は出力518×518とNaN / Infがないことを検査します。

## 5. 起動・表示・停止

```powershell
.\start_windows.ps1
```

ブラウザで **http://127.0.0.1:8000/** を開きます。左がカメラ映像、右が相対深度マップです。停止するときは次を実行します。

```powershell
.\stop_windows.ps1
```

両スクリプトは実機確認済みです。カメラなしでも起動でき、プレースホルダーを配信します。未接続起動後にカメラをつないで映像が復帰することも確認済みです。

| URL / ファイル | 内容 |
| --- | --- |
| `/` | 表示ページ |
| `/stats` | カメラ表示名とFPS。未接続時は `camera: null`、`fps: 0` |
| `/stream` | MJPEGストリーム |
| `.windows-state/stdout.log` / `stderr.log` | 起動・実行ログ |
| `.windows-state/server.json` | 起動したプロセスの識別情報 |

設定ファイルや起動待ち時間を指定する例:

```powershell
.\start_windows.ps1 -Config .\config.windows.yaml -TimeoutSeconds 180
```

### Python直接起動

```powershell
$env:CONFIG_PATH = Join-Path $PWD 'config.windows.yaml'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv-rocm72-windows\Scripts\python.exe app.py
```

この場合は起動端末で **Ctrl+C** で停止します。直接起動したプロセスは `stop_windows.ps1` の管理対象ではありません。仮想環境のactivateは不要です。

## 6. カメラ・推論設定

`config.windows.yaml` を編集します。カメラは `device: 0` のように整数で指定し、`camera.defaults.backend` は `MSMF` または `DSHOW` を選びます。各deviceにbackend、width、height、fpsを指定するとdefaultsを上書きします。複数deviceは登録順に探索します。

indexは恒久的な実機IDではなく、`name` は表示ラベルです。抜き差しや複数接続後は対応を確認してください。

モデルの既定値は `vits`、入力518、`fp16`、`compile: false`。入力サイズは14の倍数が必要です。モデル・重みの相対パスは設定ファイルのディレクトリ基準です。Python直接起動で `CONFIG_PATH` を省くとLinux用 `config.yaml` が選ばれます。

カメラ単体試験はアプリ停止中に実行します。

```powershell
.\.venv-rocm72-windows\Scripts\python.exe check_windows.py --camera MSMF
.\.venv-rocm72-windows\Scripts\python.exe check_windows.py --camera DSHOW
```

## 7. 動作実績とトラブル対応

2026-10-01、Small / 518 / fp16 / eagerで推論単体は平均 **12.6 ms（79.4 FPS）**、アプリの `/stats` は約 **36〜36.4 FPS** でした。推論単体は取得・描画・配信を含まず、`/stats` もブラウザ表示FPSの測定値ではありません。

| 症状 | 確認すること |
| --- | --- |
| 起動できない | `.windows-state/stderr.log` と専用PythonのGPU診断 |
| `NO CAMERA` | カメラ接続、Windowsのカメラアクセス設定、index、MSMF / DSHOW、他アプリの使用状況 |
| 既にサーバー状態がある | 管理中のサーバーはstopしてから再起動。識別不一致時はプロセスと状態ファイルを確認 |
| ポート使用中 | ポート8000の既存アプリを確認。別のRealtimeDepthを重複起動しない |
| モデル読み込み失敗 | ソース・重みの配置、encoderと重みの一致 |
| `xFormers not available` | 今回はこの表示のまま推論に成功 |
| WinError 4551 | Windows Code Integrityログを確認。PowerShell実行ポリシーとは別のDLL制御 |

ROCm 10はこの端末で `hipfft.dll` の読み込みを拒否されたため、Windowsの既定を7.2に変更しました。7.2の同DLLも署名表示は `NotSigned` ですが実行できました。署名済み配布物であるとの保証はしません。Smart App Controlは変更していません。

未検証: setupスクリプト全体、配信中の切断から再接続までの一連の動作、切断中の停止・再起動、複数カメラ識別、長時間安定性、torch.compile、Linux実機回帰。詳細は [WINDOWS.md](WINDOWS.md) を参照してください。
