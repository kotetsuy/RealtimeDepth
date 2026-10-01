# RealtimeDepth Windows セットアップ・検証記録

更新日: 2026-10-01 / 作業ブランチ: `windows-support`

## 現在の状態

Windows 11 / Ryzen AI MAX+ 395で、**ROCm 7.2によるGPU推論とカメラ映像のブラウザ配信を確認済み**。
PowerShellスクリプトによる起動・停止もユーザーが動作確認した。
その後、カメラ未接続で `start_windows.ps1` により起動し、プレースホルダー配信を確認した。
起動中にカメラを接続すると映像が復帰することをユーザーが確認済み。
この記録の更新時点ではアプリは起動中（`http://127.0.0.1:8000/`）。

Windowsの既定環境は `.venv-rocm72-windows`。Linux版は従来のROCm 10構成を維持する。
旧Windows用ROCm 10環境は比較・調査用に保持しているが、通常起動には使用しない。
配信中のカメラ切断から再接続までの一連の試験やLinux実機回帰など、未検証項目は末尾に記載する。

## 検証済み環境

| 項目 | 実測・導入内容 |
| --- | --- |
| CPU | AMD Ryzen AI MAX+ 395 |
| GPU / アーキテクチャ | AMD Radeon 8060S Graphics / `gfx1151` |
| 既存GPUドライバー | `32.0.31041.1004`（変更なし） |
| Python | 3.12.10 x64、公式署名付き実行ファイルを `.python` に配置 |
| 仮想環境 | `.venv-rocm72-windows` |
| ROCmパッケージ | `7.2.0.dev0`（core / devel / libraries-customも同じ版） |
| HIP実測 | `7.2.26024-f6f897bd3d` |
| PyTorch | `2.9.1+rocmsdk20260116` |
| torchvision | `0.24.1+rocmsdk20260116` |
| 共通依存 | Flask 3.1.3 / OpenCV 4.14.0.94 / PyYAML 6.0.3 / NumPy 2.5.3 |
| モデル | Depth Anything V2 Small（`vits`） |
| 推論設定 | 入力518、`cuda`、`fp16`、`compile: false` |
| サーバー | `127.0.0.1:8000` |

`7.2.0.dev0` はAMD公式7.2導入手順に掲載されたwheelのバージョン表記。
SDKはPythonパッケージとして導入しており、システム用HIP SDKインストーラーの実行、
SDK tarballの追加展開、システムPATH変更は行っていない。アプリはtorchaudioを使用しない。

AMD公式7.2手順はAdrenalin 26.1.1を前提としている。
この端末では既存ドライバーで動作したが、上記ドライバー番号と26.1.1の対応は未確認。
別端末への導入時は[AMD公式7.2手順](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installryz/windows/install-pytorch.html)と
[対応表](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/windows/windows_compatibility.html)を参照する。

## 起動と停止（この端末で確認済みの方法）

PowerShellでリポジトリ直下に移動して起動する。

```powershell
Set-Location C:\temp\RealtimeDepth
.\start_windows.ps1
```

停止する場合:

```powershell
.\stop_windows.ps1
```

どちらも実機で動作確認済み。実行ポリシーについては後述する。

専用Pythonを直接実行する方法も利用できる。
仮想環境のactivateは不要。

```powershell
Set-Location C:\temp\RealtimeDepth
$env:CONFIG_PATH = Join-Path $PWD 'config.windows.yaml'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv-rocm72-windows\Scripts\python.exe app.py
```

起動後にブラウザで `http://127.0.0.1:8000/` を開く。
`/stats` はFPSとカメラ表示名、`/stream` はMJPEGを返す。
Python直接起動の場合、停止は起動した端末で **Ctrl+C**。
この方法でもプロセス終了とHTTP応答停止を確認済み。

Python直接起動は `start_windows.ps1` のPID管理対象にはならない。
`stop_windows.ps1` は直接起動したアプリの停止には使用しない。

GPU診断と単体推論を個別に行う場合:

```powershell
$env:CONFIG_PATH = Join-Path $PWD 'config.windows.yaml'
.\.venv-rocm72-windows\Scripts\python.exe check_windows.py
.\.venv-rocm72-windows\Scripts\python.exe test_inference.py
```

`check_windows.py` はHIP、GPU利用可否、`gfx1151`、GPU上の簡単な計算を確認する。
診断・アプリの実機検証はサンドボックス外で実施した。
サンドボックス内ではカメラopenが失敗したため、実機未接続とは区別する。

## 初回セットアップ

以下は再構築用の手順。この端末では環境・ソース・重みの準備は完了している。
Python 3.12 x64とGitを用意し、リポジトリ直下で実行する。

### 専用環境と依存パッケージ

この端末に配置済みのPythonを使う場合:

```powershell
.\.python\python.exe -m venv .venv-rocm72-windows
```

Pythonランチャーがある別端末では、代わりに `py -3.12 -m venv .venv-rocm72-windows` を使える。
既に専用環境がある場合、作成は不要。

```powershell
.\.venv-rocm72-windows\Scripts\python.exe -m pip install --upgrade pip
.\.venv-rocm72-windows\Scripts\python.exe -m pip install -r requirements-rocm72-windows.txt
.\.venv-rocm72-windows\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-rocm72-windows\Scripts\python.exe -m pip check
```

`requirements-rocm72-windows.txt` はAMD公式7.2配布URLを固定している。
共通依存は従来の `requirements.txt` を利用する。既存Linux用requirementsは変更していない。

### モデルのソースと重み

未取得の場合のみ実行する。Windowsでは直接cloneし、symlink権限に依存しない。

```powershell
git clone https://github.com/DepthAnything/Depth-Anything-V2.git Depth-Anything-V2
New-Item -ItemType Directory -Force Depth-Anything-V2/checkpoints
Invoke-WebRequest -Uri 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth' -OutFile 'Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth'
```

今回検証したソースcommit:
`a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`

Small重みのSHA256:
`715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`

上記cloneはその時点の既定ブランチを取得する。今回の結果を再現する場合はソースcommitも揃える。
別encoderを利用する場合は対応する重みと設定へ変更する。

## Windows設定と実装

`config.windows.yaml` はモデル・サーバー・runtimeを含む完全なWindows用設定。
`CONFIG_PATH` で選択する。指定しない場合はLinux用 `config.yaml` が読み込まれる。
モデルの相対パスは設定ファイルのディレクトリを基準に解決する。

カメラ設定の既定値:

```yaml
camera:
  devices:
    - name: USB Camera
      device: 0
  defaults:
    backend: MSMF  # MSMF または DSHOW
    width: 640
    height: 480
    fps: 30
```

- Windowsのdeviceは0以上の整数index。backendはdefaultsまたは各deviceで指定できる。
- `name` は表示名で、実機を識別するIDではない。indexとカメラの対応は抜き差しで変わり得る。
- サーバーはローカルアクセス用 `127.0.0.1`。LANアクセスが必要ならhostを調整する。
- `app.py` と `test_inference.py` は設定をUTF-8で読む。
- `camera.py` にカメラ設定・open・探索を抽出済み。Windowsはopenと試し読みで接続を判定する。
- LinuxはV4L2、整数から `/dev/video{N}` への変換、by-idを含むパス存在確認、旧 `camera.device` 形式を維持する。
- 探索時とread失敗時の2か所でOS別の存在確認を使う。登録順の優先順位を維持する。
- Windowsではreadが連続10回失敗するとreleaseし、1秒間隔で再探索する。
  open/read自体がハングした場合の復帰は保証しておらず、実機評価が必要。
- `runtime.py` はOS別の診断を表示し、`HSA_OVERRIDE_GFX_VERSION` を除去する従来方針を維持する。
- `app.py` はWindows向けのソース・重み取得案内を追加済み。

## PowerShellスクリプト（start / stop動作確認済み）

| ファイル | 内容 |
| --- | --- |
| `setup_windows.ps1` | 7.2専用環境、依存導入、pip check、GPU確認、モデル存在確認 |
| `start_windows.ps1` | 専用PythonとWindows設定で非表示起動、ログ保存、`/stats`による準備完了確認 |
| `stop_windows.ps1` | PID・開始時刻・実行ファイルを照合して対象プロセスを強制停止 |

スクリプト実行が許可された環境向けのコマンド:

```powershell
.\setup_windows.ps1 -DownloadModel
.\start_windows.ps1
.\stop_windows.ps1
```

setupの `-Python` で環境作成用Pythonを指定できる。未指定なら `.python`、次に `py -3.12` を使う。
`-DownloadModel` は未取得のSmallモデルをダウンロードし、既存モデルは保持する。
startの `-Config` で設定ファイル、`-TimeoutSeconds` で準備完了待ち時間（既定180秒）を指定できる。
カメラなしでも `/stats` が返れば起動成功として扱う設計。
ログとPID情報は `.windows-state/stdout.log`、`stderr.log`、`server.json` に保存する。

当初は全scopeの実行ポリシーがUndefinedで、`.ps1` の実行が拒否された。
その後、ユーザーが次のコマンドを実行し、CurrentUserをRemoteSignedに変更した。

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

変更後、ユーザーが `start_windows.ps1` と `stop_windows.ps1` の成功を確認した。
さらに通常のユーザー環境でカメラ未接続のstartを実行し、
`Ready: http://127.0.0.1:8000/` と `{"camera":null,"fps":0.0}` を確認した。
エージェントのサンドボックス内では実行が引き続き拒否されたため、サンドボックス外で実施した。
3本の構文検証は成功済み。`setup_windows.ps1` 全体の実行は未検証で、環境構築は個別コマンドで確認済み。

## 検証結果（2026-10-01）

| 検証項目 | 結果 |
| --- | --- |
| 7.2環境の `pip check` | 成功 |
| HIP・GPU・gfx1151認識、GPU計算 | 成功 |
| Small / 518 / fp16 / eager / 30回推論 | 平均12.6 ms、79.4 FPS |
| 推論出力 | 518×518、NaN/Infなし、値域約0.103〜4.578 |
| カメラ接続中のアプリ | `/stats`で約36〜36.4 FPS |
| `/` | HTTP 200、ユーザーによるブラウザ画面確認済み |
| `/stats` | カメラ表示名とFPSを返す |
| `/stream` | MJPEGヘッダーとJPEGデータ取得成功 |
| アプリ停止 | 割り込み後のプロセス終了とポート8000の応答停止を確認 |
| PowerShell start / stop | ユーザーが両方の動作を確認（終了コード0、stopは `Server stopped.`） |
| カメラ未接続でのstart | 起動成功、`/stats` は `camera: null`、`fps: 0.0` |
| カメラ未接続時の配信 | `/` はHTTP 200、`/stream` はプレースホルダーのMJPEGヘッダーとJPEGデータを返す |
| 未接続起動後のカメラ接続 | 起動中に接続し、ユーザーが映像復帰を確認（既定MSMF） |
| 自動テスト | 6件成功 |
| Python・PowerShell構文、差分チェック | 成功 |

推論単体の79.4 FPSはカメラ取得・描画・配信を含まない。約36 FPSとは測定範囲が異なる。
今回の推論入力はカメラ画像で、Ubuntuとの固定入力比較ではない。

カメラindex 0の単体テスト（各30フレーム、OpenCV 4.14.0.94）:

| backend | 解像度 | 報告FPS | 測定FPS | 結果 |
| --- | --- | --- | --- | --- |
| MSMF | 640×480 | 30 | 約32.0 | 30/30フレーム成功 |
| DSHOW | 640×480 | 約30 | 約25.0 | 30/30フレーム成功 |

短時間測定であり、安定FPSの評価ではない。PnPには `AMD Camera Device` と
`USB 2.0 Camera` が存在するが、indexと実機IDの対応は未確認。

再検証コマンド（カメラの単体テストはアプリ停止中に実行）:

```powershell
.\.venv-rocm72-windows\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-rocm72-windows\Scripts\python.exe check_windows.py --camera MSMF
.\.venv-rocm72-windows\Scripts\python.exe check_windows.py --camera DSHOW
```

自動テストの対象はLinux旧設定・デバイスパス、Windows backend検証・探索優先順、
open失敗時release、連続10回失敗からの再探索、Flask routeとMJPEGプレースホルダ。
HTTPテストはモデル初期化を省いたtest clientによる検証であり、実機検証とは別に記録する。

## ROCm 10から7.2へ変更した経緯

当初は `.venv-rocm10-windows` にPyTorch `2.13.0+rocm10.0.0`、
torchvision `0.28.0+rocm10.0.0`、ROCm `10.0.0`とgfx1151用パッケージを導入した。
依存関係チェックは成功したが、`import torch` がWinError 4551で失敗した。

Code Integrityイベント3033/3077/3118から、Smart App Controlが次のDLLを拒否したことを確認した。

```text
.venv-rocm10-windows/Lib/site-packages/_rocm_sdk_libraries/bin/hipfft.dll
```

Authenticode状態は `NotSigned`。サンドボックス外でも同じ結果だった。
ユーザー指定で7.2専用環境を作成し、setup/startの既定を切り替えた。
**7.2の `hipfft.dll` も `NotSigned` だが、この端末では読み込みとGPU推論が成功した。**
署名済みになったため解決した、とは判断しない。Smart App Control設定、配布DLL、ドライバーは変更していない。
その後のPowerShell実行許可（CurrentUserのRemoteSigned）は別の設定変更である。

旧環境と `requirements-rocm10-windows.txt` は保持している。
Windowsの通常セットアップでは `requirements-rocm72-windows.txt` を使用する。

## Linux互換性と残作業

Linuxの `config.yaml`、`requirements.txt`、`requirements-rocm10.txt`、
`setup_rocm10.sh`、`start_all.sh`、`stop_all.sh` は内容・名前を維持している。
OS分岐のモックテストは成功したが、Linux実機での「回帰なし」は未判定。

- [x] Windows 7.2専用環境、モデル、依存導入とpip check
- [x] UTF-8設定読み込み、Windows設定、OS別カメラ処理と診断
- [x] gfx1151認識、GPU計算、単体推論の形状・有限値検証
- [x] MSMF / DSHOWの映像取得
- [x] Flask / MJPEG配信、ユーザーによる表示確認、直接起動したアプリの停止
- [x] setup/start/stopスクリプト作成と構文検証
- [x] PowerShell start / stopの実機動作確認（ユーザー確認）
- [x] カメラ未接続でのアプリ起動・プレースホルダー配信
- [x] 未接続起動後にカメラを接続して映像復帰（ユーザー確認、既定MSMF）
- [ ] setup_windows.ps1全体の実行検証
- [ ] 配信中のカメラ切断から再接続までの一連の試験
- [ ] カメラ未接続・切断中の停止と再起動
- [ ] 複数カメラの実機識別・優先順位、backendごとのopen/readハング検証
- [ ] 長時間の性能・安定性評価、任意のtorch.compile検証
- [ ] Linux実機の変更前後比較とWindows / Ubuntuの同条件比較

Linux回帰では既存setup/start/stop、整数・デバイスパス・by-id・旧設定形式、未接続起動、
抜き差し復帰、複数カメラ優先順位、3つのHTTP endpoint、停止・再起動、推論品質と性能を確認する。
Linux内の変更前後比較とWindows / Linux間比較は分けて記録する。

比較には同じ重み・入力サイズ・精度と、設定ファイルのディレクトリに配置した同一の `test.jpg` を使う。
`test_inference.py` は画像がなければカメラ、さらに失敗すればseed固定の合成画像を使うため、
比較入力が意図した画像になっていることを確認する。OS間で速度や出力の完全一致は要求しない。

## 参考情報

- [AMD公式 Windows / Ryzen PyTorch 7.2導入手順](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installryz/windows/install-pytorch.html)
- [AMD公式 Windows / Ryzen 7.2対応表](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/windows/windows_compatibility.html)
- [AMD HIP SDK for Windows（システム用インストーラー）](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
- [ROCm 10導入資料（旧環境の参考）](https://rocm.docs.amd.com/en/docs-10.0.0/install/rocm.html)
- [Microsoft Smart App Control FAQ](https://support.microsoft.com/en-us/windows/security/threat-malware-protection/smart-app-control-frequently-asked-questions)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
