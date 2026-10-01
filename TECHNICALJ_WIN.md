# RealtimeDepth — Windows 技術解説

[English](TECHNICAL_WIN.md) | [利用ガイド](READMEJ_WIN.md) | [検証記録](WINDOWS.md)

更新日: 2026-10-01。対象は `windows-support` の実装と、Windows 11 / Ryzen AI MAX+ 395での検証結果です。

## 1. 構成と責務

```text
OpenCV MSMF / DSHOW → BGRフレーム
  → CPU: RGB変換・518×518へresize
  → GPU: 正規化・Depth Anything V2 Small推論
  → CPU: 深度resize・色付け・左右結合・JPEG化
  → 最新JPEG共有 → Flask /stream → ブラウザ
```

| ファイル | 責務 |
| --- | --- |
| `app.py` | 設定、モデル初期化、推論、カメラworker、Flask route |
| `camera.py` | 新旧カメラ設定の正規化、OS別backend・存在確認、open・探索 |
| `runtime.py` | HIP / GPU診断、device / dtype選択 |
| `check_windows.py` | gfx1151計算確認、backend別カメラ試験、起動用port取得 |
| `test_inference.py` | 単一入力での推論ベンチマーク、形状・有限値検査 |
| `config.windows.yaml` | Windowsの完全な設定 |
| `requirements-rocm72-windows.txt` | AMD公式7.2 SDKと対応torch / torchvisionの固定URL |
| `setup_windows.ps1` / `start_windows.ps1` / `stop_windows.ps1` | 環境構築とプロセス管理 |
| `tests/test_camera.py` / `tests/test_http.py` | GPU・USBに依存しない回帰テスト |

モデルはPyTorchで直接実行します。ONNX Runtime / MIGraphXは現行の推論経路ではありません。Flaskはthreadedの開発サーバーで動き、debugとreloaderは無効です。

## 2. Windowsランタイム

専用環境は `.venv-rocm72-windows`、Pythonは3.12 x64です。ROCm関連パッケージは `7.2.0.dev0`、torchは `2.9.1+rocmsdk20260116`、torchvisionは `0.24.1+rocmsdk20260116`。`dev0`は[AMD公式7.2手順](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installryz/windows/install-pytorch.html)の配布名です。

`configure_runtime()` はprecisionを `fp16` / `fp32` に限定し、cuda指定ならHIPビルドとGPU利用可否を検査します。device indexを確定して `set_device()` を呼び、GPU名と `gcnArchName` を表示します。ROCmでもAPI名は `torch.cuda` です。gfx1151一致の強制検査と簡単なGPU演算は `check_windows.py` が担当し、共通runtime自体はgfx1151以外を拒否しません。

`HSA_OVERRIDE_GFX_VERSION` はtorch import前に環境から除去します。モデルを実アーキテクチャで実行する方針です。ドライバーとSDKは別物で、Python環境の導入はGPUドライバーの導入・更新を代行しません。

当初のROCm 10環境はSmart App Controlが `hipfft.dll` を拒否し、WinError 4551となりました。7.2では同DLLの署名表示も `NotSigned` ですが、この端末ではGPU推論に成功しています。署名が改善したとの結論ではありません。Smart App ControlやDLLは変更せず、旧環境を保持しています。後にユーザーが変更したPowerShellのRemoteSignedは別の実行ポリシーです。

## 3. モデル初期化と画像処理

設定はUTF-8で読みます。`CONFIG_PATH` 未指定時は `config.yaml`、startスクリプト使用時は `config.windows.yaml` が既定です。model.repo / checkpointの相対パスは設定ファイルのディレクトリ基準です。

Small (`vits`) はfeatures=64、out_channels=[48, 96, 192, 384]。checkpointをCPUへ `weights_only=True` で読み、state dictを設定後、選択device / dtypeへ移動して `eval()` にします。入力サイズは14の倍数である必要があります。

アプリ起動時にゼロ画像で3回warmupし、cudaなら同期します。この処理とモデル読み込みはモジュールのトップレベルで実行されるため、カメラがなくてもGPU・モデルの準備成功がサーバー起動の前提です。`compile: true` は `torch.compile(mode='reduce-overhead')` を使いますが、Windowsでの動作・性能は未検証です。

前処理:

1. OpenCVでBGR→RGB変換、INTER_CUBICで入力サイズの正方形へresize。
2. contiguousなuint8 HWCをGPUへ転送し、NCHWへ並べ替える。
3. dtype変換と255除算、mean=[0.485, 0.456, 0.406] / std=[0.229, 0.224, 0.225]で正規化。
4. `torch.inference_mode()` 下で推論し、最初のbatchの深度をfloat32のCPU NumPy配列へ戻す。

アスペクト比を保つletterboxではなく正方形への直接resizeです。出力は `(1, S, S)` で、アプリは `(S, S)` を使用します。

後処理はカメラ解像度へのINTER_LINEAR resize、フレーム内2〜98パーセンタイルによる正規化、0〜255への変換、INFERNO色付けです。範囲が1e-6未満ならゼロ画像にします。元映像と横に連結するため、640×480入力では通常1280×480のJPEGになります。未接続プレースホルダーは既定640×480です。

深度は近い側が大きくなる相対的な出力です。フレームごとの色正規化もあるため、色や数値をメートル距離として扱えません。画面内の距離レンジ表記は校正精度を保証しません。

## 4. カメラ状態と再探索

`CameraConfig` はdevice、width、height、fps、name、backendを保持します。Windowsではdeviceは0以上の整数、backendはMSMF / DSHOW（大文字小文字を正規化）。LinuxはV4L2です。各deviceの値がdefaultsより優先されます。旧 `camera.device` 形式も受け付けます。

Windowsの `device_present()` はファイルシステムを検査せず、実際のopen / readに判定を委ねます。Linuxでは整数を `/dev/video{N}` に変換し、パスの存在を確認します。この分岐は探索前とread失敗時の両方で使用します。

`open_camera()` は指定backendでVideoCaptureを作成し、解像度とFPSを要求します。isOpenedと1回の試し読みで検証し、失敗時はreleaseします。要求値が実機で採用された保証はなく、診断ツールで取得値を確認します。

```text
DISCONNECTED
  ├─ 1秒間隔で登録順に探索 → open / 試し読み成功 → STREAMING
  └─ 見つからない → プレースホルダー公開、0.2秒sleep
STREAMING
  ├─ read成功 → 失敗回数をリセット、推論・JPEG公開
  └─ read連続10回失敗 → release → DISCONNECTED
```

Linuxではパス消失も切断条件です。Windowsの途中のread失敗は0.01秒sleepして再試行します。優先順位は探索時に適用され、使用中のカメラより優先する機器が後から接続されても自動で切り替えません。

open / readが戻らなければタイマーや失敗回数は進まないため、これはハング対策を備えた別プロセス構成ではありません。カメラindexは固定IDではなく、nameは表示専用です。

確認済みなのはMSMF / DSHOWの単体取得と、既定MSMFでの未接続起動→接続後の映像復帰です。配信中に抜いて再接続する一連の挙動、複数カメラの実機優先順位、ハングは未検証です。

## 5. スレッド・HTTP・終了

daemonの `DepthWorker` がカメラ取得からJPEG化まで順番に処理します。共有するのは最新JPEG、FPS、カメラ表示名で、Lockで保護します。FIFOキューは持たず、遅いクライアントは中間フレームを取りこぼします。

`_publish()` は共有状態を更新してEventをsetします。各MJPEG generatorは最大1秒waitし、Eventをclearして最新JPEGを読みます。共有Eventはクライアント別の配信保証ではなく、複数クライアントの公平性・性能は未評価です。

| route | 応答 |
| --- | --- |
| `/` | HTML、img要素が `/stream` を参照 |
| `/stats` | `fps`（小数2桁）と `camera`。未接続は0 / null |
| `/stream` | `multipart/x-mixed-replace; boundary=frame`、各partにimage/jpegとContent-Length |

直接起動はFlask終了時のfinallyで `worker.stop()` を呼び、running=Falseと最大2秒joinを行います。カメラreleaseはworkerループ終了後です。readハングやプロセス強制終了ではこの後処理の完了を保証しません。

startスクリプトは専用Pythonを非表示で起動し、stdout / stderrを `.windows-state` へ保存します。開始前にloopbackのport競合を検査し、PID、開始時刻、Pythonパス、configをserver.jsonへ記録します。子プロセスの終了を監視しながら `/stats` を待ち、既定180秒でtimeoutします。カメラ接続はreadiness条件ではありません。

stopはPID・開始時刻・実行ファイルが一致するプロセスのみ `Stop-Process` で強制停止します。これはworkerの正常終了を要求するAPIではありません。ユーザーがstart / stopの成功を確認済みですが、切断中の停止・再起動は未検証です。setup全体は構文検証のみで、個別の環境構築コマンドが検証済みです。

## 6. 設定リファレンス

| 項目 | Windows既定 / 意味 |
| --- | --- |
| camera.devices | `name: USB Camera`, `device: 0`。登録順に探索 |
| camera.defaults | backend=MSMF、width=640、height=480、fps=30 |
| model.repo | `Depth-Anything-V2` |
| model.encoder | `vits`。コードにはvitb / vitl / vitgも構成あり、今回未検証 |
| model.checkpoint | `Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth` |
| model.input_size | 518、14の倍数 |
| server.host / port | `127.0.0.1` / 8000 |
| server.jpeg_quality | 80 |
| runtime.device / precision / compile | cuda / fp16 / false |

Windows設定とLinux設定は別ファイルです。Linuxのshellスクリプト、requirements、configは維持し、共通コードのOS分岐はモック検証済みです。Linux実機での回帰なしは未判定です。

## 7. 性能と再現性

| 測定 | 結果 / 範囲 |
| --- | --- |
| 単体推論 | 12.6 ms、79.4 FPS。前処理済みの同じ入力、3回warmup後30回、GPU同期あり |
| アプリ `/stats` | 約36〜36.4 FPS。取得開始から推論・色付け・左右結合までの逆数を平滑化 |
| カメラのみMSMF / DSHOW | 640×480、30枚成功。約32.0 / 25.0 FPSの短時間測定 |

`/stats` の計時終了は文字描画とJPEGエンコードより前です。係数0.9で過去値、0.1で新しい逆処理時間を反映します。JPEG化、HTTP転送、ブラウザ描画を含む厳密な全体FPSではありません。

`test_inference.py` は設定ディレクトリのtest.jpg、カメラindex 0（backend自動）、seed=0の合成画像の順に入力を選びます。アプリのカメラ設定はこの入力選択には使いません。画像を一度前処理してからモデルだけを計時し、最後の出力の形状・有限値を検査します。今回の入力はカメラ画像で、Linuxとの比較ではありません。

再現には同一test.jpg、重み、設定、精度、ソースcommitを使い、Linux内の変更前後比較とOS間比較を分けてください。

- モデルソースcommit: `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`
- 重みSHA256: `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`
- 実測HIP: `7.2.26024-f6f897bd3d`

## 8. テストと未検証項目

```powershell
.\.venv-rocm72-windows\Scripts\python.exe -m unittest discover -s tests -v
```

6件成功。OS分岐、旧設定・パス、探索優先順、open失敗時release、10回read失敗後の再探索、プレースホルダーHTTPを検証します。workerとrouteはapp.pyからASTで抽出してモデル初期化を避けるため、完全なアプリ起動テストではありません。

実機ではGPU計算、モデル推論、HTTP / MJPEG、ユーザーによる表示確認、start / stop、未接続起動後の接続を確認済みです。残る項目はsetup全体、配信中の抜き差し一連動作、切断中の終了・再起動、複数カメラとハング、長時間運転、torch.compile、Linux実機回帰です。
