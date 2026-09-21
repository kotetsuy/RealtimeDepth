# RealtimeDepth — Ubuntu 26.04 動作検証 進捗

## 目的
RealtimeDepth (Depth Anything V2 Small / gfx1151) を Ubuntu 26.04 / ROCm 7.14 環境で GPU 動作させる。

**現状: 達成済み。PyTorch (ROCm) 直接推論で GPU 動作、実測 26 FPS(カメラ 30fps 上限に律速)。**

---

## 【方針転換】2026-07-26: MIGraphX ソースビルド → PyTorch (ROCm) 直接推論へ

ONNX/MIGraphX 経路を放棄し、Depth Anything V2 を PyTorch ネイティブで直接推論する方針に変更。

### 転換理由
1. **DAv2 はもともと純 PyTorch モデル**(DINOv2 + DPT ヘッド)。ONNX export という中間層自体が不要で、過去に踏んだ Resize op 非互換等の罠も消える。
2. **ROCm 本体バージョンからの構造的デカップリング**。PyTorch の ROCm wheel は自前の ROCm ランタイムを同梱しており、システム `/opt/rocm` のバージョン(7.14)に依存しない。カーネルドライバ(KFD/amdgpu, 26.04 は in-tree で新しい)さえ動けば良い。「7.14 に gfx1151 用 MIGraphX が存在しない」問題は onnxruntime-migraphx がシステム ROCm に密結合していたことが根本原因。
3. **性能見込み**: CPU(ORT)で約 10 FPS 実績あり → GPU fp16 (vits/518px) ならリアルタイム域は固い。→ **実測で確認済み(推論単体 77 FPS)**
4. ORT の exec-stack ELF パッチ(下記・修正済み項)も不要になる。

### 実装結果(2026-07-26 完了)

- [x] 新規 venv 作成 — `.venv-torch`(**Python 3.14**、system `/usr/bin/python3.14` を使用)
      ※ 旧 `.venv` は Python 3.10 で gfx1151 wheel に cp310 が無いため再利用不可。
        当初はフォールバックとして残していたが、移行完了後に削除済み(下記「残タスク」)。
- [x] PyTorch ROCm wheel + torchvision をインストール
- [x] GPU 認識確認 — `torch.cuda.is_available() == True` / `gfx1151` / fp16 matmul OK
- [x] RealtimeDepth の推論部を差し替え(前処理・MJPEG/Flask/スレッド設計は流用)
- [x] fp16 eager で FPS 計測 → **十分なので `torch.compile` は不要**(config で切替可能なまま残置)

### wheel の入手元(重要)

**`pytorch.org` の ROCm index は使わないこと。** マルチアーキ kpack のため gfx1151 では
実行時に `hipErrorInvalidImage` / `kpack_load_code_object failed with error: 13` になる
(whisperX で既に踏んだのと同じ罠)。gfx1151 専用 index を使う:

```bash
uv venv --python 3.14 .venv-torch

VIRTUAL_ENV=$PWD/.venv-torch uv pip install \
  --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match --prerelease allow \
  torch==2.9.1+rocm7.13.0 torchvision==0.24.0+rocm7.13.0

VIRTUAL_ENV=$PWD/.venv-torch uv pip install flask opencv-python pyyaml
```

- **torch と torchvision のバージョン組は厳密に合わせる**。index には torchvision
  0.24.0 / 0.25.0 / 0.26.0 が並んでいるが、torch 2.9.1 に対応するのは **0.24.0** のみ。
  0.26.0 を入れると import 時に `RuntimeError: operator torchvision::nms does not exist` で落ちる。
  (cp314 の対応表: torch 2.9.1 ↔ tv 0.24.0 / 2.10.0 ↔ 0.25.0 / 2.11.0 ↔ 0.26.0)
- torchvision は必須。`depth_anything_v2/dpt.py` が `from torchvision.transforms import Compose`
  を import している。
- `rocm-sdk-libraries-gfx1151==7.13.0` が依存として自動で入る = システム `/opt/rocm` (7.14) 非依存。

### モデルリポジトリとチェックポイントの配置

PyTorch 直接推論では `depth_anything_v2` パッケージを import するため、公式リポジトリ本体が
必要になる (ONNX 経路では `.onnx` 1 ファイルで完結していた)。リポジトリは `$HOME` 直下に
clone し、プロジェクトには symlink を張る:

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git ~/Depth-Anything-V2

mkdir -p ~/Depth-Anything-V2/checkpoints
curl -L -o ~/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

ln -s ~/Depth-Anything-V2 ~/RealtimeDepth/Depth-Anything-V2
```

- チェックポイントは vits (Small) で約 95MB。`config.yaml` の `model.encoder` を変える場合は
  対応する `.pth` (`vitb` = Base, `vitl` = Large) を同じ `checkpoints/` に置く。
- symlink と clone は **git の管理外** (`.gitignore` 済み)。マシンごとに手動で用意する。
  clone 実体をプロジェクト内に置かないのは、リポジトリの中に別リポジトリが入るのを避けるため。

### `HSA_OVERRIDE_GFX_VERSION` は設定しないこと

repo.amd.com の gfx1151 wheel は gfx1151 ネイティブビルドなので、override を付けると
逆に壊れる。`start_all.sh` では明示的に `unset` している。

### 実測性能 (Radeon 8060S / gfx1151 / vits / 518px / fp16)

| 項目 | 値 |
|---|---|
| 推論単体 | **77.3 FPS** (12.9 ms/frame) |
| アプリ全体 (取得+推論+カラーマップ+JPEG) | **26 FPS** ※カメラ 30fps 上限に律速 |
| 起動時間 (モデルロード + warmup 3 回) | 約 7 秒 |
| 参考: 旧 ONNX Runtime CPU 経路 | 約 10 FPS |

MIGraphX の ~110 秒の初回 JIT コンパイルと `.migraphx_cache` が不要になった。

### コード変更点

| ファイル | 変更内容 |
|---|---|
| `app.py` | `onnxruntime` を撤去し `torch` + `DepthAnythingV2` へ。`preprocess()` は GPU 上で正規化する形に変更(uint8 のまま転送し転送量 1/4)。`infer()` / `warmup()` を追加。カメラのステートマシン・MJPEG・Flask 部分は無変更 |
| `config.yaml` | `model.path` (onnx) → `model.repo` / `model.encoder` / `model.checkpoint` へ。`runtime` を `device` / `precision` / `compile` に置換(`compile_cache_dir` は廃止) |
| `start_all.sh` | venv を `.venv-torch` に変更、`HSA_OVERRIDE_GFX_VERSION` を `unset`、起動待ちメッセージを修正 |

チェックポイントは `~/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth`
(リポジトリ内の `Depth-Anything-V2` シンボリックリンク経由で参照)。

### 既知の無害な警告

起動ログに以下が出るが動作に影響なし:
- `xFormers not available` — DINOv2 が xformers を optional import しているだけ
- `warning: xnack 'Off' was requested for a processor that does not support it!`
- `MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable: ...gfx1151_20.HIP.fdb.txt`
  — MIOpen の事前チューニング DB が無いだけで、自動チューニング結果は初回実行時に生成される

### 代替案メモ(不採用)
- **TheRock の 7.14 世代 MIGraphX/ORT マルチアーキ wheel**: gfx1151 向け配布物に migraphx コンポーネントが実際には含まれていなかった報告あり(AUR rocm-gfx1151-bin, therock-dist 7.13)。まだ不安定。
- **ORT ROCm EP**: 1.23 で削除済み。MIGraphX EP 一本化の流れのため ONNX 経路に戻る動機は薄い。
- **最終フォールバック**: ncnn + Vulkan の DAv2 ポート(ROCm 完全非依存)。今回は不要と判断。

---

## 旧方針の記録(MIGraphX ソースビルド、凍結)

### 判明した問題
1. **exec-stack 問題(修正済み)**
   - `onnxruntime_pybind11_state.so` が実行可能スタック(GNU_STACK=RWE)を要求し、26.04 カーネルが import 時に拒否。
   - 対処: ELF の PF_X ビットを落として RWE→RW に変更(バックアップ `...so.execstack-bak`)。
   - 結果: CPU 実行は可能(約 10 FPS)。
   - ※ PyTorch 直方針ではこのパッチ自体が不要。パッチ済み `.so` は旧 `.venv` ごと削除済み。
     ORT を再導入する場合は同じ手当てが再び必要になる点だけ覚えておくこと。

2. **MIGraphX 非互換(→ 対応中断)**
   - AMD 配布の唯一の MIGraphX(7.2.1)はライブラリロードは通るが、GPU カーネルの実行時 JIT が 7.14 コンパイラで失敗(`__hmax` 曖昧エラー、LLVM23 の `[[clang::lifetimebound]]` -Werror、最終的に std::bad_alloc)。
   - gfx1151/7.14 専用の MIGraphX は AMD のどのチャネルにも存在しない(確認済み)。

### ソースビルド進捗(凍結時点)
- ソース: `~/AMDMIGraphX`(`rocm-7.14` タグ、ROCm と完全一致)
- 重量級を無効化: `MIGRAPHX_ENABLE_MLIR=OFF`(rocMLIR 回避), `MIGRAPHX_USE_COMPOSABLEKERNEL=OFF`, `MIGRAPHX_USE_EIGEN=OFF`
- 有効: rocBLAS / hipBLASLt / MIOpen(全て 7.14 に存在)
- ビルドツール: `~/AMDMIGraphX/.buildvenv`(Python 3.10。3.14 では cget が FancyURLopener 廃止で動かないため 3.10 に切替)
- requirements.txt から rocMLIR/CK/eigen を除外(`requirements.txt.orig` にバックアップ)
- [x] ソース取得(rocm-7.14 タグ)
- [x] ビルドツール準備(rbuild + cget、Python 3.10 venv)
- [x] 依存ライブラリのビルド完了(`rbuild prepare -d deps -s main`)
      - ログ末尾 `Successfully installed ROCm/rocm-cmake` = 最後の依存まで到達
      - `deps/lib` に libprotobuf.a / libprotobuf-lite.a / libsqlite3.a 生成済み
      - abseil は `libabsl_*.a` の名前で入っている(`grep abseil` では出ない、正常)
- [ ] ~~MIGraphX 本体の cmake 構成 → make -j~~ **← ここで凍結**
- [ ] ~~onnxruntime 経由の GPU 推論テスト~~

### 凍結時の注意点(万一 MIGraphX に戻る場合)
- 依存ビルドの `rbuild prepare` は正常終了済み。deps は残してあるので、ステップ2(本体 cmake 構成)から再開可能:
  ```
  -DCMAKE_PREFIX_PATH=~/AMDMIGraphX/deps
  -DMIGRAPHX_ENABLE_MLIR=OFF -DMIGRAPHX_USE_COMPOSABLEKERNEL=OFF
  -DMIGRAPHX_USE_EIGEN=OFF -DGPU_TARGETS=gfx1151
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++
  ```
  その後 `make -j$(nproc) && make install`(プレフィックス要指定)。
- MLIR/CK 無効でモデルがコンパイルできない場合、rocMLIR 有効化が必要になり追加 1 時間以上かかる可能性あり。
- ただし PyTorch 直経路が推論単体 77 FPS 出ているため、戻る動機は実質ない。

---

## 残タスク

なし。移行は完了。

- [x] `README.md` / `READMEJ.md` / `TECHNICAL.md` / `TECHNICALJ.md` を PyTorch 経路へ更新
      (`docs/*.svg` の図 6 枚と `test_inference.py` も併せて更新)
- [x] 旧経路の成果物を削除 (合計 約 2.2 GB)
      - `depth_anything_v2_vits_518.onnx` (95MB)
      - `.migraphx_cache/` (753MB、`.mxr` 1 ファイル)
      - 旧 `.venv` (1.4GB、Python 3.10 + onnxruntime-migraphx)
      - 削除前に現行コードからの参照がゼロであることを確認済み
      - exec-stack パッチを当てた `onnxruntime_pybind11_state.so` と
        そのバックアップも `.venv` ごと消滅 (PyTorch 経路では不要)


## ROCm 10 migration (2026-09-21)

- Added `requirements-rocm10.txt` with AMD's official `whl-next` index:
  `torch[device-gfx1151]==2.13.0+rocm10.0.0` and
  `torchvision[device-gfx1151]==0.28.0+rocm10.0.0`.
- `bash setup_rocm10.sh` installs into `.venv-rocm10`; the previous
  `.venv-torch` is retained. `start_all.sh` uses the new environment.
- Shared runtime validation reports the selected GPU, PyTorch and HIP versions,
  and removes the architecture override before importing PyTorch.
- Installed dependencies passed `uv pip check`.
- Radeon 8060S / gfx1151, vits, 518x518, fp16, eager: 30 measured forwards
  after 3 warmups averaged 12.5 ms (80.2 FPS). Output shape and finite values passed.
- Flask test client: `/` and `/stats` returned 200; `/stream` produced a JPEG.
  Jieli Camera was detected; sampled pipeline rate was 25.46 FPS.
- This AMD ROCm 10 wheel reports `torch.version.hip == 7.15.26333`;
  its package version is `2.13.0+rocm10.0.0` and SDK packages are `10.0.0`.
- Shell syntax, Python compilation and `git diff --check` passed.
- GPU validation ran outside the sandbox, which hides `/dev/kfd` and `/dev/dri`.
  The smoke test stopped its camera worker; no persistent server was started.
