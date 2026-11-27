#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
repo="${DEPTH_PRO_2_REPO:-$PWD/vendor/depth-pro-2}"
[[ -f "$repo/pyproject.toml" ]] || { echo "Depth Pro 2 公式リポジトリがありません: $repo" >&2; exit 1; }
export UV_CACHE_DIR="$PWD/.cache/uv"
export TORCH_HOME="$PWD/.cache/torch"
uv venv --python 3.14 .venv-depthpro2
uv pip install --python .venv-depthpro2/bin/python --index-url https://stable.repo.amd.com/rocm/whl-next/ 'torch[device-gfx1151]==2.13.0+rocm10.2.0' 'torchvision[device-gfx1151]==0.28.0+rocm10.2.0'
uv pip install --python .venv-depthpro2/bin/python -e "$repo" opencv-python flask pyyaml --constraint depthpro2-constraints.txt
uv pip check --python .venv-depthpro2/bin/python
