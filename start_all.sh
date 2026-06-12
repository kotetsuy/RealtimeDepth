#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PID_FILE="$PWD/.depth_app.pid"
LOG_FILE="$PWD/depth_app.log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already running (pid $(cat "$PID_FILE")). use ./stop_all.sh to stop." >&2
  exit 1
fi
rm -f "$PID_FILE"

# shellcheck disable=SC1091
source .venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.5.1

PORT=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['server']['port'])")

: > "$LOG_FILE"
nohup python app.py >>"$LOG_FILE" 2>&1 &
APP_PID=$!
echo "$APP_PID" > "$PID_FILE"
echo "started (pid $APP_PID), log: $LOG_FILE"

echo "waiting for ready (cold start ~110s, cached ~3s)..."
for _ in $(seq 1 60); do
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "app exited unexpectedly, last log lines:" >&2
    tail -20 "$LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
  # /stats が 200 を返せばサーバ稼働 (MIGraphX コンパイルは Flask 起動前に完了している)。
  # カメラ未接続でもプレースホルダ配信で稼働するため fps>0 は条件にしない。
  resp=$(curl -fs --max-time 1 "http://127.0.0.1:${PORT}/stats" 2>/dev/null || true)
  if [[ -n "$resp" ]]; then
    cam=$(printf '%s' "$resp" | sed -n 's/.*"camera":\s*"\([^"]*\)".*/\1/p')
    LAN_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
    URL="http://localhost:${PORT}/"
    if [[ -n "$cam" ]]; then
      echo "ready (camera: ${cam}). open ${URL} or http://${LAN_IP:-<host-ip>}:${PORT}/"
    else
      echo "ready (no camera connected; serving placeholder). open ${URL} or http://${LAN_IP:-<host-ip>}:${PORT}/"
    fi

    if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v google-chrome >/dev/null 2>&1; then
      echo "launching Chrome..."
      nohup google-chrome --new-window "${URL}" >/dev/null 2>&1 &
      disown
    else
      echo "no display or chrome not found; open ${URL} manually."
    fi
    exit 0
  fi
  sleep 3
done

echo "compile did not complete within 180s; check $LOG_FILE" >&2
exit 1
