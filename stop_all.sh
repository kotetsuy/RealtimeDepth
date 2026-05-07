#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PID_FILE="$PWD/.depth_app.pid"

stop_pid() {
  local pid="$1"
  if ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  echo "stopping pid $pid..."
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.5
  done
  echo "force killing pid $pid"
  kill -9 "$pid" 2>/dev/null || true
  return 0
}

stopped=false
if [[ -f "$PID_FILE" ]]; then
  if stop_pid "$(cat "$PID_FILE")"; then
    stopped=true
  fi
  rm -f "$PID_FILE"
fi

# 念のため pid ファイル経由で見つからなかった残骸も拾う
while read -r pid; do
  [[ -z "$pid" ]] && continue
  if stop_pid "$pid"; then
    stopped=true
  fi
done < <(pgrep -f "[p]ython app.py" || true)

if $stopped; then
  echo "stopped."
else
  echo "no running app.py found."
fi
