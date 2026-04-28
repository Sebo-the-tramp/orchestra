#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${ORCHESTRA_MANUAL_TMUX_SESSION:-orchestra_manual}"
API_HOST="${ORCHESTRA_MANUAL_API_HOST:-127.0.0.1}"
API_PORT="${ORCHESTRA_MANUAL_API_PORT:-8011}"
PROCESS_MANAGER="${ORCHESTRA_PROCESS_MANAGER:-local}"
OUT_DIR="${ORCHESTRA_MANUAL_OUT_DIR:-/tmp/orchestra-manual-$(date +%Y%m%d-%H%M%S)}"
BROKER_LOG="$OUT_DIR/broker.log"
API_LOG="$OUT_DIR/api.log"

mkdir -p "$OUT_DIR"
cd "$ROOT"

have() {
    command -v "$1" >/dev/null 2>&1
}

kill_port() {
    local port="$1"
    local pids=""
    if have lsof; then
        pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    fi
    if [ -z "$pids" ] && have fuser; then
        pids="$(fuser "$port/tcp" 2>/dev/null || true)"
    fi
    if [ -n "$pids" ]; then
        kill $pids >/dev/null 2>&1 || true
        sleep 1
        kill -9 $pids >/dev/null 2>&1 || true
    fi
}

broker_bind_port() {
    python - <<'PY'
from orchestra.config import load_config

print(load_config().broker_bind_address.rsplit(":", 1)[-1])
PY
}

send_window() {
    local window="$1"
    local command="$2"
    tmux send-keys -t "$SESSION:$window" "$command" C-m
}

monitor_command() {
    if have nvtop; then
        echo "nvtop"
    elif have nvidia-smi; then
        echo "watch -n 1 nvidia-smi"
    else
        echo "echo 'no GPU monitor found'; sleep infinity"
    fi
}

gqueue_command() {
    if have gqueue; then
        echo "watch -n 1 gqueue"
    else
        echo "echo 'gqueue missing'; sleep infinity"
    fi
}

tmux kill-session -t "$SESSION" 2>/dev/null || true
kill_port "$API_PORT"
kill_port "$(broker_bind_port)"

tmux new-session -d -s "$SESSION" -n "broker" -c "$ROOT"
tmux new-window -t "$SESSION" -n "api" -c "$ROOT"
tmux new-window -t "$SESSION" -n "gpu" -c "$ROOT"
tmux new-window -t "$SESSION" -n "gqueue" -c "$ROOT"
tmux new-window -t "$SESSION" -n "broker-log" -c "$ROOT"
tmux new-window -t "$SESSION" -n "api-log" -c "$ROOT"

send_window "broker" \
    "export ORCHESTRA_PROCESS_MANAGER='$PROCESS_MANAGER'; uv run orchestra broker start 2>&1 | tee '$BROKER_LOG'"
send_window "api" \
    "uv run orchestra api start --host '$API_HOST' --port '$API_PORT' 2>&1 | tee '$API_LOG'"
send_window "gpu" "$(monitor_command)"
send_window "gqueue" "$(gqueue_command)"
send_window "broker-log" "touch '$BROKER_LOG'; tail -n +1 -f '$BROKER_LOG'"
send_window "api-log" "touch '$API_LOG'; tail -n +1 -f '$API_LOG'"
tmux select-window -t "$SESSION:broker"

cat <<EOF
Manual ORCHESTRA stack avviato.

Sessione:
  tmux attach -t $SESSION

Process manager:
  $PROCESS_MANAGER

API:
  http://$API_HOST:$API_PORT

Log:
  $BROKER_LOG
  $API_LOG

Per testare davvero il caricamento GPU, usa ORCHESTRA_PROCESS_MANAGER=local.
EOF
