#!/usr/bin/env bash
set -euo pipefail

SESSION="o_session"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BROKER_COMMAND="\"$ROOT\"/.venv/bin/python broker_core.py"
NVTOP_COMMAND="nvtop"
BTOP_COMMAND="btop"
GQUEUE_COMMAND="watch -n 0.1 gqueue"

if ! gqueue >/dev/null 2>&1; then
    echo "Starting gflow..."
    gflowd up
else
    echo "gflow is already running."
fi

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "orchestra" -c "$ROOT"
    tmux split-window -h -t "$SESSION:0.0" -c "$ROOT"
    tmux split-window -v -t "$SESSION:0.0" -c "$ROOT"
    tmux split-window -v -t "$SESSION:0.1" -c "$ROOT"
    tmux select-layout -t "$SESSION:0" tiled

    tmux send-keys -t "$SESSION:0.0" "$BROKER_COMMAND" C-m
    tmux send-keys -t "$SESSION:0.1" "$NVTOP_COMMAND" C-m
    tmux send-keys -t "$SESSION:0.2" "$GQUEUE_COMMAND" C-m
    tmux send-keys -t "$SESSION:0.3" "$BTOP_COMMAND" C-m
    tmux select-pane -t "$SESSION:0.0"
fi

if [ -n "${TMUX:-}" ]; then
    tmux switch-client -t "$SESSION"
else
    tmux attach-session -t "$SESSION"
fi
