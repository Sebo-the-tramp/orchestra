#!/usr/bin/env bash
set -euo pipefail

SESSION="o_session"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ORCHESTRA_PROCESS_MANAGER="${ORCHESTRA_PROCESS_MANAGER:-auto}"
BROKER_COMMAND="uv run orchestra broker start"
NVTOP_COMMAND="nvtop"
BTOP_COMMAND="btop"
GQUEUE_COMMAND="watch -n 0.1 gqueue"

if command -v gflowd >/dev/null 2>&1 && command -v gqueue >/dev/null 2>&1; then
    if ! gqueue >/dev/null 2>&1; then
        echo "Starting gflow..."
        gflowd up
    else
        echo "gflow is already running."
    fi
else
    if [ "$ORCHESTRA_PROCESS_MANAGER" = "gflow" ]; then
        echo "gflow is required but missing."
        exit 1
    fi
    GQUEUE_COMMAND="echo 'gflow missing; broker will use local process manager'; sleep infinity"
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
