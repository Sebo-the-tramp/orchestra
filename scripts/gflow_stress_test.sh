#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ORCHESTRA_TEST_OUT_DIR:-/tmp/orchestra-gflow-test-$(date +%Y%m%d-%H%M%S)}"
API_HOST="${ORCHESTRA_TEST_API_HOST:-127.0.0.1}"
API_PORT="${ORCHESTRA_TEST_API_PORT:-8010}"
API_URL="${ORCHESTRA_TEST_API_URL:-http://$API_HOST:$API_PORT}"
TIMEOUT_S="${ORCHESTRA_TEST_TIMEOUT_S:-1200}"
TIMEOUT_MS="${ORCHESTRA_TEST_TIMEOUT_MS:-1200000}"
PROGRESS_SECONDS="${ORCHESTRA_TEST_PROGRESS_SECONDS:-10}"
START_STACK="${ORCHESTRA_TEST_START_STACK:-1}"
USE_TMUX="${ORCHESTRA_TEST_USE_TMUX:-1}"
RESET_TMUX="${ORCHESTRA_TEST_RESET_TMUX:-1}"
STOP_EXISTING="${ORCHESTRA_TEST_STOP_EXISTING:-$RESET_TMUX}"
TMUX_SESSION="${ORCHESTRA_TEST_TMUX_SESSION:-orchestra_gflow_test}"
PROCESS_MANAGER="${ORCHESTRA_PROCESS_MANAGER:-auto}"
QWEN_35B_MODEL="${ORCHESTRA_TEST_QWEN_35B_MODEL:-qwen3.6-35b}"
QWEN_9B_MODEL="${ORCHESTRA_TEST_QWEN_9B_MODEL:-qwen3.5-9b}"
WHISPER_MODEL="${ORCHESTRA_TEST_WHISPER_MODEL:-whisper}"
FASTER_WHISPER_MODEL="${ORCHESTRA_TEST_FASTER_WHISPER_MODEL:-faster-whisper}"
DINO_MODEL="${ORCHESTRA_TEST_DINO_MODEL:-facebook/dinov3-vits16-pretrain-lvd1689m}"
TRANSLATE_MODEL="${ORCHESTRA_TEST_TRANSLATE_MODEL:-translategemma}"
EMBED_MODEL="${ORCHESTRA_TEST_EMBED_MODEL:-bge-m3}"
AUDIO_PATH="${ORCHESTRA_TEST_AUDIO:-$OUT_DIR/test-tone.wav}"
IMAGE_PATH="${ORCHESTRA_TEST_IMAGE:-$OUT_DIR/test-image.png}"
BROKER_LOG="$OUT_DIR/broker.log"
API_LOG="$OUT_DIR/api.log"
GQUEUE_LOG="$OUT_DIR/gqueue-watch.log"
SUMMARY="$OUT_DIR/summary.tsv"

BROKER_PID=""
API_PID=""
GQUEUE_PID=""
STACK_IN_TMUX="0"

mkdir -p "$OUT_DIR"

cd "$ROOT"

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

have() {
    command -v "$1" >/dev/null 2>&1
}

elapsed() {
    local start="$1"
    python - "$start" <<'PY'
import sys
import time

print(f"{time.time() - float(sys.argv[1]):.3f}")
PY
}

record() {
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$SUMMARY"
}

run_timed() {
    local name="$1"
    shift
    local out="$OUT_DIR/$name.out"
    local err="$OUT_DIR/$name.err"
    local status_file="$OUT_DIR/$name.status"
    local start
    rm -f "$status_file"
    start="$(python - <<'PY'
import time

print(time.time())
PY
)"
    log "START $name"
    (
        if have timeout; then
            timeout "${TIMEOUT_S}s" "$@" >"$out" 2>"$err"
        else
            "$@" >"$out" 2>"$err"
        fi
        printf '%s\n' "$?" >"$status_file"
    ) &
    local command_pid="$!"
    while [ ! -f "$status_file" ]; do
        sleep "$PROGRESS_SECONDS"
        if [ ! -f "$status_file" ]; then
            log "WAIT  $name $(elapsed "$start")s; tmux: tmux attach -t $TMUX_SESSION"
        fi
    done
    wait "$command_pid" >/dev/null 2>&1
    local status
    status="$(cat "$status_file")"
    local seconds
    seconds="$(elapsed "$start")"
    if [ "$status" = "0" ]; then
        log "PASS  $name ${seconds}s"
        record "$name" "PASS" "$seconds" "$out"
    else
        log "FAIL  $name ${seconds}s status=$status"
        record "$name" "FAIL" "$seconds" "$err"
    fi
    return "$status"
}

wait_http() {
    local url="$1"
    local deadline="$((SECONDS + 60))"
    while [ "$SECONDS" -lt "$deadline" ]; do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

broker_bind_port() {
    python - <<'PY'
from orchestra.config import load_config

print(load_config().broker_bind_address.rsplit(":", 1)[-1])
PY
}

kill_port() {
    local port="$1"
    local label="$2"
    local pids=""
    if have lsof; then
        pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    fi
    if [ -z "$pids" ] && have fuser; then
        pids="$(fuser "$port/tcp" 2>/dev/null || true)"
    fi
    if [ -z "$pids" ]; then
        return
    fi
    log "Reset: chiudo $label sulla porta $port: $pids"
    kill $pids >/dev/null 2>&1 || true
    sleep 1
    kill -9 $pids >/dev/null 2>&1 || true
}

reset_existing_stack() {
    if [ "$STOP_EXISTING" != "1" ]; then
        return
    fi
    if have tmux && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        log "Reset: chiudo tmux session $TMUX_SESSION"
        tmux kill-session -t "$TMUX_SESSION"
    fi
    kill_port "$API_PORT" "API"
    kill_port "$(broker_bind_port)" "broker"
}

tmux_window() {
    local window="$1"
    local command="$2"
    tmux send-keys -t "$TMUX_SESSION:$window" "$command" C-m
}

gpu_monitor_command() {
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

log_tail_command() {
    local path="$1"
    echo "touch '$path'; tail -n +1 -f '$path'"
}

start_tmux_stack() {
    local broker_command
    local api_command
    local gqueue_cmd
    local gpu_cmd
    local broker_tail
    local api_tail

    broker_command="export ORCHESTRA_PROCESS_MANAGER='$PROCESS_MANAGER'; uv run orchestra broker start 2>&1 | tee '$BROKER_LOG'"
    api_command="uv run orchestra api start --host '$API_HOST' --port '$API_PORT' 2>&1 | tee '$API_LOG'"
    gqueue_cmd="$(gqueue_command)"
    gpu_cmd="$(gpu_monitor_command)"
    broker_tail="$(log_tail_command "$BROKER_LOG")"
    api_tail="$(log_tail_command "$API_LOG")"

    if [ "$RESET_TMUX" = "1" ] && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux kill-session -t "$TMUX_SESSION"
    fi
    if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux new-session -d -s "$TMUX_SESSION" -n "broker" -c "$ROOT"
        tmux new-window -t "$TMUX_SESSION" -n "api" -c "$ROOT"
        tmux new-window -t "$TMUX_SESSION" -n "gqueue" -c "$ROOT"
        tmux new-window -t "$TMUX_SESSION" -n "gpu" -c "$ROOT"
        tmux new-window -t "$TMUX_SESSION" -n "broker-log" -c "$ROOT"
        tmux new-window -t "$TMUX_SESSION" -n "api-log" -c "$ROOT"
        tmux_window "broker" "$broker_command"
        tmux_window "api" "$api_command"
        tmux_window "gqueue" "$gqueue_cmd"
        tmux_window "gpu" "$gpu_cmd"
        tmux_window "broker-log" "$broker_tail"
        tmux_window "api-log" "$api_tail"
        tmux select-window -t "$TMUX_SESSION:broker"
    fi
    STACK_IN_TMUX="1"
    log "Stack avviato in tmux: tmux attach -t $TMUX_SESSION"
}

start_background_stack() {
    export ORCHESTRA_PROCESS_MANAGER="$PROCESS_MANAGER"
    if curl -fsS "$API_URL/health" >/dev/null 2>&1; then
        log "API gia' attiva: $API_URL"
        log "Uso lo stack gia' avviato; il process manager dipende da quel broker."
        return
    fi
    log "Avvio broker in background con process manager=$ORCHESTRA_PROCESS_MANAGER"
    uv run orchestra broker start >"$BROKER_LOG" 2>&1 &
    BROKER_PID="$!"
    sleep 2
    log "Avvio API in background: $API_URL"
    uv run orchestra api start --host "$API_HOST" --port "$API_PORT" >"$API_LOG" 2>&1 &
    API_PID="$!"
    if ! wait_http "$API_URL/health"; then
        log "API non raggiungibile dopo 60s; i test produrranno errori utili nei log."
    fi
}

start_stack() {
    export ORCHESTRA_PROCESS_MANAGER="$PROCESS_MANAGER"
    reset_existing_stack
    if have gflowd && have gqueue; then
        uv run orchestra gflow up >"$OUT_DIR/gflow-up.out" 2>"$OUT_DIR/gflow-up.err"
    fi
    if curl -fsS "$API_URL/health" >/dev/null 2>&1; then
        if [ "$STOP_EXISTING" = "1" ]; then
            log "API ancora attiva dopo reset: $API_URL"
            log "Chiudi il processo manualmente o usa ORCHESTRA_TEST_API_PORT=8011."
            exit 1
        fi
        log "API gia' attiva: $API_URL"
        log "Uso lo stack gia' avviato; il process manager dipende da quel broker."
        return
    fi
    if [ "$USE_TMUX" = "1" ] && have tmux; then
        start_tmux_stack
        if ! wait_http "$API_URL/health"; then
            log "API non raggiungibile dopo 60s; controlla: tmux attach -t $TMUX_SESSION"
        fi
        return
    fi
    start_background_stack
}

start_watchers() {
    if have gqueue; then
        while true; do
            date
            gqueue
            sleep 1
        done >"$GQUEUE_LOG" 2>&1 &
        GQUEUE_PID="$!"
    fi
}

cleanup() {
    if [ -n "$GQUEUE_PID" ]; then
        kill "$GQUEUE_PID" >/dev/null 2>&1
    fi
    if [ "$STACK_IN_TMUX" = "1" ]; then
        return
    fi
    if [ "$START_STACK" = "1" ] && [ -n "$API_PID" ]; then
        kill "$API_PID" >/dev/null 2>&1
    fi
    if [ "$START_STACK" = "1" ] && [ -n "$BROKER_PID" ]; then
        kill "$BROKER_PID" >/dev/null 2>&1
    fi
}

make_assets() {
    python - "$AUDIO_PATH" "$IMAGE_PATH" <<'PY'
import math
import struct
import sys
import wave
from pathlib import Path

from PIL import Image, ImageDraw

audio_path = Path(sys.argv[1])
image_path = Path(sys.argv[2])
audio_path.parent.mkdir(parents=True, exist_ok=True)
image_path.parent.mkdir(parents=True, exist_ok=True)

if not audio_path.exists():
    rate = 16_000
    samples = []
    for i in range(rate * 3):
        value = 0.25 * math.sin(2 * math.pi * 440 * i / rate)
        samples.append(struct.pack("<h", int(value * 32767)))
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"".join(samples))

if not image_path.exists():
    image = Image.new("RGB", (384, 384), (18, 24, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 344, 344), outline=(240, 220, 90), width=8)
    draw.ellipse((112, 96, 272, 256), fill=(70, 150, 230))
    draw.text((96, 292), "ORCHESTRA", fill=(255, 255, 255))
    image.save(image_path)
PY
}

json_payload() {
    python - "$1" "$2" "$3" <<'PY'
import json
import sys

model, prompt, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(json.dumps({
    "model_name": model,
    "prompt": prompt,
    "config": {"max_tokens": max_tokens, "temperature": 0},
    "timeout_ms": 1_200_000,
}))
PY
}

run_generate() {
    local name="$1"
    local model="$2"
    local prompt="$3"
    local max_tokens="${4:-64}"
    local payload
    payload="$(json_payload "$model" "$prompt" "$max_tokens")"
    run_timed "$name" curl -fsS --max-time "$TIMEOUT_S" \
        "$API_URL/generate" \
        -H "content-type: application/json" \
        -d "$payload"
}

run_dino() {
    local name="$1"
    local out="$OUT_DIR/$name.out"
    local err="$OUT_DIR/$name.err"
    local start
    start="$(python - <<'PY'
import time

print(time.time())
PY
)"
    log "START $name"
    python - "$DINO_MODEL" "$IMAGE_PATH" "$TIMEOUT_MS" >"$out" 2>"$err" <<'PY'
import json
import sys
import uuid
from pathlib import Path

import zmq

from orchestra.config import load_config
from utils.image_io import image_bytes

model_name = sys.argv[1]
image_path = Path(sys.argv[2])
timeout_ms = int(sys.argv[3])
payload = {
    "type": "REQUEST",
    "request_id": str(uuid.uuid4()),
    "model_name": model_name,
    "num_images": 1,
    "args_per_model": {
        "image_size": 224,
        "device_map": "auto",
        "torch_dtype": "float16",
    },
}
socket = zmq.Context.instance().socket(zmq.REQ)
socket.connect(load_config().broker_address)
socket.send_multipart([json.dumps(payload).encode("utf-8"), image_bytes(image_path)])
assert socket.poll(timeout_ms), f"timeout after {timeout_ms} ms"
frames = socket.recv_multipart()
response = json.loads(frames[0].decode("utf-8"))
response["tensor_frames"] = len(frames) - 1
print(json.dumps(response, indent=2))
PY
    local status="$?"
    local seconds
    seconds="$(elapsed "$start")"
    if [ "$status" = "0" ]; then
        log "PASS  $name ${seconds}s"
        record "$name" "PASS" "$seconds" "$out"
    else
        log "FAIL  $name ${seconds}s status=$status"
        record "$name" "FAIL" "$seconds" "$err"
    fi
}

snapshot() {
    uv run orchestra config show >"$OUT_DIR/config.txt" 2>&1
    uv run orchestra gflow status >"$OUT_DIR/gflow-status.txt" 2>&1
    uv run orchestra models status >"$OUT_DIR/models-status.txt" 2>&1
    uv run orchestra env status >"$OUT_DIR/env-status.txt" 2>&1
    uv run orchestra jobs tail --limit 80 >"$OUT_DIR/jobs-before.txt" 2>&1
    uv run orchestra metrics tail --limit 80 >"$OUT_DIR/metrics-before.txt" 2>&1
}

final_snapshot() {
    uv run orchestra jobs tail --limit 120 >"$OUT_DIR/jobs-after.txt" 2>&1
    uv run orchestra metrics tail --limit 160 >"$OUT_DIR/metrics-after.txt" 2>&1
    if have gqueue; then
        gqueue >"$OUT_DIR/gqueue-final.txt" 2>&1
    fi
    if have nvidia-smi; then
        nvidia-smi >"$OUT_DIR/nvidia-smi-final.txt" 2>&1
    fi
}

main() {
    trap cleanup EXIT
    printf 'case\tstatus\tseconds\tfile\n' >"$SUMMARY"
    log "Output: $OUT_DIR"
    log "tmux session: $TMUX_SESSION"
    make_assets
    snapshot
    if [ "$START_STACK" = "1" ]; then
        start_stack
    else
        if ! wait_http "$API_URL/health"; then
            log "API non raggiungibile: $API_URL"
        fi
    fi
    start_watchers

    run_generate "01_qwen35_cold" "$QWEN_35B_MODEL" "Rispondi solo con: qwen35 ok" 32
    run_generate "02_qwen9_after_35_eviction" "$QWEN_9B_MODEL" "Rispondi solo con: qwen9 ok" 32
    run_generate "03_qwen35_reload" "$QWEN_35B_MODEL" "Rispondi solo con: qwen35 reload ok" 32

    run_generate "04_pressure_qwen35" "$QWEN_35B_MODEL" "Scrivi 5 righe numerate, concise." 128 &
    pid_a="$!"
    sleep 1
    run_generate "05_pressure_qwen9_queued" "$QWEN_9B_MODEL" "Rispondi solo con: qwen9 queued ok" 32 &
    pid_b="$!"
    wait "$pid_a"
    wait "$pid_b"

    run_generate "06_burst_qwen9_a" "$QWEN_9B_MODEL" "Rispondi solo con: burst a" 32 &
    pid_a="$!"
    run_generate "07_burst_qwen9_b" "$QWEN_9B_MODEL" "Rispondi solo con: burst b" 32 &
    pid_b="$!"
    run_generate "08_burst_qwen9_c" "$QWEN_9B_MODEL" "Rispondi solo con: burst c" 32 &
    pid_c="$!"
    wait "$pid_a"
    wait "$pid_b"
    wait "$pid_c"

    run_timed "09_translate_gemma" uv run orchestra request translate \
        "Ciao, questa e' una prova del sistema Orchestra." it en \
        --model-name "$TRANSLATE_MODEL" --timeout-ms "$TIMEOUT_MS"
    run_timed "10_embedding_bge" uv run orchestra request embed \
        "Orchestra deve caricare e scaricare modelli in base alla pressione di coda." \
        --model-name "$EMBED_MODEL" --timeout-ms "$TIMEOUT_MS"
    run_timed "11_whisper_transformers" uv run orchestra request transcribe \
        "$AUDIO_PATH" --model-name "$WHISPER_MODEL" --timeout-ms "$TIMEOUT_MS"
    run_timed "12_faster_whisper" uv run orchestra request transcribe \
        "$AUDIO_PATH" --model-name "$FASTER_WHISPER_MODEL" --timeout-ms "$TIMEOUT_MS"
    run_dino "13_dinov3_image_embedding"

    sleep 5
    final_snapshot
    log "Summary:"
    column -t -s $'\t' "$SUMMARY" || cat "$SUMMARY"
    log "Apri questi file:"
    log "  $SUMMARY"
    log "  $OUT_DIR/jobs-after.txt"
    log "  $OUT_DIR/metrics-after.txt"
    log "  $GQUEUE_LOG"
}

main
