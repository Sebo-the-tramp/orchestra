import json
import time
from typing import Any

from orchestra.config import load_config


def emit(event: str, **fields: Any) -> None:
    config = load_config()
    config.logs.mkdir(parents=True, exist_ok=True)
    payload = {"time_ns": time.time_ns(), "event": event} | fields
    path = config.logs / "events.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, default=str) + "\n")


def tail_events(limit: int = 20) -> list[dict[str, Any]]:
    path = load_config().logs / "events.jsonl"
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(line) for line in lines]
