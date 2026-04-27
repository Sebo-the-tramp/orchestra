import json
import time
from typing import Any

from orchestra.config import load_config


def append_job_event(event: str, request_id: str | None, **fields: Any) -> None:
    config = load_config()
    config.logs.mkdir(parents=True, exist_ok=True)
    payload = {"time_ns": time.time_ns(), "event": event, "request_id": request_id} | fields
    with (config.logs / "jobs.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, default=str) + "\n")


def tail_jobs(limit: int = 20) -> list[dict[str, Any]]:
    path = load_config().logs / "jobs.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[-limit:]]
