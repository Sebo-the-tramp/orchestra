from typing import Any

import yaml

from orchestra.config import CONFIG_PATH

CONFIG_KEYS = {
    "paths.root",
    "paths.model_cache",
    "paths.engine_cache",
    "paths.env_cache",
    "paths.logs",
    "broker.address",
    "broker.bind_address",
    "broker.connect_address",
}


def config_data() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}


def set_config_value(key: str, value: str, force: bool = False) -> None:
    assert key in CONFIG_KEYS, f"Unknown config key: {key}"
    data = config_data()
    section, field = key.split(".", 1)
    data.setdefault(section, {})
    current = data[section].get(field)
    if current is not None and not force:
        answer = input(f"{key} is {current}. Replace it? [y/N]: ").strip().lower()
        assert answer in {"y", "yes"}, f"Refusing to replace {key}"
    data[section][field] = value
    CONFIG_PATH.write_text(yaml.safe_dump(data, sort_keys=False))
