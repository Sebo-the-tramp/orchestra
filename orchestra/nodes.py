from dataclasses import dataclass
from typing import Any

import yaml

from orchestra.config import CONFIG_PATH, OrchestraConfig
from orchestra.hardware import HardwareReport
from orchestra.model_store import model_status
from orchestra.registry import Registry, runtime_for_model
from orchestra.runtime import compatible, env_status


@dataclass(frozen=True)
class NodeModelStatus:
    model_name: str
    engine: str
    runtime: str
    model_status: str
    env_status: str
    compatible: bool


@dataclass(frozen=True)
class NodeSpec:
    name: str
    address: str
    role: str
    labels: dict[str, str]


def load_node_specs() -> list[NodeSpec]:
    if not CONFIG_PATH.is_file():
        return []
    data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    nodes = data.get("nodes", [])
    return [
        NodeSpec(
            name=node["name"],
            address=node["address"],
            role=node.get("role", "broker"),
            labels=dict(node.get("labels", {})),
        )
        for node in nodes
    ]


def write_node_spec(spec: NodeSpec, force: bool = False) -> None:
    data: dict[str, Any] = {}
    if CONFIG_PATH.is_file():
        data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    nodes = list(data.get("nodes", []))
    exists = [node for node in nodes if node["name"] == spec.name]
    if exists and not force:
        answer = input(f"Node {spec.name} exists. Replace it? [y/N]: ").strip().lower()
        assert answer in {"y", "yes"}, f"Refusing to replace node {spec.name}"
    nodes = [node for node in nodes if node["name"] != spec.name]
    nodes.append(
        {
            "name": spec.name,
            "address": spec.address,
            "role": spec.role,
            "labels": spec.labels,
        }
    )
    data["nodes"] = nodes
    CONFIG_PATH.write_text(yaml.safe_dump(data, sort_keys=False))


def local_node_status(
    config: OrchestraConfig,
    registry: Registry,
    hardware: HardwareReport,
) -> list[NodeModelStatus]:
    statuses = []
    for model in registry.models.values():
        runtime = runtime_for_model(registry, model)
        statuses.append(
            NodeModelStatus(
                model_name=model.name,
                engine=model.default_engine,
                runtime=model.default_runtime,
                model_status=model_status(config, model),
                env_status=env_status(runtime),
                compatible=compatible(runtime, hardware),
            )
        )
    return statuses
