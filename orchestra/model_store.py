import shutil
from fnmatch import fnmatch
from pathlib import Path

from orchestra.config import OrchestraConfig
from orchestra.registry import ModelSpec

COMPLETE = ".orchestra-complete"


def model_path(config: OrchestraConfig, model: ModelSpec) -> Path:
    if model.local_path is not None and not Path(model.local_path).suffix:
        return Path(model.local_path).expanduser()
    safe_name = model.name.replace("/", "__")
    return config.model_cache / safe_name


def model_file_path(config: OrchestraConfig, model: ModelSpec) -> Path:
    if model.local_path is not None:
        return Path(model.local_path).expanduser()
    path = model_path(config, model)
    if model.artifact_file is not None:
        return path / model.artifact_file
    return path


def marker_path(config: OrchestraConfig, model: ModelSpec) -> Path:
    return model_path(config, model) / COMPLETE


def model_status(config: OrchestraConfig, model: ModelSpec) -> str:
    if model.source == "builtin":
        return "downloaded"
    if model.local_path is not None:
        path = model_file_path(config, model)
        return "downloaded" if path.exists() else "missing"
    path = model_path(config, model)
    if marker_path(config, model).is_file():
        return "downloaded"
    if path.exists():
        return "partial"
    return "missing"


def download_plan(config: OrchestraConfig, model: ModelSpec) -> str:
    path = model_file_path(config, model)
    if model.source == "huggingface":
        if model.artifact_file:
            return f"huggingface hf_hub_download {model.artifact}:{model.artifact_file} -> {path}"
        return f"huggingface snapshot_download {model.artifact} -> {path}"
    if model.source == "local":
        return f"copy local artifact {model.artifact} -> {path}"
    if model.source == "builtin":
        return "builtin artifact"
    return f"manual download required: {model.artifact}"


def download_model(config: OrchestraConfig, model: ModelSpec, force: bool = False) -> Path:
    path = model_file_path(config, model)
    assert model.source != "builtin", f"Builtin model does not need download: {model.name}"
    if path.exists() and not force:
        assert model_status(config, model) != "downloaded", f"{model.name} is already downloaded"
    if path.exists() and force:
        answer = input(f"{path} exists. Override it? [y/N]: ").strip().lower()
        assert answer in {"y", "yes"}, f"Refusing to override {path}"
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    if model.source == "huggingface":
        if model.artifact_file:
            from huggingface_hub import hf_hub_download

            path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=model.artifact,
                filename=model.artifact_file,
                local_dir=path.parent,
            )
            assert path.is_file(), path
            if model.local_path is None:
                marker_path(config, model).write_text("complete\n")
        else:
            from huggingface_hub import snapshot_download

            path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=model.artifact,
                local_dir=path,
                local_dir_use_symlinks=False,
            )
            marker_path(config, model).write_text("complete\n")
    elif model.source == "local":
        source = Path(model.artifact).expanduser()
        assert source.exists(), source
        if source.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, path, dirs_exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            destination = path if path.suffix else path / source.name
            shutil.copy2(source, destination)
    else:
        raise AssertionError(f"Manual download required for {model.name}: {model.artifact}")
    return path


def remote_files(repo_id: str, pattern: str = "*") -> list[str]:
    from huggingface_hub import list_repo_files

    return sorted(path for path in list_repo_files(repo_id) if fnmatch(path, pattern))


def remove_model(config: OrchestraConfig, model: ModelSpec) -> None:
    path = model_file_path(config, model)
    assert path.exists(), path
    answer = input(f"Remove {path}? [y/N]: ").strip().lower()
    assert answer in {"y", "yes"}, f"Refusing to remove {path}"
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()
