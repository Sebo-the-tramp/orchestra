# Roadmap

## Near-Term Cleanup

| Area | Why It Matters |
| --- | --- |
| Schema and worker payload alignment | The docs and runtime should describe the same contract |
| Packaging cleanup in the root project | `uv add` currently fails on the repo layout, which blocks cleaner docs dependency management |
| Better examples under `recipes/` | The docs can then link to stable, runnable patterns instead of test scripts |
| Port and host centralization | Hardcoded broker addresses make deployment brittle |

## Project TODOs

These goals come directly from the current project notes:

- Extend connectivity beyond the local LAN.
- Support a hierarchy of brokers or a larger routing broker that understands all available resources.
- Evaluate a worker layout closer to `models/{lab}/{model_family}` everywhere and see what breaks.

## Longer-Term Docs Targets

- Auto-generate the model catalog from `models/*/config.yaml`.
- Document a stable client API once the request schemas are enforced end-to-end.
- Add deployment notes for multi-host brokers and remote workers.
