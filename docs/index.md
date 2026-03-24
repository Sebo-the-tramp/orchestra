---
hide:
  - toc
---

<div class="hero">
  <div class="hero-copy">
    <p class="hero-kicker">Minimal Broker-Worker Runtime</p>
    <h1>Route model jobs without building a heavy control plane.</h1>
    <p>
      Orchestra keeps the control layer intentionally small: one ZeroMQ broker owns the queues,
      workers are started only when needed, and idle workers terminate on their own.
    </p>
    <div class="hero-actions">
      <a href="getting-started/" class="md-button md-button--primary">Start locally</a>
      <a href="architecture/" class="md-button">Read the architecture</a>
    </div>
  </div>
  <div class="hero-panel">
    <p class="hero-panel-label">Current shape</p>
    <ul class="hero-list">
      <li>Broker listens on <code>tcp://10.10.151.14:5556</code></li>
      <li>Workers live under <code>models/&lt;lab&gt;/&lt;family&gt;/worker.py</code></li>
      <li>Model registry comes from <code>models/*/config.yaml</code></li>
      <li>Current families: InternVL and SAM3</li>
    </ul>
  </div>
</div>

## Design Goals

<div class="feature-grid">
  <div class="feature-card">
    <h3>Keep the broker stupid</h3>
    <p>
      The broker only validates the model name, queues the job, spawns a worker if needed,
      and forwards the result back to the original client.
    </p>
  </div>
  <div class="feature-card">
    <h3>Isolate workers hard</h3>
    <p>
      Every worker folder is expected to own its own runtime, usually through a local
      <code>.venv</code>, so failures stay local to one model family.
    </p>
  </div>
  <div class="feature-card">
    <h3>Stay registry-first</h3>
    <p>
      New capability is declared through <code>config.yaml</code>, a schema file, and a
      single worker entrypoint rather than through a large orchestration framework.
    </p>
  </div>
</div>

## What Exists Today

| Area | Status |
| --- | --- |
| Broker | `broker_core.py` queues jobs per model, spawns missing workers, and forwards `SUCCESS` / `ERROR` replies |
| Worker transport | `utils/transport.py` connects workers through a `DEALER` socket identity of `model_name-worker_id` |
| Model families | `models/OpenGVLab/InternVL` and `models/facebook/sam3` are wired today |
| Example clients | Basic request scripts live in `tests/test_intern.py` and `tests/sam.py` |
| Docs publishing | GitHub Pages deployment is prepared through `.github/workflows/docs.yml` |

## Where To Go Next

- Start with [Getting Started](getting-started.md) for local preview and runtime commands.
- Read [Architecture](architecture.md) for the broker and worker lifecycle.
- Check [Publishing](publishing.md) for the GitHub Pages flow and one-time repo settings.
