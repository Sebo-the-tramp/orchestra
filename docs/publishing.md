# Publishing

## Local

Preview:

```bash
uv run --no-project --with mkdocs-material mkdocs serve
```

Build:

```bash
uv run --no-project --with mkdocs-material mkdocs build --strict
```

Output goes to `site/`.

## GitHub Pages

Workflow: `.github/workflows/docs.yml`

It:

1. Checks out the repo.
2. Builds MkDocs with `uv`.
3. Uploads `site/`.
4. Deploys through GitHub Pages.

## Repo Setup

For `https://github.com/Sebo-the-tramp/orchestra`:

1. Push the workflow to `main`.
2. Open `Settings -> Pages`.
3. Set `Build and deployment -> Source` to `GitHub Actions`.
4. Wait for the `docs` workflow on `main`.

Expected URL:

```text
https://sebo-the-tramp.github.io/orchestra/
```

## Notes

| Setting | Notes |
| --- | --- |
| `site_url` | Assumes owner and repo stay the same |
| Custom domain | Add `docs/CNAME` later |
