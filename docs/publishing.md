# Publishing

## Local Build

Preview the docs locally:

```bash
uv run --no-project --with mkdocs-material mkdocs serve
```

Create the static output:

```bash
uv run --no-project --with mkdocs-material mkdocs build --strict
```

The generated site lands in `site/`, which is ignored by git.

## GitHub Pages

This repo now includes a Pages workflow at `.github/workflows/docs.yml`.

It does three things:

1. Checks out the repository.
2. Builds the MkDocs site with `uv`.
3. Uploads `site/` and deploys it through GitHub Pages.

## One-Time GitHub Setup

For the repository at `https://github.com/Sebo-the-tramp/orchestra`:

1. Push the new workflow to `main`.
2. Open `Settings -> Pages`.
3. Set `Build and deployment -> Source` to `GitHub Actions`.
4. Let the `docs` workflow finish once on `main`.

With the current repo name, the published URL is expected to be:

```text
https://sebo-the-tramp.github.io/orchestra/
```

## Notes

- `site_url` in `mkdocs.yml` assumes the repository keeps the same owner and name.
- If you later add a custom domain, you can drop a `CNAME` file into `docs/`.
