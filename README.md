# Chrona Showcase

A public showcase site for sharing **demos, results, roadmap, and brainstorming** — hosted
via **Azure Blob static website hosting**.

**Live site:** https://privacypshowcase12114.z19.web.core.windows.net/

## Pages
| Page | Path |
|------|------|
| Overview | `/` |
| Benchmarks & Results | `/results.html` |
| Roadmap | `/roadmap.html` |
| Brainstorm | `/brainstorm.html` |
| Live Demo (redaction, runs in-browser) | `/demo.html` |

## This repo
This repository is now **webapp-only**. It builds the Chrona static site from the seed data:

- `db.py` — page/section seed content
- `build_static.py` — renders the DB into static HTML in `dist/`
- `templates/` + `static/` — the FastAPI templates & stylesheet (single-file pages are also
  emitted as static HTML by `build_static.py`)
- `SHOWCASE_DEPLOY.md` — deployment, how to publish, and how to host a new project's page

The historical research content (ensemble-redaction privacy pipeline) was removed from this
repo; it remains in the git history if ever needed.

## Quick publish
```bash
python build_static.py        # regenerate dist/
# then upload dist/* to the $web container (see SHOWCASE_DEPLOY.md)
```

See **`SHOWCASE_DEPLOY.md`** for the full deploy + add-a-page walkthrough.
