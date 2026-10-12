# Chrona Showcase

A showcase site for sharing **demos, results, roadmap, and brainstorming** — hosted on
**Azure Static Web Apps**, gated behind **Microsoft sign-in** so it's internal-only.

**Live site:** https://agreeable-bush-035f3c510.4.azurestaticapps.net/

Visitors see a **"Sign in with Microsoft"** page and use any Microsoft / Azure AD account to
enter. This keeps the site internal without relying on fragile IP allowlists (Microsoft
corpnet/VPN egress has no clean CIDR to list).

## Pages
| Page | Path |
|------|------|
| Overview | `/` |
| Benchmarks & Results | `/results.html` |
| Roadmap | `/roadmap.html` |
| Brainstorm | `/brainstorm.html` |
| Live Demo (redaction, runs in-browser) | `/demo.html` |

## This repo
This repository builds the Chrona static site from the seed data:

- `db.py` — page/section seed content
- `build_static.py` — renders the DB into static HTML in `dist/`
- `templates/` + `static/` — the FastAPI templates & stylesheet (single-file pages are also
  emitted as static HTML by `build_static.py`)
- `staticwebapp.config.json` — ASWA routing + auth (all routes require `authenticated`,
  anonymous → `/.auth/login/aad`)
- `SHOWCASE_DEPLOY.md` — deployment, how to publish, and how to host a new project's page

The historical research content (ensemble-redaction privacy pipeline) was removed from this
repo; it remains in the git history if ever needed.

## Quick publish
```bash
python build_static.py        # regenerate dist/
swa deploy ./dist --deployment-token "<token>" --env production   # see SHOWCASE_DEPLOY.md
```

See **`SHOWCASE_DEPLOY.md`** for the full deploy + add-a-page walkthrough.
