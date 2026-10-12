# Chrona Showcase — Deploy & Add-a-Page Guide

This repo hosts the **Chrona** showcase site. It's published via **Azure Static Web Apps
(ASWA)** and gated behind **Microsoft sign-in** (built-in Entra ID provider) so it's
internal-only.

**Live site:** https://agreeable-bush-035f3c510.4.azurestaticapps.net/

## Access model (why sign-in, not blob/IP)
The site originally ran on Azure Blob static hosting and was public. Chen wanted it
internal-only ("only VPN and Microsoft internal"). Attempts to reach that were constrained:

- **IP allowlist** fails — there's no published CIDR for "Microsoft corpnet/VPN egress,"
  so you can't cleanly allow all Microsoft employees.
- **Custom single-tenant Entra SSO** is blocked in the Microsoft corporate tenant
  (`72f988bf`) by the **ServiceTree** requirement — every new app registration must carry a
  `ServiceManagementReference`, which normal user credentials can't satisfy.
- **ASWA built-in Entra provider** is the working answer: ASWA manages the app
  registration itself, needs **zero app registration / secret**, works on the **Free** SKU,
  and gives a real "Sign in with Microsoft" page. Tradeoff: it accepts **any** Microsoft /
  Azure AD account (it can't be pinned to @microsoft.com only without a custom provider).

## The core workflow: build + deploy to ASWA
ASWA serves whatever is deployed to it (no GitHub source control — we deploy manually with
the SWA CLI + a deployment token).

```bash
# 1. Need the SWA CLI (once):
npm install -g @azure/static-web-apps-cli

# 2. Regenerate the build
python build_static.py

# 3. Get a deployment token (any contributor can fetch it, or use the one from the portal)
TOKEN=$(az staticwebapp secrets list -n chrona-showcase-internal -g CSE-Showcase-RG \
        --query "properties.apiKey" -o tsv)

# 4. Deploy dist/ (includes the auth/routing config from staticwebapp.config.json)
cp staticwebapp.config.json dist/
swa deploy ./dist --deployment-token "$TOKEN" --env production
```

To add or edit content, edit `db.py` (a `Page` + `Section`s: `slug` = URL path,
`nav_label` = nav bar), rerun `python build_static.py`, then `swa deploy`.

## Auth / access control in `staticwebapp.config.json`
- Route `/*` requires the `authenticated` role — anonymous users are blocked.
- A `401` override redirects them to `/.auth/login/aad` (the Microsoft sign-in page).
- Everything is served to anyone who signs in with a Microsoft account.

## How the ASWA was created (reference)
```bash
az staticwebapp create -n chrona-showcase-internal -g CSE-Showcase-RG -l centralus --sku Free
```
Built-in Entra auth needs no extra config. If you ever need to restrict to @microsoft.com
only, that requires a custom `azureActiveDirectory` provider (Standard SKU + an app
registration released in the corp tenant by an internal owner).

## Previous hosting: Azure Blob static hosting (now redirect-only)
The site used to live at
`https://privacypshowcase12114.z19.web.core.windows.net/` via the `$web` container of
storage account `privacypshowcase12114` (`CSE-Showcase-RG`). All content was migrated off
the blob to the ASWA, and the old URL is now a **redirect-only** endpoint: its only blob
(`index.html`) immediately redirects visitors to this ASWA sign-in site (so old bookmarks
still get through). The old content files (AgenticGem catalog, dashboards, prompts,
schemas) were **deleted** from `$web` and now return 404 — do **not** upload content back to
the blob; it is no longer the host. The storage account stays in `publicNetworkAccess=Enabled`
/ default-allow solely so the redirect serves; only `$web`'s `index.html` is public.

## The deployed showcase source is version-controlled
The live site content (all pages, prompts, JSON schemas, agentic catalog) ships from this
repo's **`showcase/`** folder — a copy of what's deployed to the ASWA. To add or edit a
live page, edit the file in `showcase/`, then `swa deploy ./showcase --deployment-token
"$TOKEN" --env production` (the folder already includes `staticwebapp.config.json`). The
`dist/` build from `build_static.py` serves the chrona app pages; `showcase/` holds the hub
+ catalog + dashboard pages that make up the public site.

## Constraints / gotchas
- **Later change:** `clientSecretCertificateSettingName` is NOT a valid field in
  `staticwebapp.config.json` (SWA CLI rejects it) — only `openIdIssuer`,
  `clientIdSettingName`, `clientSecretSettingName` are valid for a custom AAD provider.
- `*.json` is gitignored by this repo's `.gitignore`, so `staticwebapp.config.json` stays
  out of git. If you ever want it tracked, `git add -f` it.
- **No server-side logic.** Both blob and ASWA-only setups have no backend; anything
  dynamic runs in-browser (the redaction demo runs client-side in `demo.js`).
