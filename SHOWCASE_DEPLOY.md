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
ASWA serves whatever is deployed to it (no GitHub source control — we deploy manually).

> **CRITICAL — an SWA deploy REPLACES THE WHOLE SITE.** Stage ALL the files you want live
> into one directory and deploy ONCE. Any file not in the staged folder is **dropped** from
> the live site. Never deploy just a single edited page.

```bash
# 1. Regenerate the build (only for the chrona app pages in dist/)
python build_static.py

# 2. Get a deployment token (any contributor can fetch it, or use the portal one)
TOKEN=$(az staticwebapp secrets list -n chrona-showcase-internal -g CSE-Showcase-RG \
        --query "properties.apiKey" -o tsv)

# 3. Stage the whole site, then deploy ONCE.
#    The site root = the `showcase/` folder (hub + catalog + dashboards + config).
#    Copy dist/ pages in if they should also be live, then:
swa deploy ./showcase --deployment-token "$TOKEN" --env production
```

Two deploy tools work (in order of preference):
- **`swa` (global `@azure/static-web-apps-cli` 2.0.10)** — verified working:
  `npm install -g @azure/static-web-apps-cli`, then `swa deploy ./showcase --deployment-token "$TOKEN" --env production`.
- **`StaticSitesClient.exe`** (bundled with the SWA CLI; a plain `npx @azure/static-web-apps-cli`
  invocation may fail/fetch a broken runner — if so, call the exe directly):
  `cd /c/Users/wuc/.swa/deploy/134664351bac2104cb9ec773226e5b3fc265675d && ./StaticSitesClient.exe run --deploymentaction upload --workdir "<abs path to staged folder>" --apiToken "$TOKEN" --defaultHostname "agreeable-bush-035f3c510.4.azurestaticapps.net" --skipAppBuild --skipApiBuild --isStaticExport`

To add or edit content, edit the file in `showcase/`, then deploy the whole `showcase/`
folder (see the next section). The `dist/` build from `build_static.py` serves the chrona
app pages; keep them in the same staged folder as `showcase/` so nothing is dropped.

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
live page: edit the file in `showcase/`, commit, then deploy the whole `showcase/` folder
(it already includes `staticwebapp.config.json`) using one of the commands above. Because
the site is replaced wholesale, remember to deploy the complete folder, not a lone page.

The canonical / best recoverable copy of every live page is
`Q:\CodeGuru\scratch\site_index\staging_web\`; keep `showcase/` in sync with it.
The `dist/` build from `build_static.py` serves the chrona app pages; `showcase/` holds the
hub + catalog + dashboard pages that make up the rest of the site.

## Constraints / gotchas
- **Later change:** `clientSecretCertificateSettingName` is NOT a valid field in
  `staticwebapp.config.json` (SWA CLI rejects it) — only `openIdIssuer`,
  `clientIdSettingName`, `clientSecretSettingName` are valid for a custom AAD provider.
- `*.json` is gitignored by this repo's `.gitignore`, so `staticwebapp.config.json` stays
  out of git. If you ever want it tracked, `git add -f` it.
- **No server-side logic.** Both blob and ASWA-only setups have no backend; anything
  dynamic runs in-browser (the redaction demo runs client-side in `demo.js`).
