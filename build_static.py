"""Render the seeded showcase DB into static HTML for Azure Static Web Apps.

Run:  python build_static.py      (writes to ./dist)
"""
import os
import shutil
import markdown
from sqlalchemy.orm import Session

from models import Page, get_engine
from db import seed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(BASE_DIR, "dist")
engine = get_engine()
seed(engine)

md = markdown.Markdown(extensions=["tables", "fenced_code"])


def render_md(text):
    md.reset()
    return md.convert(text or "")


NAV = (
    '<a class="brand" href="/">Chrona</a>'
    '<nav class="nav">'
)

SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Chrona</title>
<link rel="stylesheet" href="/style.css">
</head>
<body>
<header class="site-header"><div class="wrap header-inner">{nav}</div></header>
<main class="wrap">{content}</main>
<footer class="site-footer wrap"><p>Chrona — ensemble-redaction privacy pipeline showcase.</p></footer>
</body>
</html>
"""


def page_to_html(page, pages, active_slug):
    nav = NAV
    for p in pages:
        cls = ' class="active"' if p.slug == active_slug else ""
        href = "/" if not p.slug else f"/{p.slug}.html"
        nav += f'<a href="{href}"{cls}>{p.nav_label}</a>'
    nav += "</nav>"

    hero = f'<div class="hero">{render_md(page.hero)}</div>'
    sections = "".join(
        f'<section class="card">'
        f'<h2>{s.heading}</h2><div class="md">{render_md(s.body)}</div></section>'
        for s in sorted(page.sections, key=lambda s: s.position)
    )
    content = f"<article>{hero}{sections}</article>"
    return SHELL.format(title=page.title, nav=nav, content=content)


def main():
    if os.path.exists(DIST):
        shutil.rmtree(DIST)
    os.makedirs(DIST)

    shutil.copytree(
        os.path.join(BASE_DIR, "static"), os.path.join(DIST, "static"),
    )
    # serve style.css at root too (linked as /style.css)
    shutil.copy(
        os.path.join(BASE_DIR, "static", "style.css"),
        os.path.join(DIST, "style.css"),
    )

    with Session(engine) as session:
        pages = session.query(Page).order_by(Page.id).all()
        for page in pages:
            html = page_to_html(page, pages, page.slug)
            # root overview -> index.html
            sees = "index.html" if not page.slug else f"{page.slug}.html"
            with open(os.path.join(DIST, sees), "w", encoding="utf-8") as f:
                f.write(html)

    # demo page is served by the API function; a thin static shell explains it
    with Session(engine) as session:
        pages = session.query(Page).order_by(Page.id).all()
        demo = next(p for p in pages if p.slug == "demo")
        html = page_to_html(demo, pages, "demo")
    # replace the empty sections with a live form that posts to /api/redact
    form = (
        '<section class="card"><h2>Live redaction demo</h2>'
        '<form class="demo-form" id="demo-form">'
        '<textarea id="raw" rows="8" spellcheck="false">'
        '{"raw_queries": ["how to treat diabetes", "best mortgage rates 2026", '
        '"John Smith from Seattle", "https://youtube.com/watch?v=abc"]}'
        "</textarea>"
        '<button type="button" onclick="runDemo()">Redact</button>'
        "</form>"
        '<div id="out"></div></section>'
        '<script src="/demo.js"></script>'
    )
    start = html.index("</article>")
    html = html[:start] + form + html[start:start]
    with open(os.path.join(DIST, "demo.html"), "w", encoding="utf-8") as f:
        f.write(html)

    with open(os.path.join(DIST, "demo.js"), "w", encoding="utf-8") as f:
        f.write(JS)
    print("Static build written to", DIST)


JS = r"""
// Client-side mirror of src/privacy_core.PrivacyRedactor._mask_query demo logic.
const NOISE = [
  /^https?:\/\//i, /youtube\.com/i, /login/i, /homepage/i,
  /translator/i, /^google$/i, /^facebook$/i, /^mail$/i, /^\w+\.\w+$/i,
];
const counters = {};

function maskQuery(query, category) {
  const lower = query.toLowerCase().trim();
  for (const re of NOISE) if (re.test(lower)) return null;
  counters[category] = (counters[category] || 0) + 1;
  const n = String(counters[category]).padStart(3, '0');
  return { token: 'QUERY_' + category + '_' + n };
}

function runDemo() {
  const out = document.getElementById('out');
  let data;
  try {
    data = JSON.parse(document.getElementById('raw').value);
  } catch (e) {
    out.innerHTML = '<div class="error"><strong>Error:</strong> Invalid JSON: ' + e.message + '</div>';
    return;
  }
  const result = {};
  if (Array.isArray(data.raw_queries)) {
    const q = [];
    for (const item of data.raw_queries) {
      if (typeof item === 'string') {
        const t = maskQuery(item, 'QUERY');
        if (t) q.push(t);
      }
    }
    if (q.length) result.queries = q;
  }
  if (Array.isArray(data.browsing_history)) {
    const b = [];
    for (const item of data.browsing_history) {
      if (typeof item === 'string') {
        const t = maskQuery(item, 'BROWSING');
        if (t) b.push(t);
      }
    }
    if (b.length) result.browsing = b;
  }
  out.innerHTML = '<h3>Masked output (raw PII never reaches the LLM)</h3>'
    + '<pre class="output">' + JSON.stringify(result, null, 2) + '</pre>';
}
"""


if __name__ == "__main__":
    main()
