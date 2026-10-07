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
    '<a class="brand" href="/">Ensemble-Redaction Privacy Pipeline</a>'
    '<nav class="nav">'
)

SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Ensemble Privacy Pipeline</title>
<link rel="stylesheet" href="/style.css">
</head>
<body>
<header class="site-header"><div class="wrap header-inner">{nav}</div></header>
<main class="wrap">{content}</main>
<footer class="site-footer wrap"><p>Showcase for the ensemble-redaction privacy pipeline.</p></footer>
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
async function runDemo() {
  const out = document.getElementById('out');
  out.innerHTML = '<em>Redacting…</em>';
  try {
    const resp = await fetch('/api/redact', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: document.getElementById('raw').value,
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || (await resp.text()));
    out.innerHTML = '<h3>Masked output (raw PII never reaches the LLM)</h3>'
      + '<pre class="output">' + JSON.stringify(data, null, 2) + '</pre>';
  } catch (e) {
    out.innerHTML = '<div class="error"><strong>Error:</strong> ' + e.message + '</div>';
  }
}
"""


if __name__ == "__main__":
    main()
