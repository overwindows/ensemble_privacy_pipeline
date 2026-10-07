import os
import markdown
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from models import Page, get_engine, init_db
from db import seed

BASE_DIR = os.path.dirname(__file__)
engine = get_engine()
seed(engine)

app = FastAPI(title="Ensemble-Redaction Privacy Pipeline")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

md = markdown.Markdown(extensions=["tables", "fenced_code"])


def render_md(text: str) -> str:
    md.reset()
    return md.convert(text or "")


def get_pages(session: Session):
    return session.query(Page).order_by(Page.id).all()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _page(request, "")


def _page(request: Request, slug: str):
    with Session(engine) as session:
        pages = get_pages(session)
        page = session.query(Page).filter(Page.slug == slug).first()
        if page is None:
            return templates.TemplateResponse(
                request, "404.html",
                {"pages": pages, "title": "Not found"},
            )
        sections = [(s.heading, render_md(s.body)) for s in sorted(page.sections, key=lambda s: s.position)]
        return templates.TemplateResponse(
            request, "page.html",
            {"pages": pages, "page": page, "hero": render_md(page.hero), "sections": sections},
        )


@app.get("/page/{slug}", response_class=HTMLResponse)
def page(request: Request, slug: str):
    return _page(request, slug)


@app.get("/demo", response_class=HTMLResponse)
def demo_form(request: Request):
    with Session(engine) as session:
        pages = get_pages(session)
    return templates.TemplateResponse(
        request, "demo.html",
        {"pages": pages, "page": None, "result": None, "input": DEFAULT_SAMPLE},
    )


DEFAULT_SAMPLE = (
    '{"raw_queries": ["how to treat diabetes", '
    '"https://youtube.com/watch?v=abc", '
    '"best mortgage rates 2026", '
    '"John Smith from Seattle", '
    '"cancer treatment options"], '
    '"browsing_history": ["webmd.com", "bankofamerica.com"]}'
)


@app.post("/demo", response_class=HTMLResponse)
def demo_run(request: Request, raw: str = Form("")):
    with Session(engine) as session:
        pages = get_pages(session)
    output = None
    error = None
    try:
        import json
        from src.privacy_core import PrivacyRedactor
        data = json.loads(raw)
        redactor = PrivacyRedactor()
        masked = redactor.redact_user_data(data)
        output = json.dumps(masked, indent=2)
    except Exception as exc:  # noqa: BLE001 - surface errors to the user
        error = f"{type(exc).__name__}: {exc}"
    return templates.TemplateResponse(
        request, "demo.html",
        {"pages": pages, "page": None, "result": output, "error": error, "input": raw},
    )
