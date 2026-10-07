from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
import os

Base = declarative_base()

# DB file lives on App Service's persistent /home disk so content survives restarts.
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "showcase.db"))


class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True)
    slug = Column(String, unique=True, nullable=False)   # url path, e.g. "results"
    title = Column(String, nullable=False)
    nav_label = Column(String, nullable=False)
    hero = Column(Text, default="")   # optional short intro (markdown)
    sections = relationship(
        "Section", order_by="Section.position", back_populates="page",
        cascade="all, delete-orphan",
    )


class Section(Base):
    __tablename__ = "sections"

    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    position = Column(Integer, default=0)
    heading = Column(String, default="")
    body = Column(Text, default="")   # markdown rendered to HTML

    page = relationship("Page", back_populates="sections")


def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def init_db(engine):
    Base.metadata.create_all(engine)
