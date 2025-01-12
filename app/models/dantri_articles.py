from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.dialects.postgresql import ARRAY
from app.db import Base

class DantriArticles(Base):
    __tablename__ = "dantri_articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    url = Column(String, unique=True, index=True)
    summary = Column(String)