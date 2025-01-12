from sqlalchemy.orm import Session
from fastapi import Depends
from app.models.dantri_articles import DantriArticles
from app.models.predict import Predict
from app.db import get_db

class ArticleRepository:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db

    def get_all_article(self):
        # return self.db.query(DantriArticles, Predict.predicted).join(Predict, DantriArticles.id == Predict.id).all()
        return self.db.query(DantriArticles).all()

    def get_article_by_id(self, article_id):
        return self.db.query(DantriArticles).filter(DantriArticles.id == article_id).first()
