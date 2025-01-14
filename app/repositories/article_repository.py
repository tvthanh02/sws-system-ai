from sqlalchemy.orm import Session
from fastapi import Depends
from models.dantri_articles import DantriArticles
from models.predict import Predict
from db import get_db

class ArticleRepository:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db

    def get_all_article(self):
        # return self.db.query(DantriArticles, Predict.predicted).join(Predict, DantriArticles.id == Predict.id).all()
        return self.db.query(DantriArticles).all()

    def get_article_by_id(self, article_id):
        return self.db.query(DantriArticles).filter(DantriArticles.id == article_id).first()
    
    def delete_article_by_id(self,article_id):
        article=self.db.query(DantriArticles).filter(DantriArticles.id == article_id).first()
        if article:
            self.db.delete(article)
            self.db.commit()
            return True
        return False
