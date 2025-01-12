from sqlalchemy.orm import Session
from fastapi import Depends
from app.models.predict import Predict
from app.db import get_db

class PredictRepository:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db

    def get_all_predict(self):
        return self.db.query(Predict).all()

    def get_article_by_id(self, predict_id):
        return self.db.query(Predict).filter(Predict.id == predict_id).first()
