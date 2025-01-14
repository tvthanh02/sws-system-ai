from fastapi import APIRouter, Depends
from services.article_service import ArticleService
from schemas.articles_schemas import ArticlesResponseSchema
from db import get_db
from sqlalchemy.orm import Session

router = APIRouter()

@router.get("/articles/", response_model=ArticlesResponseSchema)
def get_article(db: Session = Depends(get_db)):
    article_service = ArticleService(db)
    return article_service.get_articles()
