from app.repositories.article_repository import ArticleRepository
from app.repositories.predict_repository import PredictRepository

class ArticleService:
    def __init__(self, db):
        self.db = db
        self.article_repo = ArticleRepository(db)

    def get_articles(self, skip: int = 0, limit: int = 100):
        return self.article_repo.get_all_article()

    def get_article(self, article_id: int):
        return self.article_repo.get_article(article_id)

    def delete_article(self, article_id: int):
        return self.article_repo.delete_article(article_id)
