from repositories.article_repository import ArticleRepository
from repositories.predict_repository import PredictRepository

class ArticleService:
    def __init__(self, db):
        self.db = db
        self.article_repo = ArticleRepository(db)

    def get_articles(self, skip: int = 0, limit: int = 100):
        return self.article_repo.get_all_article()

    def get_article(self, article_id: int):
        return self.article_repo.get_article_by_id(article_id)

    def delete_article(self, article_id: int):
        result = self.article_repo.delete_article_by_id(article_id)
        if not result:
            raise ValueError(f"Không tìm thấy bài viết có id {article_id}.")
        return {"message":"Bài viết đã được xóa thành công"}
