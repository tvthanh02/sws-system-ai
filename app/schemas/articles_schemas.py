# app/schemas/user_schema.py
from pydantic import BaseModel

class ArticlesResponseSchema(BaseModel):
    id: int
    title: str
    url: str
    summary: str
    # predicted: str

    class Config:
        orm_mode = True  # Cho phép chuyển đổi từ các model SQLAlchemy        
