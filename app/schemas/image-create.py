from pydantic import BaseModel

class ImageCreate(BaseModel):
    filename: str
    path: str

    class Config:
        orm_mode = True
