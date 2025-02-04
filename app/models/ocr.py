from sqlalchemy import Column, Integer, String,Text, TIMESTAMP, func
from db import Base

class UploadedFile(Base):
    __tablename__ = "ocr"
    id = Column(Integer, primary_key=True, index=True,autoincrement=True)
    filename = Column(String, nullable=False)
    text = Column(Text, nullable=True)
    upload_time = Column(TIMESTAMP, server_default=func.now())
