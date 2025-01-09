from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.dialects.postgresql import ARRAY
from database import Base

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    path = Column(String, nullable=False)
    predicted = Column(Integer, nullable=True)
    prob = Column(ARRAY(Float), nullable=True) 
