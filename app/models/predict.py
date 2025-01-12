from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.dialects.postgresql import ARRAY
from app.db import Base

class Predict(Base):
    __tablename__ = "dudoan"

    id = Column(Integer, primary_key=True, index=True)
    summary = Column(String)
    predicted = Column(Integer, nullable=True)
    probs = Column(ARRAY(Float), nullable=True)