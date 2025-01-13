from sqlalchemy import Column, String, Integer
from db.__init__ import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    account = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, default="user")  # Có thể có giá trị "admin", "user", ...