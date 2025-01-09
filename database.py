from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://postgres:06122002@localhost/data_images"
DATABASE_URL_RENDER = "postgresql://dantri_data_user:4lSj7wD5RZe5y3T2m2SR8cmWyvq2yOko@dpg-ctvnprtds78s73eohh40-a/dantri_data"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
