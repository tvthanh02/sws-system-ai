# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from dotenv import load_dotenv
# import os

# load_dotenv()

# POSTGRES_URL = os.getenv("POSTGRES_URL_CONSTR")
# POSTGRES_URL_RENDER = os.getenv("POSTGRES_URL_RENDER")

# engine = create_engine(POSTGRES_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
