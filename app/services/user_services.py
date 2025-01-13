# app/services/user.py
from sqlalchemy.orm import Session
from models.user_model import User
from schemas.user_schema import UserCreate
from utils.hash import hash_password, verify_password

def get_user_by_account(db: Session, account: str):
    return db.query(User).filter(User.account == account).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = hash_password(user.password)
    db_user = User(account=user.account, password=hashed_password, role="user")
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
