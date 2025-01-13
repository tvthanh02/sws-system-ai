# app/schemas/user.py
from pydantic import BaseModel

class UserBase(BaseModel):
    account: str
    role: str

class UserCreate(BaseModel):
    account: str
    password: str

class LoginRequest(BaseModel):
    account: str
    password: str

class UserResponse(UserBase):
    id: int

    class Config:
        orm_mode = True
