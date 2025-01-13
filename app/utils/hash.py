from passlib.context import CryptContext

# Sử dụng PassLib để quản lý băm mật khẩu
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """
    Băm mật khẩu bằng bcrypt.
    """
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Kiểm tra mật khẩu gốc với mật khẩu đã băm.
    """
    return pwd_context.verify(plain_password, hashed_password)
