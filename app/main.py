from fastapi import FastAPI
from db import Base, engine
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.v1.router import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chấp nhận tất cả origin
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP (GET, POST, PUT, DELETE, ...)
    allow_headers=["*"],  # Cho phép tất cả các headers
)

# Mount thư mục static vào route `/static`
app.mount("/public", StaticFiles(directory="app/public"), name="public")

# Gắn router chính vào ứng dụng
app.include_router(router, prefix="/api/v1")

# Khởi tạo cơ sở dữ liệu
Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)



