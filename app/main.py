from fastapi import FastAPI
from app.db import Base, engine
from fastapi.staticfiles import StaticFiles
from app.api.v1.router import router

app = FastAPI()

# Mount thư mục static vào route `/static`
app.mount("/public", StaticFiles(directory="app/public"), name="public")

# Gắn router chính vào ứng dụng
app.include_router(router, prefix="/api/v1")

# Khởi tạo cơ sở dữ liệu
Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)



