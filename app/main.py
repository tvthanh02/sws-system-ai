from fastapi import FastAPI
from db import Base, engine
from fastapi.staticfiles import StaticFiles
from api.v1.router import router
import os
import sys
import uvicorn
from app.api.v1.text_generation_controller import setup_app

app = FastAPI()
# Thêm thư mục cha vào PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Mount thư mục static vào route `/static`
app.mount("/public", StaticFiles(directory="app/public"), name="public")

# Gắn router chính vào ứng dụng
app.include_router(router, prefix="/api/v1")

# Khởi tạo cơ sở dữ liệu
Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    # uvicorn.run(app, host="127.0.0.1", port=5000)


