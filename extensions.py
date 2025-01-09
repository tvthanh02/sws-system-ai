import os
from uuid import uuid4
from fastapi import UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from models import  Image
from database import get_db
from PIL import Image as PILImage

UPLOAD_FOLDER = './uploads'

async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Kiểm tra định dạng file
    ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
    extension = file.filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Tạo tên file duy nhất
    unique_filename = f"{uuid4()}.{extension}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    # Lưu file vào server
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Xử lý ảnh (tuỳ chọn: resize, convert, v.v.)
    try:
        img = PILImage.open(file_path)
        img.verify()  # Kiểm tra ảnh hợp lệ
    except Exception:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Invalid image file")


    return {
        "filename": unique_filename,
        "path": file_path
    }