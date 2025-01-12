from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from app.utils.common import upload_image, load_model_classify, predict_image
from app.models.image import Image
from app.db import get_db
from sqlalchemy.orm import Session
from app.services.article_service import ArticleService
from app.schemas.articles_schemas import ArticlesResponseSchema
import pytesseract
import os
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path

router = APIRouter()

# Thư mục tạm để lưu file
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Đường dẫn Poppler (cần thiết nếu bạn dùng Windows)
POPLER_PATH = r"C:\Program Files\Poppler\poppler-24.08.0\Library\bin"


@router.get("/test")
def test():
    return {"message": "Hello World"}

@router.post("/predict-image")
async def predict_single_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    
    # load classify model
    semi_classify_model = load_model_classify()

    # Tải ảnh lên và lấy thông tin đường dẫn và tên tệp
    upload_results = await upload_image(file)
    image_filename = upload_results.get("filename")
    image_path = upload_results.get("path")

    if not image_filename or not image_path:
        raise HTTPException(status_code=400, detail="Image upload failed. Filename or path is missing.")

    try:
        # Dự đoán ảnh
        predicted_data = predict_image(image_path, semi_classify_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

    # Lấy các giá trị dự đoán
    predicted, prob = predicted_data.get("predicted"), predicted_data.get("prob")
  
    # Kiểm tra dữ liệu đã đầy đủ chưa
    if predicted is None or prob is None or len(prob) == 0:
        raise HTTPException(status_code=400, detail="Prediction data is missing or invalid.")

    # Lưu vào cơ sở dữ liệu
    image_data = Image(
        filename=image_filename,
        path=image_path,
        predicted=predicted,
        prob=prob
    )

    # Lưu thay đổi vào cơ sở dữ liệu
    db.add(image_data)
    db.commit()
    db.refresh(image_data)

    # Trả về dữ liệu đã lưu
    return {
        "id": image_data.id,
        "filename": image_data.filename,
        "path": image_data.path,
        "predicted": image_data.predicted,
        "prob": image_data.prob
    }

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    # Kiểm tra file hợp lệ
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Lưu file tạm thời
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # Phân loại và xử lý OCR
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            text = process_image(file_path)
        elif file.filename.lower().endswith('.pdf'):
            text = process_pdf(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        return JSONResponse(content={"text": text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        # Xóa file tạm
        if os.path.exists(file_path):
            os.remove(file_path)


def process_image(image_path: str) -> str:
    """
    Xử lý OCR trên file ảnh.
    """
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='vie')
    return text


def process_pdf(pdf_path: str) -> str:
    """
    Xử lý OCR trên file PDF bằng cách chuyển PDF sang ảnh.
    """
    images = convert_from_path(pdf_path, dpi=300, poppler_path=POPLER_PATH)
    text = ""
    for page_number, image in enumerate(images, start=1):
        page_text = pytesseract.image_to_string(image, lang='vie')
        text += f"\n--- Page {page_number} ---\n"
        text += page_text
    return text