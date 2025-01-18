from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request,Query
from utils.common import upload_image, load_model_classify, predict_image
from models.image import Image 
from db import get_db
from sqlalchemy.orm import Session,sessionmaker
from services.article_service import ArticleService
from schemas.articles_schemas import ArticlesResponseSchema
import pytesseract
import os
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
from PIL import Image 
from models.ocr import UploadedFile
from schemas.process_schema import ProcessRequest, ProcessResponse
from utils.phogpt_utils import classify_text,generate_text


router = APIRouter()
# Thư mục tạm để lưu file
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

@router.get("/images")
def get_images(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    try:
        # Tính toán offset và tổng số ảnh
        offset = (page - 1) * limit
        total = db.query(Image).count()
        total_pages = (total + limit - 1) // limit  # Tính số trang

        # Lấy danh sách ảnh với phân trang
        images = (
            db.query(Image)
            .order_by(Image.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        # Chuẩn bị response
        response = {
            "data": [
                {
                    "_id": image.id,
                    "image": image.path,
                    "predicted": image.predicted,
                    "prob": image.prob,
                    "created_at": None,  # Nếu cần thêm `created_at` thì cần sửa model
                    "updated_at": None,  # Tương tự với `updated_at`
                }
                for image in images
            ],
            "meta": {
                "total": total,
                "currentPage": page,
                "totalPages": total_pages,
                "limit": limit,
            },
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/images/{image_id}")
def delete_image(image_id: int, db: Session = Depends(get_db)):
    try:
        # Tìm ảnh cần xóa
        image = db.query(Image).filter(Image.id == image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        # Xóa ảnh khỏi cơ sở dữ liệu
        db.delete(image)
        db.commit()

        return {"message": f"Image with ID {image_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ví dụ localhost:8000/api/v1/process nhập json chứa : text trả vể json : giống return
@router.post("/generate-text", response_model=ProcessResponse, tags=["Process"])
async def process_text(request: ProcessRequest):
    try:
        # Kiểm tra nếu không có văn bản
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")

        # Phân loại nội dung
        is_anti_state = classify_text(request.text)
        
        if is_anti_state:
            return {
                "title": request.text,
                "isAntiState": True,
                "generated_text": "Nội dung không được phép xử lý."
            }
        else:
            # Sinh văn bản nếu không phải nội dung phản động
            generated_text = generate_text(request.text)
            return {
                "title": request.text,
                "isAntiState": False,
                "generated_text": generated_text
            }
    
    except Exception as e:
        # Nếu có lỗi trong quá trình xử lý, trả về lỗi 500
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):

    # Kiểm tra file hợp lệ
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Đường dẫn đầy đủ để lưu file trong thư mục upload
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        # Lưu file vào thư mục
        with open(file_path, "wb") as f:
            f.write(await file.read())


        # Phân loại và xử lý OCR
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            text = process_image(file_path)
        elif file.filename.lower().endswith('.pdf'):
            text = process_pdf(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Lưu thông tin vào cơ sở dữ liệu
        uploaded_file = UploadedFile(
            filename=file.filename,
            text=text
        )
        db.add(uploaded_file)
        db.commit()
        db.refresh(uploaded_file)

        return JSONResponse(content={
            "id": uploaded_file.id,
            "filename": uploaded_file.filename,
            "text": uploaded_file.text
            })

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
    images = convert_from_path(pdf_path, dpi=300)
    text = ""
    for page_number, image in enumerate(images, start=1):
        page_text = pytesseract.image_to_string(image, lang='vie')
        text += f"\n--- Page {page_number} ---\n"
        text += page_text
    return text