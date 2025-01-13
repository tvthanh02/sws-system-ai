from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from utils.common import upload_image, load_model_classify, predict_image 
from models.image import Image
from db import get_db
from sqlalchemy.orm import Session
from utils.phogpt_utils import classify_text, generate_text
from schemas.process_schema import ProcessRequest, ProcessResponse
from core.security import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES , get_current_user
from schemas.user_schema import UserCreate , LoginRequest
from services.user_services import get_user_by_account, create_user, verify_password
from utils.hash import verify_password
from models.user_model import User

router = APIRouter()

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

    # ví dụ localhost:8000/api/v1/process nhập json chứa : text trả vể json : giống return
@router.post("/process", response_model=ProcessResponse, tags=["Process"])
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

    # ví dụ localhost:8000/api/v1/token nhập json chứa : account, password trả vể json : access_token, token_type
@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = get_user_by_account(db, account=request.account)
    if not user or not verify_password(request.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user.account})
    return {"access_token": access_token, "token_type": "bearer"}

    # ví dụ localhost:8000/api/v1/register nhập json chứa : account, password, role trả vể json : id, account, role
@router.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_account(db, user.account):
        raise HTTPException(status_code=400, detail="Account already registered")
    return create_user(db, user)

    # ví dụ localhost:8000/api/v1/protected-endpoint nhập json chứa : access_token trả vể json : message
@router.get("/protected-endpoint")
def protected_route(current_user: User = Depends(get_current_user)):
    return {"message": f"Hello {current_user.account}, you are authorized!"}