from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from model import predict_image, load_model
import os
from models import Base, Image
from database import engine, get_db
from fastapi import FastAPI, UploadFile, File, HTTPException
from extensions import upload_image
import os
from sqlalchemy.orm import Session

# Tạo thư mục nếu chưa tồn tại
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app = FastAPI()

mymodel = load_model();

# Khởi tạo cơ sở dữ liệu
Base.metadata.create_all(bind=engine)



@app.get("/images/{image_id}")
def get_image(image_id: int, db: Session = Depends(get_db)):
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"id": image.id, "filename": image.filename, "path": image.path}


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict-image")
async def predict_single_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Tải ảnh lên và lấy thông tin đường dẫn và tên tệp
    upload_results = await upload_image(file, db)
    image_filename = upload_results.get("filename")
    image_path = upload_results.get("path")

    if not image_filename or not image_path:
        raise HTTPException(status_code=400, detail="Image upload failed. Filename or path is missing.")

    try:
        # Dự đoán ảnh
        predicted_data = predict_image(image_path, mymodel)
        print(predicted_data)  # In ra dữ liệu dự đoán để debug
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

    # Lấy các giá trị dự đoán
    predicted, prob = predicted_data.get("predicted"), predicted_data.get("probs")
  
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