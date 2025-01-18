import os
from uuid import uuid4
from fastapi import UploadFile, File, HTTPException
from PIL import Image as PILImage
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch.nn as nn
import timm
from models.convnet import ConvNet

PHOBERT_MODEL_NAME = "vinai/phobert-base"
SWINV2_MODEL_NAME = "swinv2_base_window12_192.ms_in22k"
SWINV2_INPUT_HEIGHT = 192
NUM_CLASSES = 2
INPUT_SIZE = 1792
UPLOAD_FOLDER = './app/public/images'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_classify_2():
    # Tạo mô hình
    model = ConvNet(INPUT_SIZE, NUM_CLASSES) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tải checkpoint
    checkpoint = torch.load("app/ml-models/full_model_state_dict.pth", map_location=device)
    
    # Xử lý tiền tố 'module.' trong tên các lớp (nếu có)
    state_dict = checkpoint
    new_state_dict = {}
    for key, value in state_dict.items():
        # Nếu có tiền tố 'module.', loại bỏ nó
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    # Tải lại state_dict đã xử lý vào mô hình
    model.load_state_dict(new_state_dict)
    
    return model

def load_model_classify():
    # Tải lại mô hình đã lưu
    try:
        # Tạo lại mô hình với cùng kiến trúc
        model = ConvNet(INPUT_SIZE, NUM_CLASSES)   
        # Tải mô hình từ tệp và chuyển nó vào thiết bị (CPU/GPU)
        model.load_state_dict(torch.load("app/ml-models/semi-model-state-dict.pth"))
        model.eval()  # Đặt mô hình vào chế độ đánh giá (evaluation mode)
        
        return model

    except FileNotFoundError:
        print("Error: The model file was not found.")
        exit(1)
    except RuntimeError as e:
        print(f"Runtime error occurred: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


async def upload_image(file: UploadFile = File(...)):
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        "path": file_path, 
    }


# Hàm trích xuất đặc trưng văn bản
def extract_text_features(texts, device):
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    phobert_model = AutoModel.from_pretrained(PHOBERT_MODEL_NAME).to(device)
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=256).to(device)
    with torch.no_grad():
        model_output = phobert_model(**encoded_input)
    text_features = model_output.last_hidden_state[:, 0, :].cpu().numpy()
    return text_features

def preprocess_image(image_path):
    img = PILImage.open(image_path).convert("RGB")
    img = img.resize((SWINV2_INPUT_HEIGHT, SWINV2_INPUT_HEIGHT))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_tensor = torch.tensor(img_array).unsqueeze(0)
    return img_tensor

def extract_image_features(image_path):
    swin_model = timm.create_model(SWINV2_MODEL_NAME, pretrained=True, num_classes=0).to(device)
    if torch.cuda.device_count() > 1:
        swin_model = nn.DataParallel(swin_model, device_ids=[0, 1, 2])
    swin_model.eval()
    with torch.no_grad():
        img_tensor = preprocess_image(image_path).to(device)
        features = swin_model(img_tensor).cpu().numpy()
    return features

def dummy_text_features():
    return np.zeros(768).reshape(1, -1)

def predict_image(image_path, model):
   # Trích xuất đặc trưng hình ảnh
   image_features = extract_image_features(image_path)
   # Kết hợp đặc trưng văn bản và đặc trưng giả
   combined_features_new = np.concatenate((dummy_text_features(), image_features), axis=1)
   combined_features_new_tensor = torch.tensor(combined_features_new, dtype=torch.float32).to(device)
   try:
    # Dự đoán
    with torch.no_grad():
        logits, probs = model(combined_features_new_tensor)
        _, predicted = torch.max(logits, 1)
        return {
                "predicted": predicted.tolist()[0],
                "prob": probs.tolist()[0]
            }
   except Exception as e:
    # Xử lý lỗi, ví dụ in ra thông báo lỗi
    print(f"An error occurred during prediction: {e}")
    exit(1)