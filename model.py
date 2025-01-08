import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import numpy as np
import timm
from PIL import Image



class ConvNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        logits = self.fc5(x)
        probs = torch.softmax(logits, dim=1)
        return logits, probs


PHOBERT_MODEL_NAME = "vinai/phobert-base"
SWINV2_MODEL_NAME = "swinv2_base_window12_192.ms_in22k"
SWINV2_INPUT_HEIGHT = 192
NUM_CLASSES = 2



# Thiết lập thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hàm trích xuất đặc trưng văn bản
def extract_text_features(texts, device):
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    phobert_model = AutoModel.from_pretrained(PHOBERT_MODEL_NAME).to(device)
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=256).to(device)
    with torch.no_grad():
        model_output = phobert_model(**encoded_input)
    text_features = model_output.last_hidden_state[:, 0, :].cpu().numpy()
    return text_features


def load_model():
    # Tải lại mô hình đã lưu
   try:
    # Tải mô hình từ tệp và chuyển nó vào thiết bị (CPU/GPU)
    model = ConvNet(input_size=1792, output_size=NUM_CLASSES)  # Tạo lại mô hình với cùng kiến trúc
    model.load_state_dict(torch.load("full_model_state_dict.pth"))
    model = model.to(device)
    model.eval()  # Đặt mô hình vào chế độ đánh giá (evaluation mode)
    return model
    
   except FileNotFoundError:
    print("Error: The model file 'full_model.pth' was not found.")
    exit(1)
   except RuntimeError as e:
    print(f"Runtime error occurred: {e}")
    exit(1)


# try:
#     # Tải mô hình từ tệp và chuyển nó vào thiết bị (CPU/GPU)
#     model = torch.load("full_model.pth", map_location=device)
#     model = model.to(device)
#     model.eval()  # Đặt mô hình vào chế độ đánh giá (evaluation mode)
    
# except FileNotFoundError:
#     print("Error: The model file 'full_model.pth' was not found.")
#     exit(1)
# except RuntimeError as e:
#     print(f"Runtime error occurred: {e}")
#     exit(1)
# except Exception as e:
#     print(f"An error occurred: {e}")
#     exit(1)


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((SWINV2_INPUT_HEIGHT, SWINV2_INPUT_HEIGHT))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_tensor = torch.tensor(img_array).unsqueeze(0)
    return img_tensor

def extract_single_image_features(image_path):
    swin_model = timm.create_model(SWINV2_MODEL_NAME, pretrained=True, num_classes=0).to(device)
    if torch.cuda.device_count() > 1:
        swin_model = nn.DataParallel(swin_model, device_ids=[0, 1, 2])
    swin_model.eval()
    with torch.no_grad():
        img_tensor = preprocess_image(image_path).to(device)
        features = swin_model(img_tensor).cpu().numpy()
    return features


dummy_text_features = np.zeros(768) # Vector văn bản giả với tất cả giá trị bằng 0


def predict_image(image_path, model):
   # Trích xuất đặc trưng hình ảnh
   image_features = extract_single_image_features(image_path)
   # Kết hợp đặc trưng văn bản và đặc trưng giả
   combined_features_new = np.concatenate((dummy_text_features.reshape(1, -1), image_features), axis=1)
   combined_features_new_tensor = torch.tensor(combined_features_new, dtype=torch.float32).to(device)
   try:
    # Dự đoán
    with torch.no_grad():
        logits, probs = model(combined_features_new_tensor)
        _, predicted = torch.max(logits, 1)
        return {
                "predicted": predicted.tolist()[0],
                "probs": probs.tolist()[0]
            }
   except Exception as e:
    # Xử lý lỗi, ví dụ in ra thông báo lỗi
    print(f"An error occurred during prediction: {e}")
    exit(1)

