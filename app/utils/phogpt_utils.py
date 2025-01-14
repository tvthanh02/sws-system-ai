from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import pandas as pd
from underthesea import word_tokenize








# =================== CẤU HÌNH =================== #
PHOBERT_MODEL_NAME = "vinai/phobert-base"
PHOGPT_MODEL_NAME = "vinai/phoGPT-4B-Chat"
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 0.00005
RANDOM_STATE = 42
GENERATE_LENGTH = 200
TEMPERATURE = 0.8
REPETITION_PENALTY = 1.1
TOP_K = 100
TOP_P = 0.9
MAX_LENGTH = 32
ACCUMULATION_STEPS = 16
CLASSIFICATION_MODEL_HIDDEN = 512
CLASSIFICATION_MODEL_NUM_CLASSES = 2








# =================== ĐỌC DỮ LIỆU =================== #
def read_dataset():
  """
  Đọc dữ liệu từ file CSV và trả về DataFrame.
  """
  try:
      data = pd.read_csv("Dataset_200Text_200Anh.csv", encoding='utf-8')
      return data
  except Exception as e:
      print(f"Lỗi khi đọc file: {str(e)}")
      return None








# =================== TIỀN XỬ LÝ DỮ LIỆU =================== #
def preprocess_vietnamese_text(text):
  """
  Tiền xử lý văn bản tiếng Việt: loại bỏ khoảng trắng thừa và phân tách từ.
  """
  if pd.isna(text):
      return ""
  if isinstance(text, bytes):
      text = text.decode('utf-8')
  text = text.strip()
  tokenized = word_tokenize(text)
  return ' '.join(tokenized)








# =================== TRÍCH XUẤT ĐẶC TRƯNG =================== #
def extract_features(text: str) -> torch.Tensor:
  """
  Trích xuất đặc trưng từ mô hình PhoBERT.
  Trả về hidden state của token [CLS] trong văn bản.
  """
  # Move model and tokenizer to the same device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
  model = AutoModel.from_pretrained(PHOBERT_MODEL_NAME).to(device)








  text = preprocess_vietnamese_text(text)
  inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
  with torch.no_grad():
      outputs = model(**inputs)
  # Lấy hidden states của lớp cuối cùng (CLS token)
  features = outputs.last_hidden_state[:, 0, :]
  return features








# =================== PHÂN LOẠI VĂN BẢN =================== #
def classify_text(text: str) -> bool:
  """
  Phân loại văn bản với mô hình PhoBERT.
  Trả về True nếu văn bản được phân loại là "AntiState", ngược lại False.
  """
  # Move model and tokenizer to the same device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
  model = AutoModel.from_pretrained(PHOBERT_MODEL_NAME).to(device)








  text = preprocess_vietnamese_text(text)
  inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
  with torch.no_grad():
      outputs = model(**inputs)
  logits = outputs.last_hidden_state[:, 0, :]
  probabilities = torch.softmax(logits, dim=-1).squeeze()
  return probabilities[0] > 0.5  # Giả định rằng có 2 lớp và lớp đầu tiên có thể biểu thị "AntiState"









# =================== SINH VĂN BẢN =================== #
def generate_text(text: str) -> str:
   """
   Sinh văn bản mới từ mô hình PhoGPT-4B.
   Trả về văn bản được sinh ra từ prompt đầu vào.
   """
   # Move model and tokenizer to the same device
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   tokenizer = AutoTokenizer.from_pretrained(PHOGPT_MODEL_NAME)
   model = AutoModelForCausalLM.from_pretrained(PHOGPT_MODEL_NAME).to(device)


   # Tiền xử lý văn bản
   text = preprocess_vietnamese_text(text)
   if not text:  # Nếu đầu vào rỗng sau khi tiền xử lý
       return "Đầu vào không hợp lệ hoặc trống."


   # Mã hóa đầu vào thành input_ids
   try:
       input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
   except Exception as e:
       print(f"Lỗi khi mã hóa đầu vào: {e}")
       return "Lỗi khi mã hóa đầu vào."


   # Sinh văn bản
   try:
       with torch.no_grad():
           output = model.generate(
               input_ids=input_ids,
               max_length=GENERATE_LENGTH,
               num_beams=5,
               no_repeat_ngram_size=2,
               early_stopping=True,
               top_p=TOP_P,
               temperature=TEMPERATURE,
               do_sample=True,
               top_k=TOP_K,
               repetition_penalty=REPETITION_PENALTY
           )
       # Giải mã đầu ra thành chuỗi văn bản
       generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
       return preprocess_vietnamese_text(generated_text)  # Tiền xử lý đầu ra để định dạng đẹp hơn
   except Exception as e:
       print(f"Lỗi khi sinh văn bản: {e}")
       return "Không thể sinh văn bản do lỗi nội bộ."




















