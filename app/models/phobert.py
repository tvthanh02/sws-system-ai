import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from app.core.config import PHOBERT_MODEL_NAME

class PhoBERTFeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(PHOBERT_MODEL_NAME).to(device)
        
    def extract_features(self, texts, max_length=128):
        all_features = []
        batch_size = 64
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, 
                                   return_tensors='pt', max_length=max_length).to(self.device)
            
            with torch.no_grad():
                output = self.model(**encoded)
            features = output.last_hidden_state[:, 0, :].cpu().numpy()
            all_features.append(features)
            
        return np.vstack(all_features)