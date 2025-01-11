import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.phobert_model import PhoBERTFeatureExtractor
from models.mt5_model import MT5TextGenerator
from config import (BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, 
                   MAX_LENGTH, MODEL_SAVE_PATH)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

class TextGenerationService:
    def __init__(self, device):
        self.device = device
        self.phobert = PhoBERTFeatureExtractor(device)
        self.mt5_generator = MT5TextGenerator(device)
        
    def train_model(self, train_data):
        print("Extracting features...")
        text_features = self.phobert.extract_features(train_data["title"].tolist())
        
        train_titles = train_data["title"].tolist()
        targets = [title + " tin tức " for title in train_titles]
        
        train_dataset = CustomDataset(text_features, targets=targets)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        optimizer = optim.Adam(self.mt5_generator.model.parameters(), lr=LEARNING_RATE)
        
        print("Starting training...")
        for epoch in range(NUM_EPOCHS):
            self.mt5_generator.model.train()
            total_loss = 0
            
            for batch_features, batch_targets in train_dataloader:
                inputs = self.mt5_generator.tokenizer(
                    batch_targets, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=MAX_LENGTH
                ).to(self.device)
                
                labels = inputs["input_ids"]
                outputs = self.mt5_generator.model(**inputs, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {total_loss:.4f}")
            
        # Save the trained model
        self.mt5_generator.save_model(MODEL_SAVE_PATH)
        
    def generate_text(self, prompt):
        return self.mt5_generator.generate(prompt)