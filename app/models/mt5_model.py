import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from underthesea import word_tokenize
from config import (MT5_MODEL_NAME, TEMPERATURE, TOP_P, TOP_K, 
                   REPETITION_PENALTY, GENERATE_LENGTH)

class MT5TextGenerator:
    def __init__(self, device):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(MT5_MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(MT5_MODEL_NAME).to(device)
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        
    def generate(self, prompt, max_length=GENERATE_LENGTH, temperature=TEMPERATURE):
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', padding=True).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                top_p=TOP_P,
                temperature=temperature,
                do_sample=True,
                top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY
            )
            
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        for i in range(100):
            generated_text = generated_text.replace(f"<extra_id_{i}>", "")
            
        generated_text = generated_text.strip()
        generated_text = ' '.join(word_tokenize(generated_text))
        return generated_text