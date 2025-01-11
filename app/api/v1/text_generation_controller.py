from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from ..services.text_generation_service import TextGenerationService
from ..utils.data_utils import load_and_split_data
from ..config import DATASET_PATH, MODEL_SAVE_PATH

class Message(BaseModel):
    message: str

def setup_app():
    app = FastAPI()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    service = TextGenerationService(device)
    
    try:
        service.mt5_generator.load_model(MODEL_SAVE_PATH)
        print("Loaded saved model successfully")
    except Exception as e:
        print(f"No saved model found or error loading model: {str(e)}")
        print("Training new model...")
        train_data = load_and_split_data(DATASET_PATH)
        service.train_model(train_data)
    
    @app.post("/api/receive-data")
    async def receive_data(data: Message):
        try:
            generated_text = service.generate_text(data.message)
            return JSONResponse(content={
                "status": "success",
                "receivedMessage": data.message,
                "generatedText": generated_text
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app