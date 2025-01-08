from fastapi import FastAPI
from model import predict_image, load_model


app = FastAPI()

mymodel = load_model();


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/predict-image")
def predict_single_image():
   image_path = 'images/image10.png'
   results = predict_image(image_path, mymodel)
   return results