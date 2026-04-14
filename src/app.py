from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

# 1. Definisi format input data
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

# 2. Load model yang sudah dilatih
# Pastikan file model.pkl ada di folder utama setelah training
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    model = None

@app.get("/")
def home():
    return {"message": "Iris Prediction API is Running"}

@app.post("/predict")
def predict(data: IrisInput):
    # Konversi input ke format DataFrame
    input_df = pd.DataFrame([data.dict().values()], 
                            columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    
    # Lakukan prediksi
    prediction = model.predict(input_df)
    return {"prediction": str(prediction[0])}