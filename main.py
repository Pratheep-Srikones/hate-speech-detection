import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from utils import clean_text

app = FastAPI()

PREDICTIONS_MAP = {
    0:"Hate Speech",
    1:"Offensive Language",
    2:"Neither"
}
model = joblib.load('models/rf_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

class TextData(BaseModel):
    text: str
@app.get("/")
def read_root():
    return {"message": "Welcome to the Text Classification API!"}
@app.post("/predict")
def predict(data: TextData):
    print("Request Received")
    print(f"Received {data.text}")
    text = data.text
    text = clean_text(text)
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return {"prediction": PREDICTIONS_MAP[int(prediction[0])]}