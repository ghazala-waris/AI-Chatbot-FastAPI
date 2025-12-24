from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import random

# Load trained model and data
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("responses.pkl", "rb") as f:
    responses = pickle.load(f)

app = FastAPI(title="AI Chatbot API")

class Message(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "Welcome to AI Chatbot API! Go to /docs to test."}

@app.post("/chat")
def chat(msg: Message):
    input_text = msg.message
    X_test = vectorizer.transform([input_text])
    prediction = model.predict(X_test)[0]
    reply = random.choice(responses[prediction])
    return {"reply": reply}
