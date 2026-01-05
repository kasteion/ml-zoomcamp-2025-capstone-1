from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class SentimentPayload(BaseModel):
    text:str = Field(min_length=1)

app = FastAPI(title="sentiment-classifier")

model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")

sentiment_map = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

def tokenize(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

def predict_single(text):
    device = model.device

    inputs = tokenize([text])

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    preds = torch.argmax(outputs.logits, dim=1)
    return preds.cpu().tolist()[0]

@app.post("/predict")
def predict(sentiment_payload: SentimentPayload):
    match = SentimentPayload.model_validate(sentiment_payload.model_dump())
    pred = predict_single(match.text)
    return {
        "prediction": sentiment_map[pred]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)