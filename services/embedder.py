from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()


class RerankRequest(BaseModel):
    query: str
    passages: list[str]


@app.post("/rerank")
def rerank(req: RerankRequest):
    pairs = [[req.query, p] for p in req.passages]
    encoded = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits.squeeze(-1)
        scores = torch.sigmoid(logits).tolist()

    results = [
        {"passage": p, "score": s}
        for p, s in zip(req.passages, scores if isinstance(scores, list) else [scores])
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results}