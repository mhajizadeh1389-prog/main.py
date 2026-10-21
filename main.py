import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient

app = FastAPI()

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

class ChatRequest(BaseModel):
    message: str

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "سلام! دستیار هوشمند سالن آماده است."}

@app.post("/chat")
def chat(request: ChatRequest):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN تنظیم نشده است.")
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": request.message}],
            max_tokens=100,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا: {str(e)}")

@app.get("/chat")
def chat_get(message: str):
    return chat(ChatRequest(message=message))
