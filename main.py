import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient

app = FastAPI()

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

client = None

def get_client():
    global client
    if client is None:
        if not HF_TOKEN:
            raise HTTPException(status_code=500, detail="HF_TOKEN تنظیم نشده است.")
        client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    return client

class ChatRequest(BaseModel):
    message: str

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "سلام! دستیار هوشمند سالن آماده است."}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        c = get_client()
        response = c.chat_completion(
            messages=[{"role": "user", "content": request.message}],
            max_tokens=200,
            temperature=0.7
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا: {str(e)}")

@app.get("/chat")
def chat_get(message: str):
    return chat(ChatRequest(message=message))
