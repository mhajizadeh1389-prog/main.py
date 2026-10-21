import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

app = FastAPI()

MODEL_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
MODEL_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"

# مدل را در سطح ماژول بارگذاری نمی‌کنیم. یک متغیر سراسری برای نگهداری آن داریم.
llm = None

def load_model():
    """این تابع مدل را فقط زمانی بارگذاری می‌کند که برای اولین بار فراخوانی شود."""
    global llm
    if llm is None:
        print("Downloading and loading model (lazy)...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        llm = Llama(model_path=model_path, n_ctx=256, verbose=False)
        print("Model loaded successfully!")
    return llm

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "سلام! من دستیار هوشمند سالن هستم. سرویس آماده است."}

@app.post("/chat")
def chat(request: ChatRequest):
    model = load_model() # مدل را (اگر قبلاً بارگذاری نشده) بارگذاری کن
    prompt = f"<|im_start|>user\n{request.message}<|im_end|>\n<|im_start|>assistant\n"
    output = model(
        prompt,
        max_tokens=100,
        temperature=0.7,
        stop=["<|im_end|>"],
    )
    response_text = output["choices"][0]["text"].strip()
    return {"response": response_text}

@app.get("/chat")
def chat_get(message: str):
    # برای تست از مرورگر، مستقیماً تابع POST را صدا می‌زنیم
    return chat(ChatRequest(message=message))
