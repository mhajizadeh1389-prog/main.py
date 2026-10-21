import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

app = FastAPI()

MODEL_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
MODEL_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"

print("Downloading and loading model...")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
llm = Llama(model_path=model_path, n_ctx=256, verbose=False)
print("Model loaded successfully!")


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "سلام! من دستیار هوشمند سالن هستم. به بخش chat خوش آمدید."}


@app.post("/chat")
def chat(request: ChatRequest):
    prompt = f"<|im_start|>user\n{request.message}<|im_end|>\n<|im_start|>assistant\n"
    output = llm(
        prompt,
        max_tokens=100,
        temperature=0.7,
        stop=["<|im_end|>"],
    )
    response_text = output["choices"][0]["text"].strip()
    return {"response": response_text}


@app.get("/chat")
def chat_get(message: str):
    return chat(ChatRequest(message=message))
