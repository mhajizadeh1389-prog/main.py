from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "سلام! من دستیار هوشمند سالن هستم. سیستم کار می‌کند."}
