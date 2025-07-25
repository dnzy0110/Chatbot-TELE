# === File: main.py ===
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.bot import telegram_app, start_telegram
from app.routes import setup_routes
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- Define the app lifespan events (startup/shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_telegram() # Start the Telegram bot loop
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static") #can delete, didnt use.
templates = Jinja2Templates(directory="templates")

setup_routes(app, templates)

# --- Entry point for running the app directly ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

#if want to run local backend then
# for chat log history "http://localhost:5000/chatlogs"
# for chat bot "http://127.0.0.1:5000/"
# for FASTAPI "http://127.0.0.1:8000/docs#/default/evaluate_qa_evaluate_qa__post" for testing accuracy
#run on terminal to start ↑ ".\venv\Scripts\uvicorn.exe main:app --reload"