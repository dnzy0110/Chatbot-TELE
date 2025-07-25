# === File: routes.py ===
from fastapi import Request
from fastapi.responses import JSONResponse, HTMLResponse
from app.bot import telegram_app, user_last_seen, user_warned, COOLDOWN_SECONDS
from telegram import Update
import time
import os
import json

# Setup routes for FastAPI app
def setup_routes(app, templates):

    # --- TELEGRAM WEBHOOK HANDLER ---
    @app.post(f"/webhook/{{token}}")
    async def telegram_webhook(request: Request, token: str):
        if token != os.getenv("TELEGRAM_BOT_TOKEN"):
            return JSONResponse(status_code=403, content={"error": "Forbidden"})

        # Parse incoming Telegram update payload
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        user = update.effective_user

        # If no user in update, return success response without processing
        if not user:
            return JSONResponse(content={"status": "no user found"}, status_code=200)

        # --- COOLDOWN RATE LIMITING ---
        user_id = user.id
        now = time.time()
        last_seen = user_last_seen.get(user_id, 0)
        warned = user_warned.get(user_id, False)
        time_diff = now - last_seen

        # If user sends message too soon, warn once and ignore the request
        if time_diff < COOLDOWN_SECONDS:
            if not warned:
                remaining = int(COOLDOWN_SECONDS - time_diff)
                try:
                    await telegram_app.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=f"⏳ Please wait {remaining} more second(s) before sending another message."
                    )
                except Exception:
                    pass # Fail silently if message can't be sent
                user_warned[user_id] = True
            return JSONResponse(content={"status": "ignored due to cooldown"}, status_code=200)

        # Update user state after cooldown passed
        user_last_seen[user_id] = now
        user_warned[user_id] = False

        # Forward update to Telegram bot handlers (e.g., message handler)
        await telegram_app.process_update(update)
        return JSONResponse(content={"status": "processed"}, status_code=200)

    # --- INDEX PAGE ROUTE (e.g., home page) ---
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    # --- CHATLOGS VIEWER ROUTE ---
    @app.get("/chatlogs", response_class=HTMLResponse)
    async def view_chatlogs(request: Request, query: str = ""):
        path = "chat_history.json"
        logs = []

        # Read the chat history file and build structured user-bot message pairs
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    raw_logs = json.load(f)
                    i = 0
                    while i < len(raw_logs) - 1:
                        user_entry = raw_logs[i]
                        bot_entry = raw_logs[i + 1]

                        # Match user → bot message pairs based on roles
                        if user_entry.get("role") == "user" and bot_entry.get("role") == "bot":
                            pair = {
                                "user_id": user_entry.get("user_id"),
                                "user_message": user_entry.get("message"),
                                "bot_response": bot_entry.get("message"),
                                "timestamp": user_entry.get("timestamp")
                            }
                            # If search query matches any field, include the pair
                            if (
                                query.lower() in user_entry.get("message", "").lower()
                                or query.lower() in bot_entry.get("message", "").lower()
                                or query.lower() in str(user_entry.get("user_id", "")).lower()
                                or query == ""
                            ):
                                logs.append(pair)
                            i += 2 # Move to next user-bot pair
                        else:
                            i += 1 # Skip incomplete or malformed log entries
                except json.JSONDecodeError:
                    pass # If file is corrupt or not valid JSON, just skip

        # Render logs in HTML template
        return templates.TemplateResponse("chatlogs.html", {"request": request, "logs": logs, "query": query})