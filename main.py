from fastapi import FastAPI, Request
from linebot import LineBotApi
from linebot.models import TextSendMessage
from moosy_core import recommend_by_artist, recommend_by_mood, recommend_thai
import os

app = FastAPI()

# --- Connect Line OA ---
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)

@app.post("/webhook")
async def webhook(req: Request):
    body = await req.json()
    events = body.get("events", [])

    for event in events:
        if event["type"] == "message" and event["message"]["type"] == "text":
            user_message = event["message"]["text"]
            reply_token = event["replyToken"]

            if "ขอเพลงของ" in user_message:
                artist = user_message.split("ขอเพลงของ")[-1].strip()
                reply_text = recommend_by_artist(artist, [], limit=5)

            elif "ขอเพลงไทย" in user_message.lower():
                reply_text = recommend_thai([], limit=5)

            elif "ขอเพลง" in user_message.lower() or "แนะนำเพลง" in user_message.lower():
                reply_text = recommend_by_mood(user_message, [], limit=5)

            else:
                reply_text = recommend_by_mood(user_message, [], limit=5)

            line_bot_api.reply_message(
                reply_token,
                TextSendMessage(text=reply_text)
            )

    return {"status": "ok"}