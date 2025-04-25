from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import os

# 👇 import core จากไฟล์ moosy_core.py
from moosy_core import recommend_song, df

# 📌 โหลดค่า ENV (จาก Render / Secrets)
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# ✅ LINE SDK setup
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ✅ FastAPI app
app = FastAPI()

# 🏠 root endpoint
@app.get("/")
def read_root():
    return {"message": "Moosy LINE Bot is alive!"}

# 📬 LINE Webhook
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()

    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    return "OK"

# 💬 Handle incoming text messages from user
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_input = event.message.text

    # เรียก Moosy core มาใช้งาน
    response_text, _ = recommend_song(user_input, df, seen_songs=[])
    
    # ตอบกลับผ่าน LINE
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response_text)
    )