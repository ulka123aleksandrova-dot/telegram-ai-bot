import os
import asyncio
import logging

from aiohttp import web
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from aiogram.enums import ChatAction
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from openai import OpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")              # https://....up.railway.app
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/tg/webhook")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me")

PORT = int(os.getenv("PORT", "8080"))                 # Railway сам задаёт PORT

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN (Railway Variables)")
if not OPENAI_API_KEY:
    raise RuntimeError("Не найден OPENAI_API_KEY (Railway Variables)")
if not WEBHOOK_BASE:
    raise RuntimeError("Не найден WEBHOOK_BASE (Railway Variables)")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)

@dp.message(F.photo)
async def get_photo_file_id(message: Message):
    photo = message.photo[-1]
    await message.answer(f"📸 file_id:\n{photo.file_id}")

@dp.message(F.video)
async def get_video_file_id(message: Message):
    await message.answer(f"🎥 file_id:\n{message.video.file_id}")

MODEL = "gpt-4o-mini-2024-07-18"
SYSTEM_PROMPT = "Ты дружелюбный Telegram-бот помощник. Отвечай кратко и понятно."


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer("Привет 👋 Напиши вопрос — отвечу нейросетью 🙂")


@dp.message(F.text)
async def chat(message: Message):
    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    def call_openai(text: str) -> str:
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.7,
            max_output_tokens=350,
        )
        return (resp.output_text or "").strip()

    try:
        answer = await asyncio.to_thread(call_openai, message.text)
        await message.answer(answer or "Я задумался 😅 Попробуй иначе.")
    except Exception as e:
        log.exception("OpenAI error: %s", e)
        await message.answer("⚠️ Сейчас я немного перегружен. Попробуй через минуту 🙂")


async def on_startup(app: web.Application):
    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_webhook(
        url=f"{WEBHOOK_BASE}{WEBHOOK_PATH}",
        secret_token=WEBHOOK_SECRET,
    )
    log.info("Webhook set: %s%s", WEBHOOK_BASE, WEBHOOK_PATH)


async def on_shutdown(app: web.Application):
    await bot.delete_webhook()
    await bot.session.close()


def main():
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
        secret_token=WEBHOOK_SECRET,
    ).register(app, path=WEBHOOK_PATH)

    setup_application(app, dp, bot=bot)
    web.run_app(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()





