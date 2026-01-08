import os
import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from aiogram.enums import ChatAction

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN в .env")
if not OPENAI_API_KEY:
    raise RuntimeError("Не найден OPENAI_API_KEY в .env")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o-mini-2024-07-18"


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Привет 👋\n"
        "Я бот с нейросетью.\n"
        "Напиши мне сообщение 🙂"
    )


@dp.message(F.text)
async def chat(message: Message):
    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    try:
        resp = client.responses.create(
            model=MODEL,
            input=message.text,
            temperature=0.7,
            max_output_tokens=300,
        )
        answer = (resp.output_text or "").strip() or "Я задумался 😅 Попробуй ещё раз."
        await message.answer(answer)

    except Exception as e:
        log.exception("OpenAI error: %s", e)
        await message.answer(
            "⚠️ Сейчас я не могу обратиться к OpenAI (ограничение региона/доступа).\n"
            "Чтобы работало стабильно у всех — нужно запустить бота на VPS за границей."
        )


async def main():
    try:
        await dp.start_polling(bot)
    finally:
        # чтобы не было Unclosed client session
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())

