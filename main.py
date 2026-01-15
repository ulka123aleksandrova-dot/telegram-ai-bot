import os
import re
import time
import yaml
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from aiohttp import web
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart
from aiogram.enums import ChatAction, ContentType
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram.utils.keyboard import InlineKeyboardBuilder

from openai import OpenAI

# ----------------------------
# CONFIG / ENV
# ----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")  # https://....up.railway.app
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/tg/webhook")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")  # ваш chat_id (в личке обычно = user_id)
PORT = int(os.getenv("PORT", "8080"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN (Railway Variables)")
if not OPENAI_API_KEY:
    raise RuntimeError("Не найден OPENAI_API_KEY (Railway Variables)")
if not WEBHOOK_BASE:
    raise RuntimeError("Не найден WEBHOOK_BASE (Railway Variables)")
if not WEBHOOK_SECRET:
    raise RuntimeError("Не найден WEBHOOK_SECRET (Railway Variables)")
if not ADMIN_CHAT_ID:
    raise RuntimeError("Не найден ADMIN_CHAT_ID (Railway Variables)")

ADMIN_CHAT_ID_INT = int(ADMIN_CHAT_ID)

# ----------------------------
# BOT / DISPATCHER / OPENAI
# ----------------------------
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)

KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge.yaml")

# ----------------------------
# KNOWLEDGE LOAD
# ----------------------------
def load_knowledge() -> dict:
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

knowledge = load_knowledge()

# ----------------------------
# MEMORY (in RAM)
# ----------------------------
HISTORY_MAX_TURNS = 10            # память диалога: последние 10 реплик (user+assistant)
STATE_TTL_SECONDS = 60 * 60 * 6   # 6 часов

class Stage:
    ASK_NAME = "ask_name"
    QUALIFY = "qualify"
    SELL = "sell"
    BUY_COLLECT = "buy_collect"
    WAIT_RECEIPT = "wait_receipt"
    CONFIRM_RECEIPT = "confirm_receipt"

@dataclass
class UserProfile:
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

@dataclass
class UserState:
    stage: str = Stage.ASK_NAME
    chosen_tariff: Optional[str] = None
    last_seen: float = field(default_factory=lambda: time.time())
    history: List[dict] = field(default_factory=list)  # [{"role":"user","content":...},...]
    profile: UserProfile = field(default_factory=UserProfile)
    pending_receipt_file_id: Optional[str] = None

user_state: Dict[int, UserState] = {}

def cleanup_states(now: float) -> None:
    dead = [uid for uid, st in user_state.items() if (now - st.last_seen) > STATE_TTL_SECONDS]
    for uid in dead:
        user_state.pop(uid, None)

def add_history(uid: int, role: str, content: str) -> None:
    st = user_state.setdefault(uid, UserState())
    st.history.append({"role": role, "content": content})
    # ограничиваем память
    if len(st.history) > HISTORY_MAX_TURNS * 2:
        st.history = st.history[-HISTORY_MAX_TURNS * 2 :]

# ----------------------------
# HELPERS: name parsing / validation
# ----------------------------
NAME_RE = re.compile(r"^(?:меня зовут|я)\s+([A-Za-zА-Яа-яЁё\-]+)(?:\s+([A-Za-zА-Яа-яЁё\-]+))?$", re.IGNORECASE)
TWO_WORDS_RE = re.compile(r"^([A-Za-zА-Яа-яЁё\-]{2,})\s+([A-Za-zА-Яа-яЁё\-]{2,})$")

def extract_name(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = text.strip()
    m = NAME_RE.match(t)
    if m:
        return m.group(1), m.group(2)
    m2 = TWO_WORDS_RE.match(t)
    if m2:
        # часто пишут "Имя Фамилия"
        return m2.group(1), m2.group(2)
    # одиночное слово
    if re.fullmatch(r"[A-Za-zА-Яа-яЁё\-]{2,}", t):
        return t, None
    return None, None

PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{9,}\d)")
EMAIL_RE = re.compile(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})")

def extract_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text)
    return m.group(1).strip() if m else None

def extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text)
    return m.group(1).strip() if m else None

def is_buy_intent(text: str) -> bool:
    t = text.lower()
    keywords = ["куп", "оплат", "заказать", "оформ", "беру", "хочу тариф", "готов", "покупаю"]
    return any(k in t for k in keywords)

def is_guest_request(text: str) -> bool:
    t = text.lower()
    return "гост" in t or "демо" in t or "пробн" in t

def is_presentation_request(text: str) -> bool:
    t = text.lower()
    return "презентац" in t or "презу" in t

# ----------------------------
# HELPERS: admin notifications
# ----------------------------
async def send_admin(text: str) -> None:
    try:
        await bot.send_message(ADMIN_CHAT_ID_INT, text)
    except Exception as e:
        log.exception("Failed to send admin message: %s", e)

# ----------------------------
# HELPERS: typing animation (3-5 sec)
# ----------------------------
async def typing_loop(chat_id: int, stop_event: asyncio.Event) -> None:
    try:
        while not stop_event.is_set():
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4)
    except Exception:
        # не критично
        return

# ----------------------------
# HELPERS: split answer into 1-2 messages
# ----------------------------
def split_answer(text: str, max_chars: int = 850) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    # режем по абзацам
    parts = [p.strip() for p in t.split("\n\n") if p.strip()]
    out: List[str] = []
    buf = ""
    for p in parts:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf += "\n\n" + p
        else:
            out.append(buf)
            buf = p
        if len(out) >= 2:
            break
    if buf and len(out) < 2:
        out.append(buf)

    # если всё равно слишком длинно — грубо обрезаем
    out = [s[:max_chars].rstrip() for s in out]
    return out[:2]

# ----------------------------
# KNOWLEDGE ACCESSORS
# ----------------------------
def kget(path: str, default=None):
    cur = knowledge
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def format_tariffs_short() -> str:
    tariffs = kget("тарифы", [])
    lines = []
    for t in tariffs:
        name = t.get("название")
        price = t.get("цена")
        lines.append(f"• {name} — {price}")
    return "\n".join(lines)

def find_course_price(name_query: str) -> Optional[str]:
    courses = kget("курсы", [])
    q = name_query.lower()
    for c in courses:
        if q in (c.get("название", "").lower()):
            return c.get("цена")
    return None

# ----------------------------
# SYSTEM PROMPT (конкретика + без воды)
# ----------------------------
def build_system_prompt(uid: int) -> str:
    st = user_state.setdefault(uid, UserState())
    first = st.profile.first_name or "друг"
    owner = kget("проект.куратор", "Юлия")
    assistant_name = kget("проект.ассистент", "Лиза")

    rules = (
        f"Ты — {assistant_name}, помощница куратора {owner} онлайн-школы INSTART.\n"
        f"Обращайся к клиенту по имени, если известно: {first}.\n\n"
        "КЛЮЧЕВАЯ ЗАДАЧА: вести к покупке мягко, без давления, но быстро.\n"
        "СТИЛЬ: живое общение, немного эмодзи, без воды.\n\n"
        "ОГРАНИЧЕНИЯ:\n"
        "— НИКОГДА не придумывай цены/состав тарифов/курсов. Используй ТОЛЬКО данные из базы knowledge.yaml.\n"
        "— Если информации нет в базе: скажи честно 'уточню у куратора' и предложи оставить контакты.\n"
        "— Не обещай гарантированный доход.\n\n"
        "ФОРМАТ ОТВЕТА:\n"
        "— 2–6 коротких предложений.\n"
        "— Если нужно: 1–4 пункта списком.\n"
        "— В конце: 1 конкретный вопрос (следующий шаг).\n"
        "— Иногда можно разбить на 2 сообщения, но НЕ всегда.\n\n"
        "СЕЙЛЗ-ЛОГИКА:\n"
        "— Сначала уточни цель и время.\n"
        "— Затем предложи 1–2 наиболее подходящих тарифа с ценой.\n"
        "— Если человек готов: предложи оформить покупку и собрать контакты.\n"
    )

    # компактная выжимка по тарифам
    tariffs_block = "ТАРИФЫ (коротко):\n" + format_tariffs_short()

    # гостевой ключ (если есть)
    guest = kget("инструкции.гостевой_ключ")
    guest_block = f"\nГостевой ключ (если просят): {guest}" if guest else ""

    return rules + "\n" + tariffs_block + guest_block

# ----------------------------
# START / onboarding
# ----------------------------
@dp.message(CommandStart())
async def start(message: Message):
    uid = message.from_user.id if message.from_user else message.chat.id
    now = time.time()
    cleanup_states(now)

    st = user_state.setdefault(uid, UserState())
    st.last_seen = now
    st.stage = Stage.ASK_NAME

    await message.answer(
        "Привет! 😊\n\n"
        "Я Лиза — помощница куратора Юлии в онлайн-школе INSTART.\n"
        "Очень рада знакомству 🌿\n\n"
        "Давай познакомимся: как тебя зовут?"
    )

# ----------------------------
# PHOTO / VIDEO: чек принимаем только в нужном состоянии
# ----------------------------
@dp.message(F.photo)
async def on_photo(message: Message):
    uid = message.from_user.id if message.from_user else message.chat.id
    st = user_state.setdefault(uid, UserState())
    st.last_seen = time.time()

    if st.stage != Stage.WAIT_RECEIPT:
        # обычное фото — не считаем чеком
        await message.answer("Вижу фото 🙂 Если это чек об оплате — напиши, пожалуйста, текстом: «это чек», и я попрошу загрузить его ещё раз.")
        return

    photo = message.photo[-1]
    st.pending_receipt_file_id = photo.file_id
    st.stage = Stage.CONFIRM_RECEIPT

    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Да, это чек", callback_data="receipt_yes")
    kb.button(text="❌ Нет, не чек", callback_data="receipt_no")
    kb.adjust(2)

    await message.answer(
        "Я получила фото. Подтверди, пожалуйста: это чек об оплате? 🙂",
        reply_markup=kb.as_markup(),
    )

@dp.message(F.video)
async def on_video(message: Message):
    uid = message.from_user.id if message.from_user else message.chat.id
    st = user_state.setdefault(uid, UserState())
    st.last_seen = time.time()

    # видео чаще всего не чек — реагируем мягко
    await message.answer("Вижу видео 🙂 Если хочешь — уточни, что именно показать/объяснить по INSTART, и я помогу.")

@dp.callback_query(F.data.in_(["receipt_yes", "receipt_no"]))
async def receipt_confirm(cb: CallbackQuery):
    uid = cb.from_user.id if cb.from_user else cb.message.chat.id
    st = user_state.setdefault(uid, UserState())
    st.last_seen = time.time()

    await cb.answer()

    if cb.data == "receipt_no":
        st.pending_receipt_file_id = None
        st.stage = Stage.SELL
        await cb.message.answer("Ок 🙂 Тогда продолжим. Хочешь — подберу тариф под твою цель.")
        return

    # receipt_yes
    file_id = st.pending_receipt_file_id
    st.pending_receipt_file_id = None
    st.stage = Stage.SELL

    await cb.message.answer("Принято ✅ Я передам информацию Юлии и она подтвердит оплату.")

    # уведомляем администратора
    lead = (
        "✅ ПРИШЁЛ ЧЕК ОБ ОПЛАТЕ\n"
        f"Клиент: {st.profile.first_name or ''} {st.profile.last_name or ''}\n"
        f"Тариф: {st.chosen_tariff or 'не указан'}\n"
        f"Телефон: {st.profile.phone or 'не указан'}\n"
        f"Email: {st.profile.email or 'не указан'}\n"
        f"User ID: {uid}"
    )
    await send_admin(lead)

    # форвардим фото чека админу
    if file_id:
        try:
            await bot.send_photo(ADMIN_CHAT_ID_INT, photo=file_id, caption="Чек от клиента (как подтвердили)")
        except Exception as e:
            log.exception("Failed to forward receipt photo: %s", e)

# ----------------------------
# MAIN TEXT HANDLER
# ----------------------------
@dp.message(F.text)
async def chat(message: Message):
    uid = message.from_user.id if message.from_user else message.chat.id
    now = time.time()
    cleanup_states(now)

    st = user_state.setdefault(uid, UserState())
    st.last_seen = now

    text = (message.text or "").strip()
    if not text:
        return

    # 1) Стадия ASK_NAME: ловим имя из текста
    if st.stage == Stage.ASK_NAME:
        first, last = extract_name(text)
        if first:
            st.profile.first_name = first
            st.profile.last_name = last
            st.stage = Stage.QUALIFY
            await message.answer(
                f"{first}, очень приятно познакомиться! 😊\n\n"
                "Подскажи, пожалуйста:\n"
                "1) Какая цель сейчас ближе — подработка / новая профессия / развитие в проекте?\n"
                "2) Сколько времени в неделю реально готов(а) уделять обучению? (например 3–5 часов)"
            )
            add_history(uid, "assistant", f"Запомнила имя: {first} {last or ''}")
        else:
            await message.answer("Супер 🙂 Как тебя зовут? (Можно просто имя)")
        return

    # 2) Гостевой доступ / презентация — отдаём кодом, без LLM противоречий
    if is_guest_request(text):
        guest_key = kget("инструкции.гостевой_ключ")
        if guest_key:
            await message.answer(
                "Конечно! Вот гостевой доступ 🎁\n\n"
                f"🔑 Гостевой ключ: `{guest_key}`\n\n"
                "Хочешь — я подскажу, как его активировать (в 2 шага)."
            )
            add_history(uid, "assistant", "Выдала гостевой ключ клиенту.")
        else:
            await message.answer("Гостевой ключ сейчас не найден в базе. Могу уточнить у Юлии — оставить телефон/почту?")
        return

    if is_presentation_request(text):
        pres = kget("медиа.презентация")
        if pres and pres.get("file_id"):
            await bot.send_document(message.chat.id, document=pres["file_id"], caption="Презентация INSTART 📎")
            add_history(uid, "assistant", "Отправила презентацию клиенту.")
            await message.answer("Если скажешь цель (подработка/профессия/партнёрство) — подберу лучший старт и тариф 🙂")
        else:
            await message.answer("Презентации в базе пока нет. Могу уточнить у Юлии — хочешь оставить контакт?")
        return

    # 3) Намерение купить: запускаем сбор контактов
    if is_buy_intent(text):
        st.stage = Stage.BUY_COLLECT
        await message.answer(
            "Отлично 😊 Давай оформим.\n\n"
            "Напиши, пожалуйста, одним сообщением:\n"
            "• Имя и фамилия\n"
            "• Телефон\n"
            "• Email\n"
            "И какой тариф выбрал(а) (если уже решил(а))"
        )
        return

    # 4) Если мы собираем контакты — парсим и шлём админу, дальше даём реквизиты
    if st.stage == Stage.BUY_COLLECT:
        # имя/фамилия
        first, last = extract_name(text)
        if first and not st.profile.first_name:
            st.profile.first_name = first
        if last and not st.profile.last_name:
            st.profile.last_name = last

        phone = extract_phone(text)
        email = extract_email(text)
        if phone:
            st.profile.phone = phone
        if email:
            st.profile.email = email

        # тариф (упрощенно по ключевым словам/номеру)
        t = text.lower()
        if "1990" in t or "базов" in t or "тариф 1" in t or "1" == t.strip():
            st.chosen_tariff = "Тариф 1 «Базовые курсы» — 1990₽"
        elif "2990" in t or "новые" in t or "тариф 2" in t or "2" == t.strip():
            st.chosen_tariff = "Тариф 2 «Новые направления» — 2990₽"

        # если чего-то не хватает — попросим
        missing = []
        if not st.profile.first_name or not st.profile.last_name:
            missing.append("имя и фамилия")
        if not st.profile.phone:
            missing.append("телефон")
        if not st.profile.email:
            missing.append("email")

        if missing:
            await message.answer("Чтобы оформить правильно, мне не хватает: " + ", ".join(missing) + " 🙂\nНапиши, пожалуйста.")
            return

        # отправляем админу
        lead_text = (
            "🟩 ЗАЯВКА НА ПОКУПКУ\n"
            f"ФИО: {st.profile.first_name} {st.profile.last_name}\n"
            f"Телефон: {st.profile.phone}\n"
            f"Email: {st.profile.email}\n"
            f"Тариф: {st.chosen_tariff or 'не указан'}\n"
            f"User ID: {uid}"
        )
        await send_admin(lead_text)

        # реквизиты оплаты (как вы просили)
        await message.answer(
            "Супер, спасибо! 😊\n\n"
            "Оплата по номеру телефона:\n"
            "📞 89883873424\n"
            "🏦 Банк: Кубань Кредит\n\n"
            "После оплаты пришли, пожалуйста, чек (фото) сюда в чат — и я передам Юлии на подтверждение ✅"
        )
        st.stage = Stage.WAIT_RECEIPT
        return

    # 5) Ответ LLM (конкретный, по базе)
    add_history(uid, "user", text)

    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(typing_loop(message.chat.id, stop_event))
    start_ts = time.time()

    def call_openai(prompt_messages: List[dict]) -> str:
        resp = client.responses.create(
            model=MODEL,
            input=prompt_messages,
            temperature=0.5,
            max_output_tokens=240,  # укоротили примерно на треть и чуть больше контроля
        )
        return (resp.output_text or "").strip()

    try:
        sys_prompt = build_system_prompt(uid)

        # история + system
        msgs = [{"role": "system", "content": sys_prompt}]
        msgs.extend(st.history[-HISTORY_MAX_TURNS * 2 :])
        msgs.append({"role": "user", "content": text})

        answer = await asyncio.to_thread(call_openai, msgs)

        # гарантируем 3 секунды "печатает"
        elapsed = time.time() - start_ts
        if elapsed < 3.0:
            await asyncio.sleep(3.0 - elapsed)

        parts = split_answer(answer, max_chars=850)
        if not parts:
            parts = ["Я задумалась 😅 Напиши чуть иначе — и я помогу."]

        # отправляем 1-2 сообщения
        for p in parts:
            await message.answer(p)

        add_history(uid, "assistant", answer)

    except Exception as e:
        log.exception("OpenAI error: %s", e)
        await message.answer("⚠️ Сейчас я немного перегружена. Попробуй через минуту 🙂")
    finally:
        stop_event.set()
        try:
            await typing_task
        except Exception:
            pass

# ----------------------------
# WEBHOOK SETUP
# ----------------------------
async def on_startup(app: web.Application):
    # не дропаем апдейты, чтобы не терять сообщения при перезапуске
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
