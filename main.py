import os
import re
import time
import yaml
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from aiohttp import web
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart, Command
from aiogram.enums import ChatAction
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram.utils.keyboard import InlineKeyboardBuilder

from openai import OpenAI


# =========================
# ENV / BOOT
# =========================
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")              # https://xxxx.up.railway.app
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/tg/webhook")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me")

ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")            # можно не задавать
PORT = int(os.getenv("PORT", "8080"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN (Railway Variables)")
if not OPENAI_API_KEY:
    raise RuntimeError("Не найден OPENAI_API_KEY (Railway Variables)")
if not WEBHOOK_BASE:
    raise RuntimeError("Не найден WEBHOOK_BASE (Railway Variables), например https://xxxx.up.railway.app")

ADMIN_CHAT_ID_INT: Optional[int] = int(ADMIN_CHAT_ID) if ADMIN_CHAT_ID else None

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# ВАЖНО: используем chat.completions (а не responses), чтобы не было ошибки "no attribute responses"
client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# KNOWLEDGE
# =========================
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge.yaml")


def load_knowledge() -> dict:
    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        log.error("knowledge.yaml не найден рядом с main.py. Путь: %s", KNOWLEDGE_PATH)
        return {}
    except Exception as e:
        log.exception("Ошибка чтения knowledge.yaml: %s", e)
        return {}

    if data is None:
        return {}

    if not isinstance(data, dict):
        log.error("knowledge.yaml должен быть YAML-словарём (mapping) в корне.")
        return {}

    return data


knowledge = load_knowledge()


def kget(path: str, default=None):
    cur: Any = knowledge
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


ASSISTANT_NAME = kget("assistant.name", "Лиза")
OWNER_NAME = kget("assistant.owner_name", "Юлия")
PROJECT_NAME = kget("project.name", "INSTART")


def courses_list() -> List[dict]:
    c = kget("courses", [])
    return c if isinstance(c, list) else []


def tariffs_list() -> List[dict]:
    t = kget("tariffs", [])
    return t if isinstance(t, list) else []


# =========================
# STATE / MEMORY
# =========================
HISTORY_MAX_TURNS = 10
STATE_TTL_SECONDS = 6 * 60 * 60  # 6 часов


class Stage:
    ASK_NAME = "ask_name"
    QUALIFY = "qualify"
    NORMAL = "normal"
    BUY_COLLECT = "buy_collect"
    WAIT_RECEIPT = "wait_receipt"
    CONFIRM_RECEIPT = "confirm_receipt"


@dataclass
class UserProfile:
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    gender: Optional[str] = None  # "m"/"f"/None
    phone: Optional[str] = None
    email: Optional[str] = None


@dataclass
class UserState:
    stage: str = Stage.ASK_NAME
    interest_id: Optional[str] = None  # course/tariff id
    interest_type: Optional[str] = None  # "course"/"tariff"
    chosen_tariff_title: Optional[str] = None
    chosen_tariff_price: Optional[int] = None
    last_seen: float = field(default_factory=lambda: time.time())
    history: List[dict] = field(default_factory=list)
    profile: UserProfile = field(default_factory=UserProfile)
    pending_receipt_file_id: Optional[str] = None
    sent_media_keys: set = field(default_factory=set)


user_state: Dict[int, UserState] = {}


def cleanup_states(now: float) -> None:
    dead = [uid for uid, st in user_state.items() if now - st.last_seen > STATE_TTL_SECONDS]
    for uid in dead:
        user_state.pop(uid, None)


def add_history(uid: int, role: str, content: str) -> None:
    st = user_state.setdefault(uid, UserState())
    st.history.append({"role": role, "content": content})
    if len(st.history) > HISTORY_MAX_TURNS * 2:
        st.history = st.history[-HISTORY_MAX_TURNS * 2 :]


# =========================
# HELPERS: parsing
# =========================
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{9,}\d)")
EMAIL_RE = re.compile(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})")


def extract_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text or "")
    return m.group(1).strip() if m else None


def extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text or "")
    return m.group(1).strip() if m else None


def normalize_phone(s: str) -> str:
    return re.sub(r"[^\d+]", "", s or "")


def looks_like_email(s: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", (s or "").strip()))


def guess_gender_by_name(name: str) -> Optional[str]:
    # грубая эвристика
    n = (name or "").strip().lower()
    if not n:
        return None
    female_endings = ("а", "я")
    male_exceptions = {"никита", "илья", "фома", "кузьма"}
    if n in male_exceptions:
        return "m"
    if n.endswith(female_endings):
        return "f"
    return "m"


def extract_name(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Умеет доставать имя даже из длинной фразы:
    "привет! меня зовут Юлия. хочу узнать..." -> Юлия
    """
    if not text:
        return None, None

    # 1) "меня зовут Юлия", "я Юлия"
    m = re.search(r"(?:меня\s+зовут|я)\s+([А-ЯЁA-Z][а-яёa-z\-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1), None

    # 2) если сообщение состоит только из 1–2 слов (имя/имя фамилия)
    words = re.findall(r"[А-ЯЁA-Z][а-яёa-z\-]+", text)
    if len(words) == 1:
        return words[0], None
    if len(words) >= 2 and len(text.strip().split()) <= 3:
        return words[0], words[1]

    return None, None


BUY_INTENT_RE = re.compile(r"\b(купить|оплат(ить|а)|готов(а)?\s+купить|беру\s+тариф|хочу\s+тариф|оформим)\b", re.IGNORECASE)


def is_guest_request(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in ["гост", "демо", "пробн", "ключ"])


def is_tariff_question(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in ["тариф", "цена", "стоим", "сколько"])


# =========================
# HELPERS: find by aliases
# =========================
def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def find_item_by_alias(text: str) -> Optional[dict]:
    """
    Ищет курс/тариф по aliases (включая title).
    Возвращает сам словарь item.
    """
    q = normalize(text)

    # 1) тариф по "тариф 1/2/3"
    m = re.search(r"\bтариф\s*(\d)\b", q)
    if m:
        idx = int(m.group(1))
        for t in tariffs_list():
            aliases = [normalize(t.get("title", ""))] + [normalize(a) for a in (t.get("aliases") or [])]
            if normalize(f"тариф {idx}") in aliases:
                return t

    # 2) прямой проход по тарифам и курсам
    for item in tariffs_list() + courses_list():
        aliases = [normalize(item.get("title", ""))] + [normalize(a) for a in (item.get("aliases") or [])]
        for a in aliases:
            if not a:
                continue
            # точное совпадение или включение алиаса в запрос
            if q == a or a in q:
                return item

    return None


def format_tariffs_brief() -> str:
    lines = []
    for t in tariffs_list():
        title = t.get("title")
        price = t.get("price_rub")
        if title and price is not None:
            lines.append(f"• {title} — {price} ₽")
        elif title:
            lines.append(f"• {title}")
    return "\n".join(lines) if lines else "Пока не вижу тарифы в knowledge.yaml."


def course_price_line(course: dict) -> str:
    price = course.get("price") or {}
    with_chat = price.get("with_chat_rub")
    without_chat = price.get("without_chat_rub")

    if with_chat is not None and without_chat is not None:
        return f"Стоимость: {without_chat} ₽ (без чата) / {with_chat} ₽ (с чатом)"
    if with_chat is not None:
        return f"Стоимость: {with_chat} ₽"
    if without_chat is not None:
        return f"Стоимость: {without_chat} ₽"
    return "Стоимость: уточню у куратора."


# =========================
# MEDIA
# =========================
def media_get(key: str) -> Optional[dict]:
    media = kget("media", {})
    if isinstance(media, dict) and key in media and isinstance(media[key], dict):
        return media[key]
    return None


async def send_media_by_key(message: Message, key: str, caption_override: Optional[str] = None) -> bool:
    st = user_state.setdefault(message.from_user.id, UserState())
    if key in st.sent_media_keys:
        return False

    m = media_get(key)
    if not m:
        return False

    mtype = m.get("type")
    fid = m.get("file_id")
    caption = caption_override or m.get("caption") or m.get("title") or ""

    if not fid:
        return False

    if mtype == "photo":
        await message.answer_photo(photo=fid, caption=caption[:1024] if caption else None)
        st.sent_media_keys.add(key)
        return True
    if mtype == "video":
        await message.answer_video(video=fid, caption=caption[:1024] if caption else None)
        st.sent_media_keys.add(key)
        return True
    if mtype == "document":
        await message.answer_document(document=fid, caption=caption[:1024] if caption else None)
        st.sent_media_keys.add(key)
        return True

    return False


# =========================
# ADMIN
# =========================
async def send_admin(text: str) -> None:
    if not ADMIN_CHAT_ID_INT:
        return
    try:
        await bot.send_message(ADMIN_CHAT_ID_INT, text)
    except Exception as e:
        log.exception("Failed to send admin message: %s", e)


# =========================
# typing helper
# =========================
async def typing_loop(chat_id: int, stop_event: asyncio.Event) -> None:
    try:
        while not stop_event.is_set():
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(3.5)
    except Exception:
        return


def split_answer(text: str, max_chars: int = 900) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

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
    return [s[:max_chars].rstrip() for s in out[:2]]


# =========================
# SYSTEM PROMPT (Ваш промпт, кратко)
# =========================
def build_system_prompt(uid: int) -> str:
    st = user_state.setdefault(uid, UserState())

    name = st.profile.first_name
    gender = st.profile.gender

    # формы речи
    you_name = f"{name}" if name else "друг"
    if gender == "f":
        past = "передала"
    elif gender == "m":
        past = "передал"
    else:
        past = "передала"

    disclaim = kget("project.disclaimers.income", "Доход не гарантируется и зависит от усилий.")
    about_short = kget("project.about_short", "")

    return f"""
Вы — «{ASSISTANT_NAME}», ассистент куратора {OWNER_NAME} в онлайн-школе {PROJECT_NAME} и профессиональный менеджер по продажам (НЕ проговаривайте это клиенту).
Главная цель — помочь разобраться, подобрать курс/тариф и мягко довести до оформления заявки.

СТИЛЬ:
— Общение только на «Вы».
— Дружелюбно, тактично, живо. Без давления, без манипуляций.
— Обычно 1–6 коротких абзацев, списки уместны. Без «простыней».
— В конце обычно 1 уточняющий вопрос.

ИСТОЧНИК ЗНАНИЙ:
— Все факты о школе, курсах, тарифах, условиях, ценах, форматах, ссылках, медиа — ТОЛЬКО из knowledge.yaml.
— Если информации нет — НЕ выдумывайте. Скажите, что уточните у куратора {OWNER_NAME}, и предложите оформить заявку/оставить контакты.

ОГРАНИЧЕНИЯ:
— Не обещайте гарантированный доход. Формулировка: {disclaim}

КОНТЕКСТ:
— Клиента зовут: {you_name}
— Кратко о проекте (если нужно): {about_short}

ЕСЛИ КЛИЕНТ ОПРЕДЕЛИЛСЯ С КУРСОМ/ТАРИФОМ:
— Попросите: Фамилия Имя, Телефон, E-mail, выбранный курс/тариф.
— После получения: подтвердите и напишите «Спасибо! Я {past} заявку. Куратор {OWNER_NAME} свяжется с Вами и подскажет дальнейшие шаги.»
""".strip()


# =========================
# COMMANDS
# =========================
@dp.message(Command("myid"))
async def cmd_myid(message: Message):
    await message.answer(f"Your ID: {message.from_user.id}\nCurrent chat ID: {message.chat.id}")


@dp.message(Command("reload"))
async def cmd_reload(message: Message):
    global knowledge
    knowledge = load_knowledge()
    await message.answer("knowledge.yaml перечитала ✅")


# =========================
# START
# =========================
@dp.message(CommandStart())
async def start(message: Message):
    uid = message.from_user.id
    cleanup_states(time.time())

    st = user_state.setdefault(uid, UserState())
    st.last_seen = time.time()
    st.stage = Stage.ASK_NAME

    await message.answer(
        f"Здравствуйте! 😊\n\n"
        f"Я {ASSISTANT_NAME} — помощница куратора {OWNER_NAME} в онлайн-школе {PROJECT_NAME}.\n"
        f"Помогу подобрать курс и тариф под Вашу цель.\n\n"
        f"Как я могу к Вам обращаться?"
    )


# =========================
# PHOTO (чек)
# =========================
@dp.message(F.photo)
async def on_photo(message: Message):
    uid = message.from_user.id
    st = user_state.setdefault(uid, UserState())
    st.last_seen = time.time()

    if st.stage != Stage.WAIT_RECEIPT:
        await message.answer(
            "Вижу фото 🙂\n"
            "Если это чек — пожалуйста, сначала напишите, что хотите оформить покупку, и я подскажу дальнейшие шаги ✅"
        )
        return

    photo = message.photo[-1]
    st.pending_receipt_file_id = photo.file_id
    st.stage = Stage.CONFIRM_RECEIPT

    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Да, это чек", callback_data="receipt_yes")
    kb.button(text="❌ Нет, не чек", callback_data="receipt_no")
    kb.adjust(2)

    await message.answer("Подтвердите, пожалуйста: это чек об оплате? 🙂", reply_markup=kb.as_markup())


@dp.callback_query(F.data.in_(["receipt_yes", "receipt_no"]))
async def receipt_confirm(cb: CallbackQuery):
    uid = cb.from_user.id
    st = user_state.setdefault(uid, UserState())
    st.last_seen = time.time()
    await cb.answer()

    if cb.data == "receipt_no":
        st.pending_receipt_file_id = None
        st.stage = Stage.NORMAL
        await cb.message.answer("Хорошо 🙂 Тогда продолжим. Подскажите, пожалуйста, какая у Вас цель обучения?")
        return

    fid = st.pending_receipt_file_id
    st.pending_receipt_file_id = None
    st.stage = Stage.NORMAL

    await cb.message.answer("Спасибо ✅ Я передам это куратору Юлии, и она подтвердит оплату.")

    if ADMIN_CHAT_ID_INT:
        lead = (
            "✅ ПРИШЁЛ ЧЕК ОБ ОПЛАТЕ\n"
            f"ФИО: {(st.profile.first_name or '')} {(st.profile.last_name or '')}\n"
            f"Тариф: {st.chosen_tariff_title or 'не указан'} — {st.chosen_tariff_price or '—'} ₽\n"
            f"Телефон: {st.profile.phone or 'не указан'}\n"
            f"Email: {st.profile.email or 'не указан'}\n"
            f"User ID: {uid}"
        )
        await send_admin(lead)

        if fid:
            try:
                await bot.send_photo(ADMIN_CHAT_ID_INT, photo=fid, caption="Чек от клиента ✅")
            except Exception as e:
                log.exception("Failed to send receipt photo to admin: %s", e)


# =========================
# TEXT HANDLER
# =========================
@dp.message(F.text)
async def chat(message: Message):
    uid = message.from_user.id
    st = user_state.setdefault(uid, UserState())
    st.last_seen = time.time()
    cleanup_states(st.last_seen)

    text = (message.text or "").strip()
    if not text:
        return

    # 1) Сбор имени (важно: умеет доставать имя из длинной фразы)
    if st.stage == Stage.ASK_NAME:
        first, last = extract_name(text)
        if first:
            st.profile.first_name = first
            st.profile.last_name = last
            st.profile.gender = guess_gender_by_name(first)
            st.stage = Stage.QUALIFY

            await message.answer(
                f"Очень приятно, {first}! 😊\n\n"
                "Подскажите, пожалуйста, какая у Вас цель сейчас ближе:\n"
                "1) подработка\n"
                "2) новая профессия\n"
                "3) развитие в проекте\n\n"
                "Что выбираете?"
            )
        else:
            await message.answer("Подскажите, пожалуйста, как я могу к Вам обращаться? (Можно просто имя)")
        return

    # 2) Если клиент снова представился позже — обновим имя (чтобы не было повтора «как Вас зовут»)
    if not st.profile.first_name:
        first, last = extract_name(text)
        if first:
            st.profile.first_name = first
            st.profile.last_name = last
            st.profile.gender = guess_gender_by_name(first)
            st.stage = Stage.QUALIFY

    # 3) Гостевой доступ
    if is_guest_request(text):
        guest_key = kget("guest_access.key")
        if guest_key:
            await message.answer(
                "Конечно 🙂\n\n"
                f"🔑 Ваш гостевой ключ: `{guest_key}`\n\n"
                "Хотите — отправлю короткую инструкцию активации?"
                , parse_mode="Markdown"
            )
        else:
            await message.answer("Сейчас не вижу гостевой ключ в базе 🙈 Могу уточнить у куратора Юлии. Оставите контакт?")

        # медиа из guest_access.media_refs (если у Вас ключи совпадают)
        pres_key = kget("guest_access.media_refs.presentation")
        if pres_key:
            await message.answer("Отправляю презентацию по проекту, чтобы было проще сориентироваться ✅")
            await send_media_by_key(message, pres_key, caption_override="Презентация INSTART 📎")

        memo_key = kget("guest_access.media_refs.registration_memo_photo")
        if memo_key:
            await message.answer("И ещё памятка по регистрации ✅")
            await send_media_by_key(message, memo_key, caption_override="Памятка по регистрации ✅")

        instr_video = kget("guest_access.media_refs.registration_instruction_video")
        if instr_video:
            await message.answer("И видео-инструкция по активации ключа ✅")
            await send_media_by_key(message, instr_video, caption_override="Видео-инструкция ✅")

        return

    # 4) Вопрос про тарифы
    if is_tariff_question(text):
        await message.answer(
            "Вот актуальные тарифы и цены 🙂\n\n"
            f"{format_tariffs_brief()}\n\n"
            "Подскажите, пожалуйста, какая цель у Вас сейчас ближе?"
        )
        return

    # 5) Поиск курса/тарифа по aliases
    found = find_item_by_alias(text)
    if found:
        st.interest_id = found.get("id")
        st.interest_type = found.get("type")

        if found.get("type") == "tariff":
            title = found.get("title", "Тариф")
            price = found.get("price_rub")
            about = found.get("short_about") or ""
            who_for = found.get("who_for") or []

            msg = f"По запросу вижу **{title}**.\n"
            if price is not None:
                msg += f"Стоимость: **{price} ₽**\n"
            if about:
                msg += f"\n{about}\n"
            if who_for:
                msg += "\nКому подходит:\n" + "\n".join([f"• {x}" for x in who_for[:6]])

            msg += "\n\nПодскажите, пожалуйста, Вы рассматриваете обучение для подработки или как новую профессию?"
            await message.answer(msg, parse_mode="Markdown")
            return

        if found.get("type") == "course":
            title = found.get("title", "Курс")
            short_desc = found.get("short_description") or ""
            category = found.get("category") or ""
            chat_available = found.get("chat_available")

            msg = f"По запросу вижу курс **«{title}»**.\n"
            if category:
                msg += f"Категория: {category}\n"
            msg += f"{course_price_line(found)}\n"
            if chat_available is not None:
                msg += f"Чат поддержки: {'да' if chat_available else 'нет'}\n"
            if short_desc:
                msg += f"\n{short_desc}\n"

            # попробуем отправить макеты, если они есть
            media_block = found.get("media") or {}
            desc_mock = (media_block.get("description_mockup") or {})
            program_mock = (media_block.get("program_mockup") or {})

            await message.answer(msg, parse_mode="Markdown")

            # отправим медиа, если есть file_id
            if isinstance(desc_mock, dict) and desc_mock.get("file_id"):
                await message.answer("Отправляю описание курса (макет) ✅")
                await message.answer_photo(photo=desc_mock["file_id"], caption=(desc_mock.get("title") or "")[:1024])

            if isinstance(program_mock, dict) and program_mock.get("file_id"):
                await message.answer("И программу курса ✅")
                await message.answer_photo(photo=program_mock["file_id"], caption=(program_mock.get("title") or "")[:1024])

            await message.answer("Подскажите, пожалуйста, Вы новичок в этом направлении или уже был опыт?")
            return

    # 6) Готов купить → сбор данных
    if BUY_INTENT_RE.search(text):
        st.stage = Stage.BUY_COLLECT
        await message.answer(
            "Хорошо 🙂 Чтобы оформить заявку, напишите одним сообщением:\n"
            "• Фамилия Имя\n"
            "• Телефон\n"
            "• E-mail\n"
            "• Выбранный курс или тариф (название)"
        )
        return

    if st.stage == Stage.BUY_COLLECT:
        first, last = extract_name(text)
        if first:
            st.profile.first_name = st.profile.first_name or first
            st.profile.gender = st.profile.gender or guess_gender_by_name(first)
        if last:
            st.profile.last_name = st.profile.last_name or last

        phone = extract_phone(text)
        email = extract_email(text)
        if phone:
            st.profile.phone = normalize_phone(phone)
        if email:
            st.profile.email = email.strip()

        picked = find_item_by_alias(text)
        if picked and picked.get("type") == "tariff":
            st.chosen_tariff_title = picked.get("title")
            st.chosen_tariff_price = picked.get("price_rub")

        missing = []
        if not st.profile.first_name or not st.profile.last_name:
            missing.append("Фамилия Имя")
        if not st.profile.phone or len(re.sub(r"\D", "", st.profile.phone)) < 10:
            missing.append("телефон")
        if not st.profile.email or not looks_like_email(st.profile.email):
            missing.append("e-mail")
        if not st.chosen_tariff_title:
            missing.append("выбранный курс/тариф")

        if missing:
            await message.answer("Мне не хватает: " + ", ".join(missing) + " 🙂\nНапишите, пожалуйста, одним сообщением.")
            return

        # сформировать заявку (и отправить админу, если задан)
        gender = st.profile.gender or ""
        lead_text = (
            "🟩 ЗАЯВКА (INSTART)\n"
            f"Имя клиента: {st.profile.first_name}\n"
            f"Пол: {gender}\n"
            f"Фамилия Имя: {st.profile.first_name} {st.profile.last_name}\n"
            f"Телефон: {st.profile.phone}\n"
            f"Email: {st.profile.email}\n"
            f"Курс/Тариф: {st.chosen_tariff_title} — {st.chosen_tariff_price} ₽\n"
            f"Источник: Telegram\n"
            f"User ID: {uid}\n"
            f"Дата/время: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await send_admin(lead_text)

        # клиенту
        past = "передала" if st.profile.gender != "m" else "передал"
        await message.answer(
            f"Спасибо! 😊 Я {past} заявку.\n"
            f"Куратор {OWNER_NAME} свяжется с Вами и подскажет дальнейшие шаги."
        )
        st.stage = Stage.NORMAL
        return

    # =========================
    # OpenAI fallback (если YAML не покрыл вопрос)
    # =========================
    add_history(uid, "user", text)

    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(typing_loop(message.chat.id, stop_event))
    start_ts = time.time()

    def call_openai_sync(messages: List[dict]) -> str:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=260,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content

    try:
        sys_prompt = build_system_prompt(uid)
        msgs = [{"role": "system", "content": sys_prompt}]
        msgs.extend(st.history[-HISTORY_MAX_TURNS * 2 :])
        msgs.append({"role": "user", "content": text})

        answer = await asyncio.to_thread(call_openai_sync, msgs)

        elapsed = time.time() - start_ts
        if elapsed < 2.5:
            await asyncio.sleep(2.5 - elapsed)

        parts = split_answer(answer, max_chars=900)
        if not parts:
            parts = ["Я немного задумалась 🙈 Переформулируйте, пожалуйста, вопрос — и я помогу."]

        for p in parts:
            await message.answer(p)

        add_history(uid, "assistant", answer)
        st.stage = Stage.NORMAL

    except Exception as e:
        log.exception("OpenAI error: %s", e)
        await message.answer("⚠️ Сейчас сервис ответов перегружен. Попробуйте, пожалуйста, через минуту 🙂")
    finally:
        stop_event.set()
        try:
            await typing_task
        except Exception:
            pass


# =========================
# WEBHOOK
# =========================
async def on_startup(app: web.Application):
    # очищаем старый webhook и ставим новый
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        pass

    await bot.set_webhook(
        url=f"{WEBHOOK_BASE}{WEBHOOK_PATH}",
        secret_token=WEBHOOK_SECRET,
    )
    log.info("Webhook set: %s%s", WEBHOOK_BASE, WEBHOOK_PATH)


async def on_shutdown(app: web.Application):
    try:
        await bot.delete_webhook()
    except Exception:
        pass
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
