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

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")              # https://....up.railway.app
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/tg/webhook")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")            # chat_id Юлии
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

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# KNOWLEDGE LOADER
# =========================
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge.yaml")

def load_knowledge() -> dict:
    """
    Поддерживаем 2 формата:
    1) dict (mapping) в корне
    2) list (ваш формат: - id: ..., - id: ...)
       -> оборачиваем в {"items": [...]}
    """
    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        log.warning("knowledge.yaml не найден рядом с main.py")
        return {}
    except Exception as e:
        log.exception("Ошибка чтения knowledge.yaml: %s", e)
        return {}

    if data is None:
        return {}

    if isinstance(data, list):
        return {"items": data}

    if isinstance(data, dict):
        return data

    log.warning("knowledge.yaml имеет неожиданный тип: %s", type(data))
    return {}

knowledge: Dict[str, Any] = load_knowledge()


def kget(path: str, default=None):
    cur: Any = knowledge
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    return s


def knowledge_items() -> List[dict]:
    """
    Если в корне список — он лежит в knowledge["items"].
    Если у вас dict-структура — можете тоже хранить items: [...]
    """
    items = knowledge.get("items")
    return items if isinstance(items, list) else []


# Индекс по алиасам/тайтлам для быстрого поиска
ALIAS_INDEX: Dict[str, List[dict]] = {}


def rebuild_index() -> None:
    global ALIAS_INDEX
    ALIAS_INDEX = {}
    for it in knowledge_items():
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or "").strip()
        aliases = it.get("aliases") or []
        keys = set()

        if title:
            keys.add(normalize_text(title))
        if isinstance(aliases, list):
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    keys.add(normalize_text(a))

        # доп. ключи по id
        if it.get("id"):
            keys.add(normalize_text(str(it["id"])))

        for k in keys:
            ALIAS_INDEX.setdefault(k, []).append(it)


rebuild_index()


def find_items_by_query(text: str, types: Optional[List[str]] = None) -> List[dict]:
    """
    Ищем по:
    - точному совпадению алиаса/тайтла
    - вхождению алиаса в запрос (если алиас >= 4 символов)
    """
    q = normalize_text(text)
    if not q:
        return []

    results: List[dict] = []

    # 1) точное совпадение
    if q in ALIAS_INDEX:
        results.extend(ALIAS_INDEX[q])

    # 2) поиск по вхождению алиаса в запрос
    for k, items in ALIAS_INDEX.items():
        if len(k) >= 4 and k in q:
            results.extend(items)

    # уникализация по id
    seen = set()
    uniq = []
    for it in results:
        _id = it.get("id") or id(it)
        if _id in seen:
            continue
        seen.add(_id)
        uniq.append(it)

    if types:
        types_norm = {t.lower() for t in types}
        uniq = [x for x in uniq if str(x.get("type", "")).lower() in types_norm]

    return uniq


def find_one_item(text: str, types: Optional[List[str]] = None) -> Optional[dict]:
    items = find_items_by_query(text, types=types)
    return items[0] if items else None


# =========================
# PROJECT META (если есть в YAML-словаре)
# =========================
ASSISTANT_NAME = kget("assistant.name", "Лиза")
OWNER_NAME = kget("assistant.owner_name", "Юлия")
PROJECT_NAME = kget("project.name", "INSTART")


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
    sex: Optional[str] = None  # "m" / "f" / None
    phone: Optional[str] = None
    email: Optional[str] = None

@dataclass
class UserState:
    stage: str = Stage.ASK_NAME
    chosen_item_id: Optional[str] = None
    chosen_item_title: Optional[str] = None
    chosen_item_price: Optional[int] = None
    last_seen: float = field(default_factory=lambda: time.time())
    history: List[dict] = field(default_factory=list)  # [{"role":"user","content":...}]
    profile: UserProfile = field(default_factory=UserProfile)
    pending_receipt_file_id: Optional[str] = None
    sent_media_file_ids: set = field(default_factory=set)  # чтобы не слать повторно

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
def extract_name(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Достаёт имя даже из длинной фразы:
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
    if len(words) == 1 and len(text.strip().split()) <= 3:
        return words[0], None
    if len(words) >= 2 and len(text.strip().split()) <= 4:
        return words[0], words[1]

    return None, None


def guess_sex_by_name(name: str) -> Optional[str]:
    n = normalize_text(name)
    if not n:
        return None
    # супер-лёгкая эвристика
    if n.endswith(("а", "я")) and n not in {"илья", "никита"}:
        return "f"
    return "m"


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


BUY_INTENT_RE = re.compile(r"\b(купить|оплат(ить|а)|готов(а)? купить|беру|хочу оформить|оформим)\b", re.IGNORECASE)

def is_guest_request(text: str) -> bool:
    t = normalize_text(text)
    return any(w in t for w in ["гост", "демо", "пробн", "ключ"])

def is_presentation_request(text: str) -> bool:
    t = normalize_text(text)
    return "презентац" in t

def is_price_question(text: str) -> bool:
    t = normalize_text(text)
    return any(w in t for w in ["цена", "стоим", "сколько"])


# =========================
# HELPERS: media send (из карточки)
# =========================
async def send_item_media(message: Message, st: UserState, item: dict, caption: Optional[str] = None) -> bool:
    media = item.get("media")
    if not isinstance(media, dict):
        return False

    mtype = media.get("type")
    fid = media.get("file_id")
    if not fid:
        return False

    # не слать повторно
    if fid in st.sent_media_file_ids:
        return False

    cap = caption or media.get("caption") or media.get("title") or ""
    cap = cap[:1024] if cap else None

    if mtype == "photo":
        await message.answer_photo(photo=fid, caption=cap)
    elif mtype == "video":
        await message.answer_video(video=fid, caption=cap)
    elif mtype == "document":
        await message.answer_document(document=fid, caption=cap)
    else:
        return False

    st.sent_media_file_ids.add(fid)
    return True


# =========================
# HELPERS: admin
# =========================
async def send_admin(text: str) -> None:
    try:
        await bot.send_message(ADMIN_CHAT_ID_INT, text)
    except Exception as e:
        log.exception("Failed to send admin message: %s", e)


# =========================
# HELPERS: typing
# =========================
async def typing_loop(chat_id: int, stop_event: asyncio.Event) -> None:
    try:
        while not stop_event.is_set():
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4)
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
# PROMPT (ваш)
# =========================
def build_system_prompt(uid: int) -> str:
    st = user_state.setdefault(uid, UserState())
    name = st.profile.first_name or "пожалуйста"

    # Не выдумываем факты — только из knowledge.yaml
    # Если у вас есть project.disclaimers.income — используем, иначе дефолт:
    disclaim = kget("project.disclaimers.income", "Доход не гарантируется и зависит от усилий.")

    return f"""
Вы — “{ASSISTANT_NAME}”, ассистент куратора {OWNER_NAME} в онлайн-школе {PROJECT_NAME}.

ВАЖНО:
- Общение только на «Вы».
- Все факты о школе, курсах, тарифах, цене, бонусах, ссылках и медиа — ТОЛЬКО из knowledge.yaml.
- Если в knowledge.yaml нет нужной информации — не выдумывайте: предложите оставить контакт для уточнения у куратора.
- Не обещайте гарантированный доход. Формулировка: {disclaim}

СТИЛЬ:
- Дружелюбно, живо, без канцелярита, без давления.
- Обычно 1–6 коротких абзацев.
- В конце задавайте 1 уточняющий вопрос.

ПАМЯТЬ:
- Используйте историю чата. Не повторяйте одни и те же вопросы.
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
    rebuild_index()
    await message.answer("knowledge.yaml перечитан ✅")


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
        "Помогу подобрать курс и тариф под Вашу цель.\n\n"
        "Как я могу к Вам обращаться?"
    )


# =========================
# PHOTO: чек только после оплаты
# =========================
@dp.message(F.photo)
async def on_photo(message: Message):
    uid = message.from_user.id
    st = user_state.setdefault(uid, UserState())
    st.last_seen = time.time()

    if st.stage != Stage.WAIT_RECEIPT:
        await message.answer(
            "Вижу фото 🙂\n"
            "Если это чек — сначала выберите курс/тариф, я оформлю заявку и подскажу дальнейшие шаги ✅"
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
        await cb.message.answer("Хорошо 🙂 Тогда продолжим. Подскажите, что именно хотите выбрать — курс или тариф?")
        return

    fid = st.pending_receipt_file_id
    st.pending_receipt_file_id = None
    st.stage = Stage.NORMAL

    await cb.message.answer("Принято ✅ Я передам Юлии на подтверждение оплаты.")

    lead = (
        "✅ ПРИШЁЛ ЧЕК ОБ ОПЛАТЕ\n"
        f"Имя: {st.profile.first_name or '—'} {st.profile.last_name or ''}\n"
        f"Выбор: {st.chosen_item_title or 'не указан'} — {st.chosen_item_price or '—'} ₽\n"
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
# MAIN TEXT HANDLER
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

    # 1) стадия "спросили имя"
    if st.stage == Stage.ASK_NAME:
        first, last = extract_name(text)
        if first:
            st.profile.first_name = first
            st.profile.last_name = last
            st.profile.sex = guess_sex_by_name(first)
            st.stage = Stage.QUALIFY
            await message.answer(
                f"{first}, очень приятно познакомиться! 😊\n\n"
                "Подскажите, пожалуйста, что Вам сейчас ближе?\n"
                "1) Подработка\n"
                "2) Новая онлайн-профессия\n"
                "3) Развитие в проекте (партнёрство/кураторство)\n\n"
                "Можно просто цифрой."
            )
        else:
            await message.answer("Как я могу к Вам обращаться? (Можно просто имя) 🙂")
        return

    # 2) Гостевой доступ: ищем либо блок guest_access в dict, либо карточку в items
    if is_guest_request(text):
        # вариант А: структурный блок
        guest_key = kget("guest_access.key")
        guest_site = kget("guest_access.site") or kget("guest_access.site_url") or kget("guest_access.url")

        # вариант Б: карточка
        guest_item = find_one_item(text, types=["guest_access", "info", "guest"])

        if guest_key or guest_item:
            lines = ["Конечно 🙂"]

            if guest_site:
                lines.append(f"\nСайт для регистрации: {guest_site}")

            if guest_key:
                lines.append(f"\n🔑 Гостевой ключ: `{guest_key}`")

            if guest_item and isinstance(guest_item.get("description"), str):
                lines.append("\n" + guest_item["description"].strip())

            await message.answer("\n".join(lines), parse_mode="Markdown")

            # если у карточки есть медиа — отправим
            if guest_item:
                await send_item_media(message, st, guest_item, caption="Отправляю материалы по гостевому доступу ✅")

        else:
            await message.answer(
                "Я не вижу данных по гостевому доступу в базе 🙈\n"
                "Могу уточнить у Юлии и вернуться к Вам. Подскажите, пожалуйста, как удобнее — телефон или email?"
            )
        return

    # 3) Презентация проекта: ищем карточку по алиасу/слову "презентация"
    if is_presentation_request(text):
        pres_item = find_one_item(text, types=["media", "presentation", "info", "project_media"])
        if not pres_item:
            # если типы не совпали — попробуем просто любой item, где есть media и "презентация" в title/aliases
            candidates = find_items_by_query(text)
            pres_item = next((x for x in candidates if isinstance(x.get("media"), dict)), None)

        if pres_item:
            await message.answer("Сейчас отправлю презентацию проекта 📎")
            ok = await send_item_media(message, st, pres_item, caption="Презентация проекта INSTART 📎")
            if not ok:
                await message.answer("Похоже, я уже отправляла этот файл ранее 🙂 Если нужно — напомню кратко, что в презентации.")
        else:
            await message.answer(
                "Я не нашла презентацию в базе 🙈\n"
                "Скажите, пожалуйста, что именно хотите узнать про INSTART: подработка, профессия или партнёрство?"
            )
        return

    # 4) Если человек спрашивает про конкретный курс/тариф — отвечаем без OpenAI
    found = find_one_item(text, types=["course", "tariff"])
    if found:
        title = found.get("title", "Без названия")
        typ = str(found.get("type", "")).lower()
        price = found.get("price") or {}

        # поддержка разных форматов цены
        price_with = None
        price_without = None
        if isinstance(price, dict):
            price_with = price.get("with_chat_rub") or price.get("with_chat")
            price_without = price.get("without_chat_rub") or price.get("without_chat")

        chat_available = found.get("chat_available")
        short_desc = found.get("short_description") or found.get("description")

        lines = []
        if typ == "tariff":
            lines.append(f"**Тариф:** {title}")
        else:
            lines.append(f"**Курс:** {title}")

        if price_with or price_without:
            if price_with and price_without and price_with != price_without:
                lines.append(f"Цена: с чатом — {price_with} ₽, без чата — {price_without} ₽.")
            elif price_with:
                lines.append(f"Цена: {price_with} ₽.")
            elif price_without:
                lines.append(f"Цена: {price_without} ₽.")

        if isinstance(chat_available, bool):
            lines.append("Чат: " + ("есть ✅" if chat_available else "нет"))

        if isinstance(short_desc, str) and short_desc.strip():
            lines.append("\n" + short_desc.strip())

        await message.answer("\n".join(lines), parse_mode="Markdown")

        # отправим макет/медиа если есть
        sent = await send_item_media(message, st, found, caption=f"Отправляю материалы по «{title}» 📎")
        if not sent and isinstance(found.get("media"), dict):
            # если не отправили, значит уже отправляли — просто скажем
            await message.answer("Материалы по этому варианту я уже отправляла ранее 🙂")

        await message.answer("Подскажите, пожалуйста: Вы рассматриваете этот вариант для себя или хотите сравнить с ещё 1–2 вариантами?")
        st.stage = Stage.NORMAL
        return

    # 5) Если вопрос о цене вообще — предложим уточнить что именно
    if is_price_question(text):
        await message.answer(
            "Подскажите, пожалуйста, цену чего именно хотите узнать — конкретного курса или тарифа?\n"
            "Напишите название (или как Вы его называете) — я найду по базе 🙂"
        )
        return

    # 6) Готов купить → сбор данных (только когда выбрали конкретный курс/тариф)
    if BUY_INTENT_RE.search(text):
        st.stage = Stage.BUY_COLLECT
        await message.answer(
            "Хорошо 🙂 Чтобы оформить заявку, напишите одним сообщением:\n"
            "1) Фамилия Имя\n"
            "2) Телефон\n"
            "3) E-mail\n"
            "4) Какой курс/тариф выбрали (название)\n\n"
            "Если ещё не выбрали — скажите цель, и я предложу 1–3 варианта."
        )
        return

    if st.stage == Stage.BUY_COLLECT:
        first, last = extract_name(text)
        if first:
            st.profile.first_name = st.profile.first_name or first
            st.profile.sex = st.profile.sex or guess_sex_by_name(first)
        if last:
            st.profile.last_name = st.profile.last_name or last

        phone = extract_phone(text)
        email = extract_email(text)
        if phone:
            st.profile.phone = normalize_phone(phone)
        if email:
            st.profile.email = email.strip()

        chosen = find_one_item(text, types=["course", "tariff"])
        if chosen:
            st.chosen_item_id = chosen.get("id")
            st.chosen_item_title = chosen.get("title")
            # цена: берём "with_chat_rub" если есть, иначе "without_chat_rub"
            price = chosen.get("price") if isinstance(chosen.get("price"), dict) else {}
            if isinstance(price, dict):
                st.chosen_item_price = price.get("with_chat_rub") or price.get("without_chat_rub")

        if not st.chosen_item_title:
            await message.answer("Пожалуйста, уточните выбранный курс/тариф (название) — я зафиксирую в заявке 🙂")
            return

        missing = []
        if not st.profile.last_name or not st.profile.first_name:
            missing.append("Фамилия Имя")
        if not st.profile.phone or len(re.sub(r"\D", "", st.profile.phone)) < 10:
            missing.append("телефон")
        if not st.profile.email or not looks_like_email(st.profile.email):
            missing.append("email")

        if missing:
            await message.answer("Мне не хватает: " + ", ".join(missing) + " 🙂 Напишите, пожалуйста.")
            return

        # Заявка админу
        now_str = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        lead_text = (
            "🟩 ЗАЯВКА НА ПОКУПКУ (INSTART)\n"
            f"Имя клиента: {st.profile.first_name}\n"
            f"Пол: {st.profile.sex or 'не определён'}\n"
            f"Фамилия Имя: {st.profile.last_name} {st.profile.first_name}\n"
            f"Телефон: {st.profile.phone}\n"
            f"Email: {st.profile.email}\n"
            f"Курс/Тариф: {st.chosen_item_title} — {st.chosen_item_price or '—'} ₽\n"
            f"Источник: Telegram\n"
            f"Краткий запрос: {text[:200]}\n"
            f"Дата/время: {now_str}\n"
            f"User ID: {uid}"
        )
        await send_admin(lead_text)

        await message.answer(
            "Спасибо! 😊 Я передала заявку.\n"
            "Куратор Юлия свяжется с Вами и подскажет дальнейшие шаги."
        )

        st.stage = Stage.NORMAL
        return

    # =========================
    # OpenAI fallback (когда не нашли ничего в базе)
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
            max_tokens=240,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        sys = build_system_prompt(uid)
        msgs = [{"role": "system", "content": sys}]
        msgs.extend(st.history[-HISTORY_MAX_TURNS * 2 :])
        msgs.append({"role": "user", "content": text})

        answer = await asyncio.to_thread(call_openai_sync, msgs)

        elapsed = time.time() - start_ts
        if elapsed < 2.0:
            await asyncio.sleep(2.0 - elapsed)

        parts = split_answer(answer, max_chars=900)
        if not parts:
            parts = ["Я задумалась 😅 Напишите, пожалуйста, чуть иначе — и я помогу."]

        for p in parts:
            await message.answer(p)

        add_history(uid, "assistant", answer)
        st.stage = Stage.NORMAL

    except Exception as e:
        log.exception("OpenAI error: %s", e)
        await message.answer("⚠️ Сейчас я немного перегружена. Попробуйте, пожалуйста, через минуту 🙂")

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
