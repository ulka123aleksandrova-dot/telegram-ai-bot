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

ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")            # chat_id (куда слать лиды/чеки)
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
# KNOWLEDGE (ROBUST)
# =========================
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge.yaml")

def load_knowledge() -> dict:
    """
    Делает загрузку максимально «неубиваемой», чтобы бот не падал.
    Поддерживает:
      - YAML как dict (рекомендуемый)
      - YAML как list (например список сущностей - id/type/title/aliases)
    """
    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        log.exception("knowledge.yaml не найден рядом с main.py")
        return {}
    except Exception:
        log.exception("Ошибка чтения knowledge.yaml")
        return {}

    if raw is None:
        return {}

    if isinstance(raw, dict):
        return raw

    # если корень — список сущностей
    if isinstance(raw, list):
        return {"items": raw}

    log.error("knowledge.yaml: корневой тип должен быть dict или list")
    return {}

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

# ---- build entity indexes from YAML (courses/tariffs/etc) ----
def _iter_entities() -> List[dict]:
    entities: List[dict] = []

    # 1) items (если YAML корнем был список)
    items = kget("items", [])
    if isinstance(items, list):
        entities.extend([x for x in items if isinstance(x, dict)])

    # 2) entities / courses / tariffs (если YAML структурный)
    for key in ("entities", "courses", "tariffs"):
        arr = kget(key, [])
        if isinstance(arr, list):
            entities.extend([x for x in arr if isinstance(x, dict)])

    # 3) иногда у вас могли быть секции типа top_up_system и т.п.
    # их тоже можно использовать, но они не сущности курса/тарифа.
    return entities

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

@dataclass
class EntityIndex:
    by_id: Dict[str, dict] = field(default_factory=dict)
    alias_map: Dict[str, List[dict]] = field(default_factory=dict)  # alias -> list of entities

def build_index() -> EntityIndex:
    idx = EntityIndex()
    for e in _iter_entities():
        eid = str(e.get("id", "")).strip()
        if eid:
            idx.by_id[eid] = e

        # aliases
        aliases = e.get("aliases", []) or []
        if isinstance(aliases, str):
            aliases = [aliases]

        # title тоже считаем алиасом
        title = e.get("title")
        if title:
            aliases = list(aliases) + [title]

        for a in aliases:
            if not isinstance(a, str):
                continue
            key = _normalize(a)
            if not key:
                continue
            idx.alias_map.setdefault(key, []).append(e)
    return idx

INDEX = build_index()

def reload_knowledge() -> None:
    global knowledge, INDEX
    knowledge = load_knowledge()
    INDEX = build_index()

def find_entity(text: str, entity_type: Optional[str] = None) -> Optional[dict]:
    """
    Поиск по aliases/title (строгий матч).
    entity_type: "course" / "tariff" / None
    """
    q = _normalize(text)
    if not q:
        return None
    candidates = INDEX.alias_map.get(q, [])
    if entity_type:
        candidates = [c for c in candidates if str(c.get("type", "")).lower() == entity_type.lower()]
    return candidates[0] if candidates else None

def find_tariff(text: str) -> Optional[dict]:
    """
    Ищем тариф по:
      - алиасам
      - "тариф 1/2/3..." (если так написали)
    """
    by_alias = find_entity(text, "tariff")
    if by_alias:
        return by_alias

    q = _normalize(text)
    m = re.search(r"\bтариф\s*(\d)\b", q)
    if m:
        num = int(m.group(1))
        # попробуем найти в сущностях тариф с таким номером в title/aliases
        for k, ents in INDEX.alias_map.items():
            if k in (f"тариф {num}", f"тариф{num}"):
                for e in ents:
                    if str(e.get("type", "")).lower() == "tariff":
                        return e
    return None

def tariffs_list() -> List[dict]:
    t = kget("tariffs", [])
    if isinstance(t, list):
        return [x for x in t if isinstance(x, dict)]
    # если тарифы описаны сущностями в items
    out = [e for e in _iter_entities() if str(e.get("type", "")).lower() == "tariff"]
    return out

def _tariff_price_rub(t: dict) -> Optional[int]:
    # поддержка разных форматов цены
    if isinstance(t.get("price_rub"), int):
        return t["price_rub"]
    price = t.get("price")
    if isinstance(price, dict):
        # если у тарифов вдруг price.without_chat_rub / with_chat_rub
        for k in ("price_rub", "without_chat_rub", "with_chat_rub"):
            v = price.get(k)
            if isinstance(v, int):
                return v
    return None

def tariffs_brief() -> str:
    arr = tariffs_list()
    # сортировка по "тариф N" если есть
    def keyf(x: dict) -> int:
        title = str(x.get("title", "")).lower()
        m = re.search(r"\bтариф\s*(\d)\b", title)
        return int(m.group(1)) if m else 999
    arr = sorted(arr, key=keyf)

    lines = []
    for t in arr:
        title = t.get("title")
        price = _tariff_price_rub(t)
        if title and price is not None:
            lines.append(f"• {title} — {price} ₽")
        elif title:
            lines.append(f"• {title}")
    return "\n".join(lines) if lines else "Пока не вижу тарифы в базе."

def media_get(key: str) -> Optional[dict]:
    media = kget("media", {})
    if isinstance(media, dict):
        m = media.get(key)
        if isinstance(m, dict):
            return m
    return None

async def send_media_by_key(message: Message, key: str, caption_override: Optional[str] = None) -> bool:
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
        return True
    if mtype == "video":
        await message.answer_video(video=fid, caption=caption[:1024] if caption else None)
        return True
    if mtype == "document":
        await message.answer_document(document=fid, caption=caption[:1024] if caption else None)
        return True
    return False

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
    phone: Optional[str] = None
    email: Optional[str] = None

@dataclass
class UserState:
    stage: str = Stage.NORMAL
    chosen_tariff_title: Optional[str] = None
    chosen_tariff_price: Optional[int] = None
    last_seen: float = field(default_factory=lambda: time.time())
    history: List[dict] = field(default_factory=list)
    profile: UserProfile = field(default_factory=UserProfile)
    pending_receipt_file_id: Optional[str] = None

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
NAME_RE = re.compile(r"^(?:меня зовут|я)\s+([A-Za-zА-Яа-яЁё\-]+)(?:\s+([A-Za-zА-Яа-яЁё\-]+))?$", re.IGNORECASE)
TWO_WORDS_RE = re.compile(r"^([A-Za-zА-Яа-яЁё\-]{2,})\s+([A-Za-zА-Яа-яЁё\-]{2,})$")

def extract_name(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = (text or "").strip()
    m = NAME_RE.match(t)
    if m:
        return m.group(1), m.group(2)
    m2 = TWO_WORDS_RE.match(t)
    if m2:
        return m2.group(1), m2.group(2)
    if re.fullmatch(r"[A-Za-zА-Яа-яЁё\-]{2,}", t):
        return t, None
    return None, None

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

BUY_INTENT_RE = re.compile(r"\b(купить|оплат(ить|а)|готов(а)? купить|беру тариф|хочу тариф|оформим)\b", re.IGNORECASE)

def is_guest_request(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in ["гост", "демо", "пробн", "ключ"])

def is_presentation_request(text: str) -> bool:
    t = (text or "").lower()
    return "презентац" in t

def is_tariff_question(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in ["тариф", "цена", "стоим", "сколько"])

GREET_RE = re.compile(r"^(привет|здравствуйте|здравствуй|хай|добрый день|добрый вечер|доброе утро)\b", re.IGNORECASE)

# =========================
# HELPERS: admin
# =========================
async def send_admin(text: str) -> None:
    try:
        await bot.send_message(ADMIN_CHAT_ID_INT, text)
        log.info("Admin notified OK")
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

def split_answer(text: str, max_chars: int = 850) -> List[str]:
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
# SYSTEM PROMPT
# =========================
def build_system_prompt(uid: int) -> str:
    st = user_state.setdefault(uid, UserState())
    name = st.profile.first_name or "друг"

    disclaim = kget("project.disclaimers.income", "Доход не гарантируется и зависит от усилий.")
    pay_phone = kget("instructions.payment.phone", "89883873424")
    pay_bank = kget("instructions.payment.bank", "Кубань Кредит")
    guest_key = kget("guest_access.key", "")

    return f"""
Ты — {ASSISTANT_NAME}, помощница куратора {OWNER_NAME} в онлайн-школе {PROJECT_NAME}.
Обращайся по имени: {name}. Стиль: живо, тепло, чуть эмодзи 🙂. Без воды.

ВАЖНО:
— Не выдумывай цены/состав тарифов/курсов. Используй только knowledge.yaml.
— Если данных нет: скажи честно и предложи гостевой доступ или оставить контакты.
— Не обещай гарантированный доход. Формулировка: {disclaim}

ФОРМАТ:
— 2–6 предложений, иногда 1–3 пункта.
— В конце: 1 вопрос (следующий шаг).

ТАРИФЫ:
{tariffs_brief()}

ГОСТЕВОЙ КЛЮЧ:
{guest_key}

ЕСЛИ КЛИЕНТ ГОТОВ КУПИТЬ:
— собрать имя/фамилию/телефон/email + выбранный тариф
— затем реквизиты: {pay_phone} (банк {pay_bank})
— попросить чек и подтвердить.
""".strip()

# =========================
# COMMANDS
# =========================
@dp.message(Command("myid"))
async def cmd_myid(message: Message):
    await message.answer(f"Your ID: {message.from_user.id}\nCurrent chat ID: {message.chat.id}")

@dp.message(Command("pingadmin"))
async def cmd_pingadmin(message: Message):
    await send_admin("✅ Тест: бот может писать админу.")
    await message.answer("Ок 🙂 Я отправила тест админу.")

@dp.message(Command("reload"))
async def cmd_reload(message: Message):
    reload_knowledge()
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
        f"Привет! 😊\n\n"
        f"Я {ASSISTANT_NAME} — помощница куратора {OWNER_NAME} в онлайн-школе {PROJECT_NAME}.\n"
        "Очень рада знакомству 🌿\n\n"
        "Как тебя зовут?"
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
            "Если это чек — сначала напиши «хочу купить», я дам реквизиты, и после оплаты пришлёшь чек сюда ✅"
        )
        return

    photo = message.photo[-1]
    st.pending_receipt_file_id = photo.file_id
    st.stage = Stage.CONFIRM_RECEIPT

    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Да, это чек", callback_data="receipt_yes")
    kb.button(text="❌ Нет, не чек", callback_data="receipt_no")
    kb.adjust(2)

    await message.answer("Подтверди, пожалуйста: это чек об оплате? 🙂", reply_markup=kb.as_markup())

@dp.callback_query(F.data.in_(["receipt_yes", "receipt_no"]))
async def receipt_confirm(cb: CallbackQuery):
    uid = cb.from_user.id
    st = user_state.setdefault(uid, UserState())
    st.last_seen = time.time()
    await cb.answer()

    if cb.data == "receipt_no":
        st.pending_receipt_file_id = None
        st.stage = Stage.NORMAL
        await cb.message.answer("Ок 🙂 Тогда продолжим. Хочешь — подберу тариф под твою цель?")
        return

    fid = st.pending_receipt_file_id
    st.pending_receipt_file_id = None
    st.stage = Stage.NORMAL

    await cb.message.answer("Принято ✅ Я передам админу, и он подтвердит оплату.")

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

    # Если человек написал "привет" без /start — не молчим
    if GREET_RE.search(text) and not st.profile.first_name:
        st.stage = Stage.ASK_NAME
        await message.answer(
            f"Привет 😊 Я {ASSISTANT_NAME}, помощница куратора {OWNER_NAME} в {PROJECT_NAME}.\n"
            "Как тебя зовут?"
        )
        return

    # 1) Имя
    if st.stage == Stage.ASK_NAME:
        first, last = extract_name(text)
        if first:
            st.profile.first_name = first
            st.profile.last_name = last
            st.stage = Stage.QUALIFY
            await message.answer(
                f"{first}, очень приятно познакомиться! 😊\n\n"
                "Чтобы подсказать лучший старт, подскажи:\n"
                "1) цель: подработка / новая профессия / куратор?\n"
                "2) сколько времени в неделю готов(а) уделять?"
            )
        else:
            await message.answer("Супер 🙂 Как тебя зовут? (Можно просто имя)")
        return

    # 2) Гостевой доступ
    if is_guest_request(text):
        guest_key = kget("guest_access.key")
        if guest_key:
            await message.answer(
                "Конечно 🙂\n\n"
                f"🔑 Гостевой ключ: `{guest_key}`\n\n"
                "Хочешь — пришлю короткую инструкцию активации ✅",
                parse_mode="Markdown",
            )
        else:
            await message.answer("Похоже, гостевой ключ не заполнен в базе 🙈 Могу уточнить у куратора — оставить контакт?")

        # (опционально) медиа-инструкции, если в YAML есть такие ключи
        memo_key = kget("guest_access.media_refs.registration_memo_photo")
        if memo_key:
            await send_media_by_key(message, memo_key, caption_override="Памятка по регистрации ✅")
        instr_video = kget("guest_access.media_refs.registration_instruction_video")
        if instr_video:
            await send_media_by_key(message, instr_video, caption_override="Видео-инструкция ✅")
        return

    # 3) Презентация
    if is_presentation_request(text):
        # здесь ключ должен соответствовать твоему knowledge.yaml -> media: { <key>: {type,file_id...}}
        # если у тебя другой ключ — просто поменяй строку ниже
        pres_key = "презентация_проекта"
        ok = await send_media_by_key(message, pres_key, caption_override="Презентация INSTART 📎")
        if not ok:
            await message.answer("Сейчас не вижу презентацию в базе 🙈 Напиши цель — и я подберу вариант без презентации.")
        return

    # 4) Вопросы по тарифам
    if is_tariff_question(text):
        await message.answer(
            "Вот актуальные тарифы и цены 🙂\n\n"
            f"{tariffs_brief()}\n\n"
            "Какая цель у тебя сейчас ближе: подработка, профессия или кураторство?"
        )
        return

    # 5) Готов купить
    if BUY_INTENT_RE.search(text):
        st.stage = Stage.BUY_COLLECT
        await message.answer(
            "Отлично 🙂 Давай оформим.\n\n"
            "Напиши одним сообщением:\n"
            "• Имя и фамилия\n"
            "• Телефон\n"
            "• Email\n"
            "• Выбранный тариф (название или «тариф 1/2/3…»)"
        )
        return

    # 6) Сбор данных на покупку
    if st.stage == Stage.BUY_COLLECT:
        first, last = extract_name(text)
        if first:
            st.profile.first_name = st.profile.first_name or first
        if last:
            st.profile.last_name = st.profile.last_name or last

        phone = extract_phone(text)
        email = extract_email(text)
        if phone:
            st.profile.phone = normalize_phone(phone)
        if email:
            st.profile.email = email.strip()

        t = find_tariff(text)
        if t:
            st.chosen_tariff_title = t.get("title")
            st.chosen_tariff_price = _tariff_price_rub(t)

        if not st.chosen_tariff_title:
            await message.answer(
                "Осталось уточнить тариф 🙂\n\n"
                f"{tariffs_brief()}\n\n"
                "Напиши точное название или «тариф 1/2/3…»"
            )
            return

        missing = []
        if not st.profile.first_name or not st.profile.last_name:
            missing.append("имя и фамилия")
        if not st.profile.phone or len(re.sub(r"\D", "", st.profile.phone)) < 10:
            missing.append("телефон")
        if not st.profile.email or not looks_like_email(st.profile.email):
            missing.append("email")

        if missing:
            await message.answer("Мне не хватает: " + ", ".join(missing) + " 🙂 Напиши, пожалуйста.")
            return

        # Лид админу
        lead_text = (
            "🟩 ЗАЯВКА НА ПОКУПКУ (INSTART)\n"
            f"ФИО: {st.profile.first_name} {st.profile.last_name}\n"
            f"Телефон: {st.profile.phone}\n"
            f"Email: {st.profile.email}\n"
            f"Тариф: {st.chosen_tariff_title} — {st.chosen_tariff_price or '—'} ₽\n"
            f"User ID: {uid}"
        )
        await send_admin(lead_text)

        pay_phone = kget("instructions.payment.phone", "89883873424")
        pay_bank = kget("instructions.payment.bank", "Кубань Кредит")

        await message.answer(
            "Супер, оформила ✅\n\n"
            "Реквизиты для оплаты:\n"
            f"📞 Номер телефона: {pay_phone}\n"
            f"🏦 Банк: {pay_bank}\n\n"
            "После оплаты пришли, пожалуйста, чек (фото) сюда — и я передам на подтверждение 🙂"
        )

        st.stage = Stage.WAIT_RECEIPT
        return

    # =========================
    # OpenAI fallback (если не попали в правила)
    # =========================
    add_history(uid, "user", text)

    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(typing_loop(message.chat.id, stop_event))
    start_ts = time.time()

    def call_openai_sync(messages: List[dict]) -> str:
        resp = client.responses.create(
            model=MODEL,
            input=messages,
            temperature=0.5,
            max_output_tokens=220,
        )
        return (resp.output_text or "").strip()

    try:
        sys = build_system_prompt(uid)
        msgs = [{"role": "system", "content": sys}]
        msgs.extend(st.history[-HISTORY_MAX_TURNS * 2 :])
        msgs.append({"role": "user", "content": text})

        answer = await asyncio.to_thread(call_openai_sync, msgs)

        elapsed = time.time() - start_ts
        if elapsed < 2.5:
            await asyncio.sleep(2.5 - elapsed)

        parts = split_answer(answer, max_chars=850)
        if not parts:
            parts = ["Я задумалась 😅 Напиши чуть иначе — и я помогу."]
        for p in parts:
            await message.answer(p)

        add_history(uid, "assistant", answer)
        if st.stage == Stage.QUALIFY:
            st.stage = Stage.NORMAL

    except Exception as e:
        log.exception("OpenAI error: %s", e)
        await message.answer("⚠️ Сейчас я немного перегружена. Попробуй ещё раз через минуту 🙂")
    finally:
        stop_event.set()
        try:
            await typing_task
        except Exception:
            pass

# =========================
# WEBHOOK + HEALTHCHECK
# =========================
async def health(request: web.Request):
    return web.Response(text="ok")

async def on_startup(app: web.Application):
    # важно: перед установкой вебхука лучше удалить старый
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
    app.router.add_get("/", health)

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
