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

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# KNOWLEDGE LOADER
# =========================
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge.yaml")

def load_knowledge() -> dict:
    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        log.warning("knowledge.yaml не найден рядом с main.py")
        return {}
    except Exception as e:
        log.exception("Ошибка чтения knowledge.yaml: %s", e)
        return {}
    return data or {}

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


# =========================
# INDEX COURSES + TARIFFS
# =========================
@dataclass
class KBItem:
    kind: str          # "course" | "tariff"
    id: str
    title: str
    aliases: List[str]
    payload: dict

ALIAS_INDEX: Dict[str, List[KBItem]] = {}
ALL_ITEMS: List[KBItem] = []

def rebuild_index() -> None:
    global ALIAS_INDEX, ALL_ITEMS
    ALIAS_INDEX = {}
    ALL_ITEMS = []

    courses = kget("courses", [])
    tariffs = kget("tariffs", [])

    def add_items(items: list, kind: str):
        if not isinstance(items, list):
            return
        for it in items:
            if not isinstance(it, dict):
                continue
            _id = str(it.get("id") or "").strip()
            title = str(it.get("title") or "").strip()
            aliases = it.get("aliases") if isinstance(it.get("aliases"), list) else []
            aliases = [a for a in aliases if isinstance(a, str) and a.strip()]

            if not _id or not title:
                continue

            kb = KBItem(kind=kind, id=_id, title=title, aliases=aliases, payload=it)
            ALL_ITEMS.append(kb)

            keys = set()
            keys.add(normalize_text(title))
            keys.add(normalize_text(_id))
            for a in aliases:
                keys.add(normalize_text(a))

            for k in keys:
                if not k:
                    continue
                ALIAS_INDEX.setdefault(k, []).append(kb)

    add_items(courses, "course")
    add_items(tariffs, "tariff")

rebuild_index()


def find_items(text: str, kinds: Optional[List[str]] = None) -> List[KBItem]:
    q = normalize_text(text)
    if not q:
        return []

    results: List[KBItem] = []

    # 1) точное совпадение
    if q in ALIAS_INDEX:
        results.extend(ALIAS_INDEX[q])

    # 2) вхождение алиаса в запрос (чтобы "курс по нейросетям" находил)
    for k, items in ALIAS_INDEX.items():
        if len(k) >= 4 and k in q:
            results.extend(items)

    # uniq by (kind,id)
    seen = set()
    uniq = []
    for it in results:
        key = (it.kind, it.id)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)

    if kinds:
        want = {x.lower() for x in kinds}
        uniq = [x for x in uniq if x.kind.lower() in want]

    return uniq


def find_one(text: str, kinds: Optional[List[str]] = None) -> Optional[KBItem]:
    arr = find_items(text, kinds=kinds)
    return arr[0] if arr else None


# =========================
# PROJECT META
# =========================
ASSISTANT_NAME = kget("assistant.name", "Лиза")
OWNER_NAME = kget("assistant.owner_name", "Юлия")
PROJECT_NAME = kget("project.name", "INSTART")


# =========================
# STATE / MEMORY
# =========================
HISTORY_MAX_TURNS = 10
STATE_TTL_SECONDS = 6 * 60 * 60

class Stage:
    ASK_NAME = "ask_name"
    QUALIFY = "qualify"
    NORMAL = "normal"
    BUY_COLLECT = "buy_collect"

@dataclass
class UserProfile:
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    sex: Optional[str] = None  # "m"/"f"/None
    phone: Optional[str] = None
    email: Optional[str] = None

@dataclass
class UserState:
    stage: str = Stage.ASK_NAME
    last_seen: float = field(default_factory=lambda: time.time())
    history: List[dict] = field(default_factory=list)
    profile: UserProfile = field(default_factory=UserProfile)

    chosen_kind: Optional[str] = None
    chosen_id: Optional[str] = None
    chosen_title: Optional[str] = None

    sent_media_file_ids: set = field(default_factory=set)

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
# PARSING
# =========================
def extract_name(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None

    m = re.search(r"(?:меня\s+зовут|я)\s+([А-ЯЁA-Z][а-яёa-z\-]+)(?:\s+([А-ЯЁA-Z][а-яёa-z\-]+))?", text, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)

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


# =========================
# INTENTS
# =========================
BUY_INTENT_RE = re.compile(r"\b(купить|оплат(ить|а)|готов(а)? купить|беру|хочу оформить|оформим)\b", re.IGNORECASE)

def is_guest_request(text: str) -> bool:
    t = normalize_text(text)
    return any(w in t for w in ["гост", "демо", "пробн", "ключ", "гостевой"])

def is_presentation_request(text: str) -> bool:
    t = normalize_text(text)
    return "презентац" in t

def is_tariffs_question(text: str) -> bool:
    t = normalize_text(text)
    return any(w in t for w in ["тариф", "тарифи", "пакет", "пакеты", "цена тарифа"])

def is_courses_question(text: str) -> bool:
    t = normalize_text(text)
    return any(w in t for w in ["курс", "курсы", "обучение", "направлени", "программа"])

def is_project_question(text: str) -> bool:
    t = normalize_text(text)
    return any(w in t for w in ["инстарт", "instart", "проект", "школ", "платформ", "что такое"])


# =========================
# MEDIA SENDERS
# =========================
def infer_media_type(file_id: str) -> str:
    # очень грубая эвристика по префиксу telegram file_id
    if not file_id:
        return "document"
    if file_id.startswith("AgACAg"):
        return "photo"
    # BAACAg чаще всего document
    if file_id.startswith("BAACAg") or file_id.startswith("BQACAg") or file_id.startswith("BQA"):
        return "document"
    return "document"


async def send_file_id(message: Message, st: UserState, file_id: str, caption: str = "") -> bool:
    if not file_id:
        return False
    if file_id in st.sent_media_file_ids:
        return False

    mtype = infer_media_type(file_id)
    cap = caption[:1024] if caption else None

    try:
        if mtype == "photo":
            await message.answer_photo(photo=file_id, caption=cap)
        else:
            await message.answer_document(document=file_id, caption=cap)
        st.sent_media_file_ids.add(file_id)
        return True
    except Exception as e:
        log.exception("Не удалось отправить медиа: %s", e)
        return False


async def send_course_media(message: Message, st: UserState, course: KBItem) -> bool:
    payload = course.payload
    media = payload.get("media")
    if isinstance(media, dict) and media.get("file_id"):
        caption = media.get("title") or f"Материалы по курсу «{course.title}»"
        return await send_file_id(message, st, media["file_id"], caption=caption)

    media_refs = payload.get("media_refs")
    if isinstance(media_refs, dict):
        # иногда там лежит key на media-словарь
        for _, ref in media_refs.items():
            if isinstance(ref, str):
                mm = kget(f"media.{ref}")
                if isinstance(mm, dict) and mm.get("file_id"):
                    caption = mm.get("title") or f"Материалы по курсу «{course.title}»"
                    return await send_file_id(message, st, mm["file_id"], caption=caption)
            if isinstance(ref, dict) and ref.get("file_id"):
                caption = ref.get("title") or f"Материалы по курсу «{course.title}»"
                return await send_file_id(message, st, ref["file_id"], caption=caption)

    return False


async def send_tariff_media(message: Message, st: UserState, tariff: KBItem) -> bool:
    payload = tariff.payload
    media_refs = payload.get("media_refs")
    if isinstance(media_refs, dict):
        mock = media_refs.get("description_mockup")
        if isinstance(mock, dict) and mock.get("file_id"):
            caption = mock.get("title") or f"Макет тарифа «{tariff.title}»"
            return await send_file_id(message, st, mock["file_id"], caption=caption)
        # иногда может быть просто строка-ключ
        if isinstance(mock, str):
            mm = kget(f"media.{mock}")
            if isinstance(mm, dict) and mm.get("file_id"):
                caption = mm.get("title") or f"Макет тарифа «{tariff.title}»"
                return await send_file_id(message, st, mm["file_id"], caption=caption)
    return False


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
# PROMPT (ваш, но коротко)
# =========================
def build_system_prompt(uid: int) -> str:
    disclaim = kget("project.disclaimers.income", "Доход не гарантируется и зависит от усилий.")
    return f"""
Вы — “{ASSISTANT_NAME}”, ассистент куратора {OWNER_NAME} в онлайн-школе {PROJECT_NAME} и менеджер по продажам.

ПРАВИЛА:
- Общение только на «Вы».
- Факты о школе/курсах/тарифах/ценах/бонусах/медиа — ТОЛЬКО из knowledge.yaml, не выдумывать.
- Если в базе нет ответа — скажите, что уточните у куратора, и задайте 1 уточняющий вопрос.
- Не обещайте гарантированный доход. Формулировка: {disclaim}

СТИЛЬ:
- Дружелюбно, живо, без давления.
- 1–6 коротких абзацев.
- В конце 1 вопрос.
""".strip()


# =========================
# COMMANDS
# =========================
@dp.message(Command("reload"))
async def cmd_reload(message: Message):
    global knowledge
    knowledge = load_knowledge()
    rebuild_index()
    await message.answer("knowledge.yaml перечитан ✅")

@dp.message(Command("myid"))
async def cmd_myid(message: Message):
    await message.answer(f"Your ID: {message.from_user.id}\nCurrent chat ID: {message.chat.id}")


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

    # 1) имя
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
            await message.answer("Подскажите, пожалуйста, как я могу к Вам обращаться? 🙂")
        return

    # 2) проект INSTART
    if is_project_question(text):
        desc = kget("project.description", "")
        mission = kget("project.mission", "")
        founded = kget("project.founded.purpose", "")
        benefits = kget("instart_school_benefits", {})

        lines = []
        if desc:
            lines.append(desc.strip())
        if mission:
            lines.append(f"**Миссия:** {mission}".strip())
        if founded:
            lines.append(f"**Зачем создан проект:** {founded}".strip())

        # коротко 3–4 преимущества, если есть
        if isinstance(benefits, dict):
            bullets = []
            for key in ["quality", "affordable_cost", "freedom", "ease", "convenience", "right_to_choose"]:
                v = benefits.get(key)
                if isinstance(v, dict):
                    title = v.get("title")
                    short = v.get("short") or v.get("text")
                    if title and short:
                        bullets.append(f"• **{title}:** {str(short).strip()}")
                elif isinstance(v, str):
                    bullets.append(f"• {v.strip()}")
                if len(bullets) >= 4:
                    break
            if bullets:
                lines.append("\n".join(bullets))

        if not lines:
            await message.answer(
                "Я вижу, что проект INSTART есть в базе, но описания сейчас недостаточно 🙈\n"
                "Подскажите, пожалуйста: Вам интереснее подработка, новая профессия или партнёрство?"
            )
            return

        await message.answer("\n\n".join(lines), parse_mode="Markdown")
        await message.answer("Подскажите, пожалуйста, какое направление Вам интереснее всего (например: нейросети, Reels, дизайн, маркетплейсы)?")
        st.stage = Stage.NORMAL
        return

    # 3) гостевой доступ
    if is_guest_request(text):
        guest_site = kget("guest_access.guest_key.site", "")
        guest_key = kget("guest_access.guest_key.key", "")
        layout_id = kget("guest_access.registration_layout_file_id", "")
        pres_id = kget("guest_access.promo_materials.presentation_file_id", "")

        lines = ["Конечно 🙂"]
        if guest_site:
            lines.append(f"\nСайт для регистрации: {guest_site}")
        if guest_key:
            lines.append(f"\n🔑 Гостевой ключ: `{guest_key}`")

        await message.answer("\n".join(lines), parse_mode="Markdown")

        if layout_id:
            await message.answer("Отправляю макет для регистрации ✅")
            await send_file_id(message, st, layout_id, caption="Макет для регистрации ✅")

        if pres_id:
            await message.answer("И прикрепляю презентацию проекта 📎")
            await send_file_id(message, st, pres_id, caption="Презентация проекта INSTART 📎")

        await message.answer("Подскажите, пожалуйста, Вы хотите гостевой доступ, чтобы выбрать направление, или уже есть конкретный интерес?")
        return

    # 4) презентация
    if is_presentation_request(text):
        pres_id = kget("guest_access.promo_materials.presentation_file_id", "")
        if pres_id:
            await message.answer("Сейчас отправлю презентацию проекта 📎")
            ok = await send_file_id(message, st, pres_id, caption="Презентация проекта INSTART 📎")
            if not ok:
                await message.answer("Похоже, я уже отправляла презентацию ранее 🙂 Хотите, я кратко перескажу, что в ней?")
        else:
            await message.answer(
                "Я не вижу file_id презентации в базе 🙈\n"
                "Подскажите, пожалуйста, что именно хотите узнать про INSTART: подработка, профессия или партнёрство?"
            )
        return

    # 5) тарифы списком
    if is_tariffs_question(text):
        tariffs = kget("tariffs", [])
        if not isinstance(tariffs, list) or not tariffs:
            await message.answer("Сейчас я не вижу тарифы в базе 🙈 Подскажите, пожалуйста, что Вас интересует: курс или тариф?")
            return

        lines = ["Вот актуальные тарифы из базы 🙂\n"]
        for t in tariffs[:10]:
            title = t.get("title")
            price = t.get("price_rub")
            if title and price is not None:
                lines.append(f"• **{title}** — {price} ₽")
        await message.answer("\n".join(lines), parse_mode="Markdown")
        await message.answer("Подскажите, пожалуйста, какая цель у Вас сейчас: подработка или новая профессия?")
        return

    # 6) нашли конкретный курс/тариф по алиасам
    found = find_one(text, kinds=["course", "tariff"])
    if found:
        p = found.payload
        kind_ru = "Курс" if found.kind == "course" else "Тариф"

        lines = [f"**{kind_ru}:** {found.title}"]

        # цена (поддерживаем разные поля)
        if found.kind == "course":
            price = p.get("price")
            if isinstance(price, dict):
                with_chat = price.get("with_chat_rub")
                without_chat = price.get("without_chat_rub")
                if with_chat and without_chat and with_chat != without_chat:
                    lines.append(f"Цена: с чатом — {with_chat} ₽, без чата — {without_chat} ₽.")
                elif with_chat:
                    lines.append(f"Цена: {with_chat} ₽.")
                elif without_chat:
                    lines.append(f"Цена: {without_chat} ₽.")
            chat_av = p.get("chat_available")
            if isinstance(chat_av, bool):
                lines.append("Чат: " + ("есть ✅" if chat_av else "нет"))
            sd = p.get("short_description")
            if isinstance(sd, str) and sd.strip():
                lines.append("\n" + sd.strip())

        else:
            price_rub = p.get("price_rub")
            if price_rub is not None:
                lines.append(f"Цена: {price_rub} ₽.")
            short = p.get("short_about")
            if isinstance(short, str) and short.strip():
                lines.append("\n" + short.strip())

        await message.answer("\n".join(lines), parse_mode="Markdown")

        # медиа/макет
        if found.kind == "course":
            sent = await send_course_media(message, st, found)
        else:
            sent = await send_tariff_media(message, st, found)

        if sent:
            await message.answer("Прикрепила материалы, чтобы Вам было удобнее посмотреть ✅")
        else:
            # если медиа есть, но уже слали — промолчим; если вообще нет — тоже ок
            pass

        st.chosen_kind = found.kind
        st.chosen_id = found.id
        st.chosen_title = found.title

        await message.answer("Подскажите, пожалуйста, Вы рассматриваете этот вариант для себя или хотите сравнить ещё с 1–2 вариантами?")
        return

    # 7) если спрашивают "нейросети", а точного курса не нашли — предложим варианты по ключевому слову
    q = normalize_text(text)
    if "нейросет" in q or "ai" in q:
        candidates = [x for x in ALL_ITEMS if x.kind == "course" and ("нейросет" in normalize_text(x.title) or any("нейросет" in normalize_text(a) for a in x.aliases))]
        if candidates:
            lines = ["Я нашла в базе курсы по нейросетям 🙂"]
            for c in candidates[:5]:
                lines.append(f"• **{c.title}**")
            await message.answer("\n".join(lines), parse_mode="Markdown")
            await message.answer("Подскажите, пожалуйста, какой формат Вам ближе: нейросети для фото/видео или, например, под задачи контента?")
            return

    # 8) покупка — ТОЛЬКО если выбран конкретный курс/тариф
    if BUY_INTENT_RE.search(text):
        if not st.chosen_title:
            await message.answer(
                "Хорошо 🙂 Сначала уточним выбор.\n"
                "Напишите, пожалуйста, какой курс или тариф Вы хотите купить (как Вы его называете) — я найду по базе."
            )
            return

        st.stage = Stage.BUY_COLLECT
        await message.answer(
            f"Отлично 🙂 Подтверждаю: **{st.chosen_title}**.\n\n"
            "Чтобы оформить заявку, напишите одним сообщением:\n"
            "1) Фамилия Имя\n"
            "2) Телефон\n"
            "3) E-mail\n"
            "4) Выбранный курс/тариф (на всякий случай ещё раз)\n",
            parse_mode="Markdown"
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

        # если человек вдруг здесь написал название — обновим выбранное
        chosen = find_one(text, kinds=["course", "tariff"])
        if chosen:
            st.chosen_kind = chosen.kind
            st.chosen_id = chosen.id
            st.chosen_title = chosen.title

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

        now_str = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        lead_text = (
            "🟩 ЗАЯВКА НА ПОКУПКУ (INSTART)\n"
            f"Имя клиента: {st.profile.first_name}\n"
            f"Пол: {st.profile.sex or 'не определён'}\n"
            f"Фамилия Имя: {st.profile.last_name} {st.profile.first_name}\n"
            f"Телефон: {st.profile.phone}\n"
            f"Email: {st.profile.email}\n"
            f"Курс/Тариф: {st.chosen_title}\n"
            f"Источник: Telegram\n"
            f"Краткий запрос/цель: {text[:200]}\n"
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
    # OpenAI fallback (только если в базе не нашли)
    # =========================
    add_history(uid, "user", text)

    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(typing_loop(message.chat.id, stop_event))
    start_ts = time.time()

    def call_openai_sync(messages: List[dict]) -> str:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=220,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        sys = build_system_prompt(uid)

        # IMPORTANT: не даём модели выдумывать — просим уточнить, если нет данных
        msgs = [{"role": "system", "content": sys}]
        msgs.extend(st.history[-HISTORY_MAX_TURNS * 2 :])
        msgs.append({"role": "user", "content": text})

        answer = await asyncio.to_thread(call_openai_sync, msgs)

        elapsed = time.time() - start_ts
        if elapsed < 1.5:
            await asyncio.sleep(1.5 - elapsed)

        parts = split_answer(answer, max_chars=900)
        if not parts:
            parts = ["Подскажите, пожалуйста, чуть подробнее, что именно Вам важно — я помогу 🙂"]

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
