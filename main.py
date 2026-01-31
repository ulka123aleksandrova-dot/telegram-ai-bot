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

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/tg/webhook")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")
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

def load_knowledge() -> Dict[str, Any]:
    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        log.error("knowledge.yaml не найден рядом с main.py (%s)", KNOWLEDGE_PATH)
        return {}
    except Exception as e:
        log.exception("Ошибка чтения knowledge.yaml: %s", e)
        return {}

    if data is None:
        return {}
    if not isinstance(data, dict):
        log.error("knowledge.yaml должен быть YAML-словарём (dict) в корне. Сейчас: %s", type(data))
        return {}

    return data

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


def courses_list() -> List[dict]:
    c = knowledge.get("courses", [])
    return c if isinstance(c, list) else []


def tariffs_list() -> List[dict]:
    t = knowledge.get("tariffs", [])
    return t if isinstance(t, list) else []


# =========================
# INDEX: aliases/title/id → item
# =========================
ALIAS_INDEX: Dict[str, List[dict]] = {}

def rebuild_index() -> None:
    global ALIAS_INDEX
    ALIAS_INDEX = {}

    items = []
    items.extend([x for x in courses_list() if isinstance(x, dict)])
    items.extend([x for x in tariffs_list() if isinstance(x, dict)])

    for it in items:
        keys = set()

        title = str(it.get("title") or "").strip()
        if title:
            keys.add(normalize_text(title))

        _id = it.get("id")
        if _id:
            keys.add(normalize_text(str(_id)))

        aliases = it.get("aliases") or []
        if isinstance(aliases, list):
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    keys.add(normalize_text(a))

        for k in keys:
            ALIAS_INDEX.setdefault(k, []).append(it)

rebuild_index()


def find_items_by_query(text: str, types: Optional[List[str]] = None) -> List[dict]:
    q = normalize_text(text)
    if not q:
        return []

    results: List[dict] = []

    # 1) точное совпадение
    if q in ALIAS_INDEX:
        results.extend(ALIAS_INDEX[q])

    # 2) вхождение алиаса в запрос
    for k, items in ALIAS_INDEX.items():
        if len(k) >= 4 and k in q:
            results.extend(items)

    # уникализация по id
    seen = set()
    uniq = []
    for it in results:
        it_id = it.get("id") or id(it)
        if it_id in seen:
            continue
        seen.add(it_id)
        uniq.append(it)

    if types:
        types_norm = {t.lower() for t in types}
        uniq = [x for x in uniq if str(x.get("type", "")).lower() in types_norm]

    return uniq


def find_one_item(text: str, types: Optional[List[str]] = None) -> Optional[dict]:
    items = find_items_by_query(text, types=types)
    return items[0] if items else None


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
STATE_TTL_SECONDS = 6 * 60 * 60  # 6 часов

class Stage:
    ASK_NAME = "ask_name"
    QUALIFY = "qualify"
    NORMAL = "normal"
    BUY_COLLECT = "buy_collect"

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
    last_seen: float = field(default_factory=lambda: time.time())
    history: List[dict] = field(default_factory=list)
    profile: UserProfile = field(default_factory=UserProfile)
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
    """
    Достаёт имя даже из длинной фразы:
    "привет! меня зовут Юлия. хочу узнать..." -> Юлия
    """
    if not text:
        return None, None

    m = re.search(r"(?:меня\s+зовут|я)\s+([А-ЯЁA-Z][а-яёa-z\-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1), None

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


BUY_INTENT_RE = re.compile(r"\b(купить|оплат(ить|а)|готов(а)? купить|беру|хочу оформить|оформим)\b", re.IGNORECASE)

def is_guest_request(text: str) -> bool:
    t = normalize_text(text)
    return any(w in t for w in ["гост", "демо", "пробн", "ключ"])

def is_presentation_request(text: str) -> bool:
    t = normalize_text(text)
    return "презентац" in t

def is_tariffs_request(text: str) -> bool:
    t = normalize_text(text)
    return "тариф" in t or "тарифа" in t or "тарифы" in t

def is_project_request(text: str) -> bool:
    t = normalize_text(text)
    return any(w in t for w in ["инстарт", "instart", "проект", "о проекте", "про проект", "что такое"])


# =========================
# MEDIA SEND
# =========================
async def send_file_id(message: Message, st: UserState, file_id: str, kind: str = "document", caption: str = "") -> bool:
    if not file_id:
        return False
    if file_id in st.sent_media_file_ids:
        return False

    cap = caption[:1024] if caption else None

    if kind == "photo":
        await message.answer_photo(photo=file_id, caption=cap)
    elif kind == "video":
        await message.answer_video(video=file_id, caption=cap)
    else:
        await message.answer_document(document=file_id, caption=cap)

    st.sent_media_file_ids.add(file_id)
    return True

async def send_item_media(message: Message, st: UserState, item: dict, caption: Optional[str] = None) -> bool:
    media = item.get("media")
    if not isinstance(media, dict):
        return False

    mtype = media.get("type") or "document"
    fid = media.get("file_id")
    if not fid:
        return False

    cap = caption or media.get("caption") or media.get("title") or ""
    return await send_file_id(message, st, str(fid), kind=str(mtype), caption=cap)


# =========================
# ADMIN
# =========================
async def send_admin(text: str) -> None:
    try:
        await bot.send_message(ADMIN_CHAT_ID_INT, text)
    except Exception as e:
        log.exception("Failed to send admin message: %s", e)


# =========================
# TYPING + SPLIT
# =========================
async def typing_loop(chat_id: int, stop_event: asyncio.Event) -> None:
    try:
        while not stop_event.is_set():
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(3)
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
# SYSTEM PROMPT (ВАШ)
# =========================
def build_system_prompt(uid: int) -> str:
    disclaim = kget("project.disclaimers.income", "Доход не гарантируется и зависит от усилий.")
    return f"""
Вы — “{ASSISTANT_NAME}”, ассистент куратора Юлии в онлайн-школе {PROJECT_NAME} и профессиональный менеджер по продажам (не проговаривается клиенту).

ВАЖНО:
- Общение только на «Вы».
- Все факты о школе/курсах/тарифах/ценах/медиа — ТОЛЬКО из knowledge.yaml.
- Если данных нет — не выдумывайте: предложите оставить контакт для уточнения у куратора.
- Не обещайте гарантированный доход. Формулировка: {disclaim}

Стиль:
- Дружелюбно, тактично, живо. Без давления.
- 1–6 коротких абзацев. Списки уместны.
- В конце 1 уточняющий вопрос.
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
    await message.answer(f"Your ID: {message.from_user.id}\nChat ID: {message.chat.id}")


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
        f"Я {ASSISTANT_NAME} — помощница куратора Юлии в онлайн-школе {PROJECT_NAME}.\n"
        f"Помогу подобрать курс и тариф под Вашу цель.\n\n"
        f"Как я могу к Вам обращаться?"
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

    # 0) Проверка: knowledge реально загрузился
    if not knowledge:
        await message.answer(
            "Сейчас база знаний не загрузилась 🙈\n"
            "Пожалуйста, проверьте knowledge.yaml (ошибка формата/отступов) и напишите /reload."
        )
        return

    # 1) Имя
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
            await message.answer("Как я могу к Вам обращаться? 🙂")
        return

    # 2) Про проект INSTART (строго из YAML)
    if is_project_request(text):
        desc = kget("project.description") or kget("project.current_state") or ""
        mission = kget("project.mission")
        audience = kget("project.audience", [])
        cert = kget("project.current_state.certificates.description")

        lines = []
        if isinstance(desc, str) and desc.strip():
            lines.append(desc.strip())
        if isinstance(mission, str) and mission.strip():
            lines.append(f"\nМиссия: {mission.strip()}")
        if isinstance(cert, str) and cert.strip():
            lines.append(f"\nСертификат: {cert.strip()}")
        if isinstance(audience, list) and audience:
            lines.append("\nКому подходит: " + ", ".join(audience[:6]) + ("…" if len(audience) > 6 else ""))

        if not lines:
            lines = ["В базе есть INSTART, но сейчас не вижу заполненного описания проекта."]

        await message.answer("\n".join(lines).strip())
        await message.answer("Подскажите, пожалуйста, какая цель у Вас сейчас основная: подработка, новая профессия или развитие в проекте?")
        return

    # 3) Гостевой доступ
    if is_guest_request(text):
        website = kget("guest_access.website")
        guest_key = kget("guest_access.guest_key")
        pres_id = kget("guest_access.promo_materials.presentation_file_id")

        msg = ["Конечно 🙂"]
        if website:
            msg.append(f"\nСайт для гостевого доступа: {website}")
        if guest_key:
            msg.append(f"\n🔑 Гостевой ключ: `{guest_key}`")
        await message.answer("\n".join(msg), parse_mode="Markdown")

        if pres_id:
            await message.answer("Сейчас отправлю презентацию проекта 📎")
            sent = await send_file_id(message, st, str(pres_id), kind="document", caption="Презентация проекта INSTART 📎")
            if not sent:
                await message.answer("Презентацию я уже отправляла ранее 🙂")

        await message.answer("Хотите, я подскажу 1–2 направления под Вашу цель, чтобы было проще выбрать?")
        return

    # 4) Презентация
    if is_presentation_request(text):
        pres_id = kget("guest_access.promo_materials.presentation_file_id")
        if pres_id:
            await message.answer("Сейчас отправлю презентацию проекта 📎")
            sent = await send_file_id(message, st, str(pres_id), kind="document", caption="Презентация проекта INSTART 📎")
            if not sent:
                await message.answer("Презентацию я уже отправляла ранее 🙂")
        else:
            await message.answer("В базе не вижу file_id презентации 🙈 Проверьте guest_access.promo_materials.presentation_file_id")
        return

    # 5) Тарифы списком
    if is_tariffs_request(text):
        ts = tariffs_list()
        if not ts:
            await message.answer("Сейчас не вижу тарифов в базе 🙈")
            return

        lines = ["Актуальные тарифы:"]
        for t in ts[:12]:
            title = t.get("title", "Без названия")
            price = t.get("price")
            # поддерживаем price: {with_chat_rub, without_chat_rub} или price_rub
            p = None
            if isinstance(price, dict):
                p = price.get("with_chat_rub") or price.get("without_chat_rub")
            if p is None:
                p = t.get("price_rub")
            if p:
                lines.append(f"• {title} — {p} ₽")
            else:
                lines.append(f"• {title}")

        await message.answer("\n".join(lines))
        await message.answer("Подскажите, пожалуйста, какая цель у Вас сейчас и какой бюджет комфортен?")
        return

    # 6) Конкретный курс/тариф по алиасам
    found = find_one_item(text, types=["course", "tariff"])
    if found:
        title = found.get("title", "Без названия")
        typ = str(found.get("type", "")).lower()
        price = found.get("price") or {}
        short_desc = found.get("short_description") or found.get("description")

        lines = []
        lines.append(("Курс: " if typ == "course" else "Тариф: ") + str(title))

        if isinstance(price, dict):
            p_with = price.get("with_chat_rub")
            p_without = price.get("without_chat_rub")
            if p_with and p_without and p_with != p_without:
                lines.append(f"Цена: с чатом — {p_with} ₽, без чата — {p_without} ₽.")
            elif p_with:
                lines.append(f"Цена: {p_with} ₽.")
            elif p_without:
                lines.append(f"Цена: {p_without} ₽.")

        if isinstance(short_desc, str) and short_desc.strip():
            lines.append("\n" + short_desc.strip())

        await message.answer("\n".join(lines).strip())

        sent = await send_item_media(message, st, found, caption=f"Материалы по «{title}» 📎")
        if not sent and isinstance(found.get("media"), dict) and found["media"].get("file_id"):
            await message.answer("Материалы я уже отправляла ранее 🙂")

        await message.answer("Подскажите, пожалуйста: рассматриваете этот вариант для себя или хотите сравнить с ещё 1–2 вариантами?")
        return

    # 7) OpenAI fallback (только если не нашли в базе)
    add_history(uid, "user", text)

    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(typing_loop(message.chat.id, stop_event))

    def call_openai_sync(messages: List[dict]) -> str:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=260,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        sys = build_system_prompt(uid)
        msgs = [{"role": "system", "content": sys}]
        msgs.extend(st.history[-HISTORY_MAX_TURNS * 2 :])
        msgs.append({"role": "user", "content": text})

        answer = await asyncio.to_thread(call_openai_sync, msgs)

        parts = split_answer(answer, max_chars=900)
        if not parts:
            parts = ["Я задумалась 😅 Напишите, пожалуйста, чуть иначе — и я помогу."]

        for p in parts:
            await message.answer(p)

        add_history(uid, "assistant", answer)

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
