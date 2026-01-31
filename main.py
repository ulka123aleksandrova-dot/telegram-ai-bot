import os
import re
import json
import time
import yaml
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
from aiohttp import web
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart, Command
from aiogram.enums import ChatAction
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram.utils.keyboard import InlineKeyboardBuilder

# OpenAI optional (fallback)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =========================
# CONFIG / ENV
# =========================
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
INTERNAL_CHAT_ID = os.getenv("INTERNAL_CHAT_ID")  # куда слать заявки (группа/канал)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")              # https://xxxx.up.railway.app
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/tg/webhook")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me")
PORT = int(os.getenv("PORT", "8080"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN в переменных окружения.")
if not INTERNAL_CHAT_ID:
    raise RuntimeError("Не найден INTERNAL_CHAT_ID в переменных окружения.")
if not WEBHOOK_BASE:
    raise RuntimeError("Не найден WEBHOOK_BASE в переменных окружения (Railway Variables).")

INTERNAL_CHAT_ID_INT = int(INTERNAL_CHAT_ID)

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

openai_client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        log.warning("OpenAI init failed: %s", e)
        openai_client = None


# =========================
# KNOWLEDGE (YAML)
# =========================
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge.yaml")

def load_knowledge() -> Dict[str, Any]:
    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("knowledge.yaml должен быть YAML-словарём (dict) в корне.")
        return data
    except FileNotFoundError:
        log.exception("knowledge.yaml не найден рядом с main.py")
        return {}
    except Exception:
        log.exception("Ошибка чтения knowledge.yaml")
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


# =========================
# NORMALIZATION / SEARCH
# =========================
def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    return s

def safe_list(x) -> List[Any]:
    return x if isinstance(x, list) else []

def build_index(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    idx: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title", "")).strip()
        aliases = safe_list(it.get("aliases"))
        keys = set()
        if title:
            keys.add(norm(title))
        for a in aliases:
            if isinstance(a, str) and a.strip():
                keys.add(norm(a))
        # id тоже как ключ
        if it.get("id"):
            keys.add(norm(str(it["id"])))
        for k in keys:
            idx.setdefault(k, []).append(it)
    return idx

COURSES = safe_list(knowledge.get("courses"))
TARIFFS = safe_list(knowledge.get("tariffs"))
COURSE_INDEX = build_index(COURSES)
TARIFF_INDEX = build_index(TARIFFS)

def find_best(items_index: Dict[str, List[Dict[str, Any]]], query: str) -> Optional[Dict[str, Any]]:
    q = norm(query)
    if not q:
        return None

    # 1) точное совпадение
    if q in items_index:
        return items_index[q][0]

    # 2) “вхождение ключа в запрос”
    best = None
    best_len = 0
    for k, arr in items_index.items():
        if len(k) < 4:
            continue
        if k in q and len(k) > best_len:
            best = arr[0]
            best_len = len(k)

    # 3) “вхождение запроса в ключ”
    if best is None:
        for k, arr in items_index.items():
            if len(q) >= 4 and q in k and len(q) > best_len:
                best = arr[0]
                best_len = len(q)

    return best

def find_tariff(query: str) -> Optional[Dict[str, Any]]:
    return find_best(TARIFF_INDEX, query)

def find_course(query: str) -> Optional[Dict[str, Any]]:
    return find_best(COURSE_INDEX, query)


# =========================
# SQLITE STORAGE (STATE)
# =========================
DB_PATH = os.path.join(os.path.dirname(__file__), "bot.db")

class Stage:
    ASK_NAME = "ask_name"
    QUALIFY_GOAL = "qualify_goal"
    QUALIFY_TIME = "qualify_time"
    NORMAL = "normal"
    COLLECT_CONTACTS = "collect_contacts"

async def db_init():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            sex TEXT,
            stage TEXT,
            goal TEXT,
            time_budget TEXT,
            chosen_type TEXT,
            chosen_id TEXT,
            chosen_title TEXT,
            last_suggested_type TEXT,
            last_suggested_id TEXT,
            updated_at INTEGER
        );
        """)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT,
            content TEXT,
            ts INTEGER
        );
        """)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS sent_media (
            user_id INTEGER,
            file_id TEXT,
            PRIMARY KEY (user_id, file_id)
        );
        """)
        await db.commit()

async def get_user(user_id: int) -> Dict[str, Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute_fetchone("SELECT * FROM users WHERE user_id = ?", (user_id,))
        if not row:
            # create default
            await db.execute(
                "INSERT INTO users (user_id, stage, updated_at) VALUES (?, ?, ?)",
                (user_id, Stage.ASK_NAME, int(time.time()))
            )
            await db.commit()
            return {
                "user_id": user_id,
                "first_name": None,
                "last_name": None,
                "sex": None,
                "stage": Stage.ASK_NAME,
                "goal": None,
                "time_budget": None,
                "chosen_type": None,
                "chosen_id": None,
                "chosen_title": None,
                "last_suggested_type": None,
                "last_suggested_id": None,
            }
        return dict(row)

async def update_user(user_id: int, **fields):
    if not fields:
        return
    fields["updated_at"] = int(time.time())
    keys = list(fields.keys())
    sets = ", ".join([f"{k} = ?" for k in keys])
    vals = [fields[k] for k in keys]
    vals.append(user_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE users SET {sets} WHERE user_id = ?", vals)
        await db.commit()

async def add_history(user_id: int, role: str, content: str, max_turns: int = 20):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO history (user_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (user_id, role, content, int(time.time()))
        )
        # trim
        rows = await db.execute_fetchall(
            "SELECT id FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, max_turns * 2)
        )
        if rows:
            keep_ids = {r[0] for r in rows}
            await db.execute(
                f"DELETE FROM history WHERE user_id = ? AND id NOT IN ({','.join(['?'] * len(keep_ids))})",
                [user_id, *keep_ids]
            )
        await db.commit()

async def get_history(user_id: int, limit: int = 16) -> List[Dict[str, str]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT role, content FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        )
    out = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    return out

async def media_already_sent(user_id: int, file_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        row = await db.execute_fetchone(
            "SELECT 1 FROM sent_media WHERE user_id = ? AND file_id = ?",
            (user_id, file_id)
        )
        return bool(row)

async def mark_media_sent(user_id: int, file_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO sent_media (user_id, file_id) VALUES (?, ?)",
            (user_id, file_id)
        )
        await db.commit()


# =========================
# LANGUAGE / NAME / SEX
# =========================
NAME_RE = re.compile(r"(?:меня\s+зовут|я\s*[-—:]?)\s*([A-Za-zА-Яа-яЁё\-]{2,})", re.IGNORECASE)

def extract_name(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    t = text.strip()

    m = NAME_RE.search(t)
    if m:
        first = m.group(1).strip()
        first = first[:1].upper() + first[1:].lower()
        return first, None

    # если просто "марина" / "Марина"
    words = re.findall(r"[A-Za-zА-Яа-яЁё\-]{2,}", t)
    if 1 <= len(words) <= 2 and len(t.split()) <= 4:
        first = words[0]
        first = first[:1].upper() + first[1:].lower()
        last = None
        if len(words) >= 2:
            last = words[1]
            last = last[:1].upper() + last[1:].lower()
        return first, last

    return None, None

def guess_sex_by_name(name: str) -> Optional[str]:
    n = norm(name)
    if not n:
        return None
    # грубая эвристика
    if n.endswith(("а", "я")) and n not in {"илья", "никита"}:
        return "f"
    if n in {"саша", "женя"}:
        return None  # неоднозначно
    return "m"

def agree(user: Dict[str, Any], male: str, female: str, neutral: Optional[str] = None) -> str:
    sex = user.get("sex")
    if sex == "m":
        return male
    if sex == "f":
        return female
    return neutral if neutral is not None else male


# =========================
# TYPING + SAFE SEND
# =========================
async def send_typing(chat_id: int):
    try:
        await bot.send_chat_action(chat_id, ChatAction.TYPING)
    except Exception:
        pass

def to_formal_ru(text: str) -> str:
    """
    В YAML у вас встречается "ты/твой". Для безопасности заменим на "Вы/Ваш".
    (не идеальная морфология, но лучше чем “ты” при требовании общения на Вы)
    """
    if not isinstance(text, str):
        return ""
    s = text
    s = re.sub(r"\bты\b", "Вы", s, flags=re.IGNORECASE)
    s = re.sub(r"\bтебя\b", "Вас", s, flags=re.IGNORECASE)
    s = re.sub(r"\bтебе\b", "Вам", s, flags=re.IGNORECASE)
    s = re.sub(r"\bтвой\b", "Ваш", s, flags=re.IGNORECASE)
    s = re.sub(r"\bтвоя\b", "Ваша", s, flags=re.IGNORECASE)
    s = re.sub(r"\bтвои\b", "Ваши", s, flags=re.IGNORECASE)
    s = re.sub(r"\bтвою\b", "Вашу", s, flags=re.IGNORECASE)
    return s


# =========================
# MEDIA SENDER (root media + item media_refs)
# =========================
async def send_media_by_file_id(user_id: int, message: Message, mtype: str, file_id: str, caption: str) -> bool:
    if not file_id:
        return False
    if await media_already_sent(user_id, file_id):
        await message.answer("Я уже отправляла это ранее 🙂 Посмотрите, пожалуйста, выше в чате — файл там.")
        return True

    cap = (caption or "")[:1024] if caption else None
    try:
        if mtype == "photo":
            await message.answer_photo(photo=file_id, caption=cap)
        elif mtype == "video":
            await message.answer_video(video=file_id, caption=cap)
        elif mtype == "document":
            await message.answer_document(document=file_id, caption=cap)
        else:
            return False
        await mark_media_sent(user_id, file_id)
        return True
    except Exception as e:
        log.exception("Failed to send media: %s", e)
        return False

def root_media_get(key: str) -> Optional[Dict[str, Any]]:
    media = knowledge.get("media")
    if isinstance(media, dict):
        v = media.get(key)
        if isinstance(v, dict):
            return v
    return None

async def send_root_media_key(user_id: int, message: Message, key: str, caption_override: Optional[str] = None) -> bool:
    m = root_media_get(key)
    if not m:
        return False
    mtype = str(m.get("type", ""))
    fid = str(m.get("file_id", "")).strip()
    title = str(m.get("title", "")).strip()
    caption = caption_override or title or ""
    return await send_media_by_file_id(user_id, message, mtype, fid, caption)

async def send_item_media_refs(user_id: int, message: Message, item: Dict[str, Any]) -> bool:
    """
    У тарифов/курсов у вас часто есть media_refs: { description_mockup: {type, file_id, title} }
    """
    refs = item.get("media_refs")
    if not isinstance(refs, dict):
        return False

    sent_any = False
    for _, m in refs.items():
        if not isinstance(m, dict):
            continue
        mtype = str(m.get("type", "")).strip()
        fid = str(m.get("file_id", "")).strip()
        title = str(m.get("title", "")).strip()
        if fid and mtype:
            ok = await send_media_by_file_id(user_id, message, mtype, fid, title or "Материалы 📎")
            sent_any = sent_any or ok
    return sent_any


# =========================
# ANSWERS FROM YAML (NO FANTASY)
# =========================
def project_brief() -> str:
    name = kget("project.name", "INSTART")
    mission = kget("project.mission", "")
    desc = kget("project.description", "")
    founded = kget("project.founded.date", "")
    license_num = kget("project.license.license_number", "")
    license_date = kget("project.license.license_date", "")
    approved = kget("project.license.approved_by", "")

    parts = []
    if desc:
        parts.append(desc.strip())
    if mission:
        parts.append(f"Миссия: {mission.strip()}.")
    if founded:
        parts.append(f"Проект основан: {founded}.")
    if license_num and license_date:
        tail = f"Лицензия: № {license_num} от {license_date}"
        if approved:
            tail += f" ({approved})."
        else:
            tail += "."
        parts.append(tail)
    if not parts:
        return f"{name} — онлайн-проект обучения. Уточню детали у куратора Юлии, если нужно."
    return "\n\n".join(parts)

def format_tariff(t: Dict[str, Any]) -> str:
    title = t.get("title", "Тариф")
    price = t.get("price_rub")
    short = t.get("short_about") or ""
    who = safe_list(t.get("who_for"))
    main_courses = safe_list(t.get("main_courses"))
    mini = safe_list(t.get("mini_courses"))
    support = safe_list(t.get("tools_and_support"))
    adv = safe_list(t.get("advantages"))

    lines = [f"**Тариф: {title}**"]
    if price:
        lines.append(f"Цена: {price} ₽.")
    if short:
        lines.append(short.strip())

    if main_courses:
        lines.append("\n**Основные курсы (внутри тарифа):**\n" + "\n".join([f"• {x}" for x in main_courses[:8]]))
    if mini:
        lines.append("\n**Мини-курсы:**\n" + "\n".join([f"• {x}" for x in mini[:6]]))
    if support:
        lines.append("\n**Поддержка и инструменты:**\n" + "\n".join([f"• {x}" for x in support[:6]]))
    if adv:
        lines.append("\n**Преимущества:**\n" + "\n".join([f"• {x}" for x in adv[:6]]))

    return "\n\n".join(lines)

def format_course(c: Dict[str, Any]) -> str:
    title = c.get("title", "Курс")
    category = c.get("category") or ""
    chat_av = c.get("chat_available")
    sd = c.get("short_description") or ""
    price = c.get("price", {})

    with_chat = None
    without_chat = None
    if isinstance(price, dict):
        with_chat = price.get("with_chat_rub")
        without_chat = price.get("without_chat_rub")

    lines = [f"**Курс: {title}**"]
    if category:
        lines.append(f"Категория: {category}")
    if with_chat or without_chat:
        if with_chat and without_chat and with_chat != without_chat:
            lines.append(f"Цена: с чатом — {with_chat} ₽, без чата — {without_chat} ₽.")
        elif with_chat:
            lines.append(f"Цена: {with_chat} ₽.")
        elif without_chat:
            lines.append(f"Цена: {without_chat} ₽.")
    if isinstance(chat_av, bool):
        lines.append("Чат: " + ("есть ✅" if chat_av else "нет"))

    if sd:
        lines.append("\n" + sd.strip())

    return "\n\n".join(lines)

def tariffs_brief() -> str:
    lines = []
    for t in TARIFFS:
        title = t.get("title")
        price = t.get("price_rub")
        if title and price:
            lines.append(f"• {title} — {price} ₽")
    return "\n".join(lines) if lines else "Тарифы сейчас не найдены в базе."

def guest_access_text() -> str:
    ga = knowledge.get("guest_access", {})
    if not isinstance(ga, dict):
        return "Сейчас не вижу раздел гостевого доступа в базе."
    title = ga.get("title", "Гостевой доступ")
    desc = ga.get("description", "")
    website = ga.get("website", {})
    key_obj = ga.get("guest_key", {})

    site_line = ""
    if isinstance(website, dict):
        url = website.get("url")
        if url:
            site_line = f"Сайт: {url}"

    key_line = ""
    if isinstance(key_obj, dict):
        k = key_obj.get("key")
        v = key_obj.get("validity")
        if k:
            key_line = f"🔑 Гостевой ключ: `{k}`"
            if v:
                key_line += f" (действует {v})"

    parts = [f"**{title}**"]
    if desc:
        parts.append(desc.strip())
    if site_line:
        parts.append(site_line)
    if key_line:
        parts.append(key_line)

    return "\n\n".join(parts)


# =========================
# OPENAI FALLBACK (ONLY WITH YAML SNIPPETS)
# =========================
def build_system_prompt() -> str:
    assistant_name = kget("assistant.name", "Лиза")
    owner_name = kget("assistant.owner_name", "Юлия")
    project_name = kget("project.name", "INSTART")
    disclaim = kget("faq", [])
    # доход/гарантии иногда в faq — но главное правило:
    return f"""
Вы — “{assistant_name}”, ассистент куратора {owner_name} в онлайн-школе {project_name} и менеджер по продажам (не проговаривать).
Общение ТОЛЬКО на «Вы». Тон дружелюбный, тактичный, живой. Без давления.

КРИТИЧЕСКИ ВАЖНО:
- Факты (цены, состав, условия, лицензия, ссылки, длительность, медиа) берите ТОЛЬКО из предоставленного контекста YAML_SNIPPETS.
- Если чего-то нет в YAML_SNIPPETS — скажите, что уточните у куратора Юлии, и предложите оставить контакт.
- Не обещайте гарантированный доход.

ФОРМАТ:
- 1–6 коротких абзацев. Можно списки.
- В конце 1 вопрос (следующий шаг).
""".strip()

def make_yaml_snippets(user_text: str, user: Dict[str, Any]) -> str:
    """
    Даем модели НЕ весь YAML, а релевантные куски.
    """
    q = norm(user_text)

    snippets: Dict[str, Any] = {}

    # проект
    snippets["project"] = {
        "name": kget("project.name"),
        "mission": kget("project.mission"),
        "description": kget("project.description"),
        "license": kget("project.license"),
    }

    # гостевой доступ
    if any(w in q for w in ["гост", "ключ", "презентац", "доступ"]):
        snippets["guest_access"] = knowledge.get("guest_access", {})
        snippets["media_keys"] = list((knowledge.get("media") or {}).keys())

    # тариф/курс
    t = find_tariff(user_text)
    c = find_course(user_text)
    if t:
        snippets["matched_tariff"] = t
    if c:
        snippets["matched_course"] = c

    # если ранее уже предлагали что-то — подмешаем
    if user.get("last_suggested_type") and user.get("last_suggested_id"):
        snippets["last_suggested"] = {
            "type": user["last_suggested_type"],
            "id": user["last_suggested_id"],
        }

    # FAQ
    faq = knowledge.get("faq")
    if isinstance(faq, list) and faq:
        # возьмем 4-6 самых общих
        snippets["faq_sample"] = faq[:6]

    return json.dumps(snippets, ensure_ascii=False, indent=2)


async def call_openai(messages: List[Dict[str, str]]) -> str:
    if not openai_client:
        return ""
    def _sync():
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=260,
        )
        return (resp.choices[0].message.content or "").strip()
    return await asyncio.to_thread(_sync)


# =========================
# INTERNAL LEAD
# =========================
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{9,}\d)")
EMAIL_RE = re.compile(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})")

def extract_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text or "")
    if not m:
        return None
    return re.sub(r"[^\d+]", "", m.group(1))

def extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text or "")
    return m.group(1).strip() if m else None

def looks_like_email(s: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", (s or "").strip()))

BUY_INTENT_RE = re.compile(r"\b(купить|оплат(ить|а)|оформ(ить|им)|готов(а)?\s+купить|беру)\b", re.IGNORECASE)
THIS_OPTION_RE = re.compile(r"\b(этот|эта|эту|этим|для\s+себя|беру\s+этот|мне\s+этот)\b", re.IGNORECASE)


# =========================
# COMMANDS
# =========================
@dp.message(Command("reload"))
async def cmd_reload(message: Message):
    global knowledge, COURSES, TARIFFS, COURSE_INDEX, TARIFF_INDEX
    await send_typing(message.chat.id)
    knowledge = load_knowledge()
    COURSES = safe_list(knowledge.get("courses"))
    TARIFFS = safe_list(knowledge.get("tariffs"))
    COURSE_INDEX = build_index(COURSES)
    TARIFF_INDEX = build_index(TARIFFS)
    await message.answer("knowledge.yaml перечитан ✅")


@dp.message(Command("myid"))
async def cmd_myid(message: Message):
    await send_typing(message.chat.id)
    await message.answer(f"Ваш user_id: {message.from_user.id}\nchat_id: {message.chat.id}")


# =========================
# START
# =========================
@dp.message(CommandStart())
async def start(message: Message):
    user_id = message.from_user.id
    await send_typing(message.chat.id)
    await get_user(user_id)  # ensure exists
    await update_user(user_id, stage=Stage.ASK_NAME, goal=None, time_budget=None, chosen_type=None, chosen_id=None, chosen_title=None)
    await message.answer(
        "Здравствуйте! 😊\n\n"
        "Я Лиза — помощница куратора Юлии в онлайн-школе INSTART.\n"
        "Помогу подобрать курс и тариф под Вашу цель.\n\n"
        "Как я могу к Вам обращаться?"
    )


# =========================
# MAIN HANDLER
# =========================
@dp.message(F.text)
async def on_text(message: Message):
    user_id = message.from_user.id
    user = await get_user(user_id)

    text = (message.text or "").strip()
    if not text:
        return

    await add_history(user_id, "user", text)

    # typing before every reply
    await send_typing(message.chat.id)

    stage = user.get("stage") or Stage.ASK_NAME

    # 1) Ask name
    if stage == Stage.ASK_NAME:
        first, last = extract_name(text)
        if not first:
            await message.answer("Подскажите, пожалуйста, как я могу к Вам обращаться? 🙂")
            return

        sex = guess_sex_by_name(first)
        await update_user(user_id, first_name=first, last_name=last, sex=sex, stage=Stage.QUALIFY_GOAL)

        if sex is None:
            await message.answer(
                f"{first}, очень приятно познакомиться! 😊\n\n"
                "Подскажите, пожалуйста, как к Вам правильно обращаться — в мужском или женском роде?"
            )
            return

        await message.answer(
            f"{first}, очень приятно познакомиться! 😊\n\n"
            "Подскажите, пожалуйста, что Вам сейчас ближе?\n"
            "1) Подработка\n"
            "2) Новая онлайн-профессия\n"
            "3) Развитие в проекте (партнёрство/кураторство)\n\n"
            "Можно просто цифрой или словом."
        )
        return

    # 1.1) Clarify sex if needed
    if stage == Stage.QUALIFY_GOAL and user.get("sex") is None:
        t = norm(text)
        if "жен" in t:
            await update_user(user_id, sex="f")
        elif "муж" in t:
            await update_user(user_id, sex="m")
        else:
            await message.answer("Поняла 🙂 Скажите, пожалуйста: в мужском или женском роде?")
            return

        user = await get_user(user_id)
        await message.answer(
            "Спасибо! 😊\n\n"
            "Теперь подскажите, пожалуйста, что Вам сейчас ближе?\n"
            "1) Подработка\n"
            "2) Новая онлайн-профессия\n"
            "3) Развитие в проекте (партнёрство/кураторство)\n\n"
            "Можно просто цифрой или словом."
        )
        return

    # 2) Qualify goal
    if stage == Stage.QUALIFY_GOAL:
        t = norm(text)
        goal = None
        if t in {"1", "подработка"} or "подраб" in t:
            goal = "подработка"
        elif t in {"2"} or "професс" in t or "профес" in t:
            goal = "новая профессия"
        elif t in {"3"} or "развит" in t or "партнер" in t or "куратор" in t:
            goal = "развитие в проекте"

        if not goal:
            await message.answer(
                "Подскажите, пожалуйста, одним вариантом:\n"
                "1) Подработка\n"
                "2) Новая онлайн-профессия\n"
                "3) Развитие в проекте\n\n"
                "Можно цифрой 🙂"
            )
            return

        await update_user(user_id, goal=goal, stage=Stage.QUALIFY_TIME)

        await message.answer(
            f"Поняла Вас 🙂\n\n"
            "Сколько времени в неделю Вам реально удобно уделять обучению?\n"
            "Например: 2–3 часа / 5–7 часов / 10+ часов."
        )
        return

    # 3) Qualify time
    if stage == Stage.QUALIFY_TIME:
        await update_user(user_id, time_budget=text.strip(), stage=Stage.NORMAL)
        user = await get_user(user_id)

        # мягкий следующий шаг на основе цели
        goal = user.get("goal") or ""
        if "подработка" in goal:
            await message.answer(
                "Спасибо! 😊\n\n"
                "Для подработки чаще всего начинают с гостевого доступа — Вы бесплатно смотрите варианты заработка и выбираете, что ближе.\n\n"
                "Хотите, я дам гостевой ключ и сразу отправлю презентацию проекта?"
            )
            return
        if "новая профессия" in goal:
            await message.answer(
                "Отлично! 😊\n\n"
                "Чтобы подобрать направление под новую профессию, уточню один момент:\n"
                "Вам больше интересны нейросети, дизайн/инфографика, маркетплейсы или продвижение/реклама?"
            )
            return
        # развитие в проекте
        await message.answer(
            "Поняла 🙂\n\n"
            "Если цель — развитие в проекте, я могу подсказать, какие тарифы подходят для партнёрства/кураторства.\n\n"
            "Скажите, пожалуйста: Вы хотите больше про заработок куратора или про обучение + доступ к курсам?"
        )
        return

    # =========================
    # NORMAL MODE: scripted answers
    # =========================

    # A) Project / school info
    if any(w in norm(text) for w in ["школ", "проект", "инстарт", "instart", "о школе", "о проекте", "расскажи о школе"]):
        await message.answer(project_brief())
        await message.answer("Подскажите, пожалуйста, Ваша цель сейчас ближе к подработке или новой профессии?")
        return

    # B) Presentation request
    if "презентац" in norm(text):
        await message.answer("Сейчас отправлю презентацию проекта 📎")
        ok = await send_root_media_key(
            user_id,
            message,
            "презентация_проекта_с_призывом_хочу_гостевой_ключ",
            caption_override="Презентация проекта INSTART 📎"
        )
        if not ok:
            # fallback from guest_access
            fid = kget("guest_access.promo_materials.presentation_file_id") or kget("project.guest_access.presentation_file_id")
            if fid:
                await send_media_by_file_id(user_id, message, "video", str(fid), "Презентация проекта INSTART 📎")
            else:
                await message.answer("В базе не найден file_id презентации 🙈 Я уточню у куратора Юлии.")
        return

    # C) Guest access
    if any(w in norm(text) for w in ["гост", "ключ", "демо", "пробн"]):
        await message.answer(guest_access_text(), parse_mode="Markdown")

        # send promo materials if exist
        lay = kget("guest_access.promo_materials.guest_access_layout_file_id")
        pres = kget("guest_access.promo_materials.presentation_file_id")
        instr = kget("guest_access.activation_materials.instruction_file_id")
        memo = kget("guest_access.activation_materials.memo_file_id")

        if lay:
            await message.answer("Отправляю макет по гостевому доступу 📎")
            await send_media_by_file_id(user_id, message, "photo", str(lay), "Макет по гостевому доступу")
        if instr:
            await message.answer("Отправляю видео-инструкцию по регистрации и активации ✅")
            await send_media_by_file_id(user_id, message, "video", str(instr), "Инструкция по регистрации и активации ключа")
        if memo:
            await message.answer("И памятку, чтобы было удобно повторить шаги 🙂")
            await send_media_by_file_id(user_id, message, "photo", str(memo), "Памятка по регистрации и активации ключа")
        if pres:
            await message.answer("И презентацию проекта 📎")
            await send_media_by_file_id(user_id, message, "video", str(pres), "Презентация проекта INSTART")

        await message.answer("Хотите, я подскажу 1–2 направления под Вашу цель, чтобы было проще выбрать?")
        return

    # D) Tariff lookup
    if "тариф" in norm(text):
        t = find_tariff(text)
        if t:
            await update_user(user_id, last_suggested_type="tariff", last_suggested_id=str(t.get("id")))
            await message.answer(format_tariff(t), parse_mode="Markdown")
            # send media refs if exist
            await send_item_media_refs(user_id, message, t)
            await message.answer("Подскажите, пожалуйста: Вы рассматриваете этот тариф для себя или хотите сравнить с ещё 1–2?")
            return
        # if asked tariffs list
        if any(w in norm(text) for w in ["какие", "все", "список", "есть", "тарифы"]):
            await message.answer("Вот актуальные тарифы и цены 🙂\n\n" + tariffs_brief())
            await message.answer("Какую цель Вы решаете сейчас — подработка, новая профессия или развитие в проекте?")
            return

    # E) Course lookup
    if any(w in norm(text) for w in ["курс", "нейросет", "маркетплейс", "ozon", "wildberries", "вб"]):
        c = find_course(text)
        if c:
            await update_user(user_id, last_suggested_type="course", last_suggested_id=str(c.get("id")))
            await message.answer(format_course(c), parse_mode="Markdown")
            await send_item_media_refs(user_id, message, c)
            await message.answer("Подскажите, пожалуйста: Вы рассматриваете этот курс для себя или хотите сравнить с ещё 1–2 вариантами?")
            return

    # F) “этот вариант” = user confirms last suggested
    if THIS_OPTION_RE.search(text):
        last_type = user.get("last_suggested_type")
        last_id = user.get("last_suggested_id")
        if last_type and last_id:
            # find object by id
            chosen = None
            if last_type == "course":
                for x in COURSES:
                    if str(x.get("id")) == str(last_id):
                        chosen = x
                        break
            elif last_type == "tariff":
                for x in TARIFFS:
                    if str(x.get("id")) == str(last_id):
                        chosen = x
                        break

            if chosen:
                await update_user(
                    user_id,
                    chosen_type=last_type,
                    chosen_id=str(chosen.get("id")),
                    chosen_title=str(chosen.get("title")),
                )
                await message.answer(
                    f"Отлично 🙂 Зафиксировала: **{chosen.get('title')}**.\n\n"
                    "Хотите, я помогу оформить заявку на покупку? Тогда попрошу коротко 3 контакта.",
                    parse_mode="Markdown",
                )
                return

        await message.answer("Поняла 🙂 Уточните, пожалуйста: Вы про какой курс или тариф? Напишите название — я найду по базе.")
        return

    # G) Buy intent -> collect contacts (only if chosen exists)
    if BUY_INTENT_RE.search(text):
        chosen_title = user.get("chosen_title")
        if not chosen_title:
            await message.answer(
                "Конечно 🙂\n\n"
                "Скажите, пожалуйста, какой именно курс или тариф Вы выбрали (название) — и я оформлю заявку."
            )
            return
        await update_user(user_id, stage=Stage.COLLECT_CONTACTS)
        await message.answer(
            "Хорошо 🙂 Чтобы оформить заявку, напишите одним сообщением:\n"
            "1) Фамилия Имя\n"
            "2) Телефон\n"
            "3) E-mail\n"
            f"4) Выбранный курс/тариф: {chosen_title}\n\n"
            "После этого я передам заявку во внутренний чат."
        )
        return

    if stage == Stage.COLLECT_CONTACTS:
        first, last = extract_name(text)
        phone = extract_phone(text)
        email = extract_email(text)

        # last step: ensure chosen exists
        chosen_title = user.get("chosen_title")
        if not chosen_title:
            # try parse from message
            t = find_tariff(text)
            c = find_course(text)
            if t:
                chosen_title = t.get("title")
                await update_user(user_id, chosen_type="tariff", chosen_id=str(t.get("id")), chosen_title=str(chosen_title))
            elif c:
                chosen_title = c.get("title")
                await update_user(user_id, chosen_type="course", chosen_id=str(c.get("id")), chosen_title=str(chosen_title))

        missing = []
        if not (first and last):
            missing.append("Фамилия Имя")
        if not phone or len(re.sub(r"\D", "", phone)) < 10:
            missing.append("телефон")
        if not email or not looks_like_email(email):
            missing.append("e-mail")
        if not chosen_title:
            missing.append("курс/тариф (название)")

        if missing:
            await message.answer("Мне не хватает: " + ", ".join(missing) + " 🙂\nНапишите, пожалуйста, одним сообщением.")
            return

        await update_user(user_id, first_name=first, last_name=last)

        # build strict lead format
        user = await get_user(user_id)
        sex_label = user.get("sex") or "не определён"
        goal = user.get("goal") or "—"
        now_str = time.strftime("%Y-%m-%d %H:%M", time.localtime())

        lead = (
            "🟩 ЗАЯВКА (INSTART)\n"
            f"Имя клиента: {first}\n"
            f"Пол: {sex_label}\n"
            f"Фамилия Имя: {last} {first}\n"
            f"Телефон: {phone}\n"
            f"Email: {email}\n"
            f"Курс/Тариф: {chosen_title}\n"
            f"Источник: Telegram\n"
            f"Краткий запрос/цель: {goal}\n"
            f"Важные детали/возражения: —\n"
            f"Дата/время: {now_str}\n"
            f"User ID: {user_id}"
        )

        try:
            await bot.send_message(INTERNAL_CHAT_ID_INT, lead)
        except Exception as e:
            log.exception("Failed to send lead: %s", e)
            await message.answer("Я не смогла отправить заявку во внутренний чат 🙈 Я уточню у куратора Юлии и вернусь к Вам.")
            await update_user(user_id, stage=Stage.NORMAL)
            return

        await message.answer(
            "Спасибо! 😊 Я передала заявку.\n"
            "Куратор Юлия свяжется с Вами и подскажет дальнейшие шаги."
        )
        await update_user(user_id, stage=Stage.NORMAL)
        return

    # =========================
    # OPENAI fallback (only if needed)
    # =========================
    if openai_client:
        hist = await get_history(user_id, limit=12)
        snippets = make_yaml_snippets(text, user)

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "system", "content": f"YAML_SNIPPETS:\n{snippets}"},
        ]
        messages.extend(hist[-10:])
        messages.append({"role": "user", "content": text})

        try:
            answer = await call_openai(messages)
            if not answer:
                raise RuntimeError("Empty OpenAI answer")

            await message.answer(answer)
            await add_history(user_id, "assistant", answer)
            return
        except Exception as e:
            log.exception("OpenAI error: %s", e)

    # If no OpenAI or it failed:
    await message.answer(
        "Я хочу ответить максимально точно, но в базе не нашла это в явном виде 🙈\n\n"
        "Скажите, пожалуйста, что именно интересует: курс, тариф, гостевой доступ или информация о проекте?"
    )


# =========================
# WEBHOOK APP
# =========================
async def on_startup(app: web.Application):
    await db_init()
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
