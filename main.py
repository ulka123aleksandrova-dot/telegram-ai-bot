import os
import re
import json
import time
import yaml
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart, Command
from aiogram.enums import ChatAction
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram.utils.keyboard import InlineKeyboardBuilder

import aiosqlite

# OpenAI optional (бот может жить и без него, если не поднялся клиент)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =========================
# BOOT / ENV
# =========================
load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")  # https://xxxx.up.railway.app
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/tg/webhook")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me")

# ВАЖНО: именно INTERNAL_CHAT_ID (внутр. группа “INSTART заявки”)
INTERNAL_CHAT_ID = os.getenv("INTERNAL_CHAT_ID")

PORT = int(os.getenv("PORT", "8080"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN в переменных окружения.")
if not WEBHOOK_BASE:
    raise RuntimeError("Не найден WEBHOOK_BASE в переменных окружения.")
if not INTERNAL_CHAT_ID:
    raise RuntimeError("Не найден INTERNAL_CHAT_ID в переменных окружения.")

INTERNAL_CHAT_ID_INT = int(INTERNAL_CHAT_ID)

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# OpenAI client init (может не подняться из-за конфликтов httpx/proxies — тогда просто выключим AI)
client = None
if OpenAI and OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except TypeError as e:
        # типовая ошибка при несовместимом httpx (например 0.28+) => pin httpx==0.27.2
        log.warning("OpenAI init failed: %s", e)
        client = None
    except Exception as e:
        log.warning("OpenAI init failed: %s", e)
        client = None


# =========================
# KNOWLEDGE
# =========================
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge.yaml")


def load_knowledge() -> Dict[str, Any]:
    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        log.error("knowledge.yaml не найден рядом с main.py")
        return {}
    except Exception as e:
        log.exception("Ошибка чтения knowledge.yaml: %s", e)
        return {}

    if data is None:
        return {}
    if not isinstance(data, dict):
        log.error("knowledge.yaml должен быть словарём (mapping) в корне.")
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


def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    return s


# ====== структуры из вашего YAML ======
def get_project() -> Dict[str, Any]:
    return kget("project", {}) if isinstance(kget("project", {}), dict) else {}


def get_media() -> Dict[str, Any]:
    m = kget("media", {})
    return m if isinstance(m, dict) else {}


def get_guest_access() -> Dict[str, Any]:
    ga = kget("guest_access", {})
    return ga if isinstance(ga, dict) else {}


def get_tariffs() -> List[Dict[str, Any]]:
    t = kget("tariffs", [])
    return t if isinstance(t, list) else []


def get_courses() -> List[Dict[str, Any]]:
    c = kget("courses", [])
    return c if isinstance(c, list) else []


def get_faq() -> List[Dict[str, str]]:
    f = kget("faq", [])
    return f if isinstance(f, list) else []


ASSISTANT_NAME = kget("assistant.name", "Лиза")
OWNER_NAME = kget("assistant.owner_name", "Юлии")  # в тексте лучше "куратора Юлии"
PROJECT_NAME = kget("project.name", "INSTART")


# =========================
# SQLITE STORAGE
# =========================
DB_PATH = os.path.join(os.path.dirname(__file__), "bot.sqlite3")

DEFAULT_HISTORY_TURNS = 10


class Stage:
    ASK_NAME = "ask_name"
    DISCOVERY = "discovery"
    NORMAL = "normal"
    CHOSEN = "chosen"       # пользователь выбрал курс/тариф
    LEAD_COLLECT = "lead_collect"


async def db_init():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                stage TEXT,
                profile_json TEXT,
                chosen_json TEXT,
                sent_media_json TEXT,
                history_json TEXT,
                updated_at INTEGER
            )
            """
        )
        await db.commit()


def _now() -> int:
    return int(time.time())


async def db_get_user(user_id: int) -> Dict[str, Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT stage, profile_json, chosen_json, sent_media_json, history_json FROM users WHERE user_id=?",
            (user_id,),
        )
        row = await cur.fetchone()

    if not row:
        return {
            "user_id": user_id,
            "stage": Stage.ASK_NAME,
            "profile": {"first_name": None, "sex": None},
            "chosen": {"type": None, "id": None, "title": None},
            "sent_media": [],
            "history": [],
        }

    stage, profile_json, chosen_json, sent_media_json, history_json = row
    return {
        "user_id": user_id,
        "stage": stage or Stage.ASK_NAME,
        "profile": json.loads(profile_json or "{}"),
        "chosen": json.loads(chosen_json or "{}"),
        "sent_media": json.loads(sent_media_json or "[]"),
        "history": json.loads(history_json or "[]"),
    }


async def db_save_user(state: Dict[str, Any]) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO users(user_id, stage, profile_json, chosen_json, sent_media_json, history_json, updated_at)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET
                stage=excluded.stage,
                profile_json=excluded.profile_json,
                chosen_json=excluded.chosen_json,
                sent_media_json=excluded.sent_media_json,
                history_json=excluded.history_json,
                updated_at=excluded.updated_at
            """,
            (
                state["user_id"],
                state.get("stage", Stage.ASK_NAME),
                json.dumps(state.get("profile", {}), ensure_ascii=False),
                json.dumps(state.get("chosen", {}), ensure_ascii=False),
                json.dumps(state.get("sent_media", []), ensure_ascii=False),
                json.dumps(state.get("history", []), ensure_ascii=False),
                _now(),
            ),
        )
        await db.commit()


def add_history(state: Dict[str, Any], role: str, content: str) -> None:
    hist = state.get("history", [])
    hist.append({"role": role, "content": content})
    # ограничим историю
    max_msgs = DEFAULT_HISTORY_TURNS * 2 + 2
    if len(hist) > max_msgs:
        hist = hist[-max_msgs:]
    state["history"] = hist


# =========================
# NAME / SEX
# =========================
NAME_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё\-]{2,}")

def extract_name(text: str) -> Optional[str]:
    """
    Умеет вытащить имя из:
    - "меня зовут марина"
    - "привет! я марина, хочу..."
    - "марина"
    Не воспринимает слова-цели ("подработка") как имя.
    """
    t = (text or "").strip()
    if not t:
        return None

    t_norm = norm(t)
    # если человек вместо имени пишет цель — не считаем это именем
    if any(w in t_norm for w in ["подработка", "профес", "партнер", "партн", "развитие"]):
        return None

    m = re.search(r"(?:меня\s+зовут|я)\s+([A-Za-zА-Яа-яЁё\-]{2,})", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().capitalize()

    # если сообщение короткое — 1 слово
    words = NAME_WORD_RE.findall(t)
    if len(words) >= 1 and len(t.split()) <= 3:
        # первое слово и сделаем с заглавной
        return words[0].strip().capitalize()

    return None


def guess_sex_by_name(name: str) -> Optional[str]:
    n = norm(name)
    if not n:
        return None
    # простая эвристика
    if n.endswith(("а", "я")) and n not in {"илья", "никита"}:
        return "f"
    return "m"


def verb_variant(sex: Optional[str], m: str, f: str) -> str:
    return f if sex == "f" else m


# =========================
# INTENTS / SEARCH IN YAML
# =========================
def is_presentation_request(text: str) -> bool:
    return "презентац" in norm(text)

def is_guest_request(text: str) -> bool:
    t = norm(text)
    return any(x in t for x in ["гост", "ключ", "демо", "пробн", "бесплатн"])

def is_tariffs_list_request(text: str) -> bool:
    t = norm(text)
    return "тариф" in t and any(x in t for x in ["какие", "список", "есть", "все", "стоим", "цена", "цены", "сколько"])

def is_school_request(text: str) -> bool:
    t = norm(text)
    return any(x in t for x in ["о школе", "о проекте", "что такое инстарт", "расскажи о школе", "расскажи про проект", "расскажи о проекте", "instart"])

def is_buy_intent(text: str) -> bool:
    return bool(re.search(r"\b(купить|оплат(ить|а)|готов(а)? купить|хочу оформить|оформим|беру)\b", text, flags=re.IGNORECASE))


def find_tariff(query: str) -> Optional[Dict[str, Any]]:
    q = norm(query)
    for t in get_tariffs():
        title = norm(str(t.get("title", "")))
        aliases = [norm(a) for a in (t.get("aliases") or []) if isinstance(a, str)]
        if title and (title in q or q in title):
            return t
        for a in aliases:
            if a and (a in q or q in a):
                return t
    return None


def find_course(query: str) -> Optional[Dict[str, Any]]:
    q = norm(query)
    for c in get_courses():
        title = norm(str(c.get("title", "")))
        aliases = [norm(a) for a in (c.get("aliases") or []) if isinstance(a, str)]
        if title and (title in q or q in title):
            return c
        for a in aliases:
            if a and (a in q or q in a):
                return c
    return None


def find_faq_answer(query: str) -> Optional[str]:
    q = norm(query)
    for item in get_faq():
        qq = norm(str(item.get("q", "")))
        if qq and (qq in q or q in qq):
            ans = item.get("a")
            if isinstance(ans, str) and ans.strip():
                return ans.strip()
    return None


# =========================
# MEDIA SENDER
# =========================
async def send_typing(chat_id: int):
    try:
        await bot.send_chat_action(chat_id, ChatAction.TYPING)
    except Exception:
        pass


async def send_media_file_id(message: Message, state: Dict[str, Any], media_type: str, file_id: str, caption: str) -> bool:
    """
    media_type: photo/video/document
    не отправляем повторно один и тот же file_id
    """
    sent = set(state.get("sent_media", []) or [])
    if file_id in sent:
        await message.answer("Я уже отправляла это ранее 🙂 Посмотрите, пожалуйста, выше в чате — файл там сохранён.")
        return False

    cap = (caption or "").strip()
    cap = cap[:1024] if cap else None

    try:
        if media_type == "photo":
            await message.answer_photo(photo=file_id, caption=cap)
        elif media_type == "video":
            await message.answer_video(video=file_id, caption=cap)
        elif media_type == "document":
            await message.answer_document(document=file_id, caption=cap)
        else:
            return False
    except Exception as e:
        log.exception("Failed to send media: %s", e)
        await message.answer("Не получилось отправить файл 😕 Я передам это куратору Юлии.")
        return False

    sent.add(file_id)
    state["sent_media"] = list(sent)
    return True


def get_media_by_key(key: str) -> Optional[Dict[str, Any]]:
    m = get_media()
    item = m.get(key)
    return item if isinstance(item, dict) else None


# =========================
# RELEVANT CONTEXT BUILDER FOR AI
# =========================
def build_relevant_context(user_text: str, state: Dict[str, Any]) -> str:
    """
    ВАЖНО: в модель передаём только релевантное, не весь YAML.
    """
    parts: List[str] = []

    # проект
    if is_school_request(user_text):
        proj = get_project()
        if proj:
            parts.append("PROJECT:\n" + yaml.safe_dump(proj, allow_unicode=True, sort_keys=False))

    # если нашли курс/тариф — добавим его
    tar = find_tariff(user_text)
    if tar:
        parts.append("TARIFF:\n" + yaml.safe_dump(tar, allow_unicode=True, sort_keys=False))

    course = find_course(user_text)
    if course:
        parts.append("COURSE:\n" + yaml.safe_dump(course, allow_unicode=True, sort_keys=False))

    # гостевой
    if is_guest_request(user_text):
        ga = get_guest_access()
        if ga:
            parts.append("GUEST_ACCESS:\n" + yaml.safe_dump(ga, allow_unicode=True, sort_keys=False))

    # презентация (media)
    if is_presentation_request(user_text):
        m = get_media()
        # кинем только ключи, чтобы модель знала, что есть
        parts.append("MEDIA_KEYS:\n" + ", ".join(list(m.keys())[:50]))

    # FAQ
    ans = find_faq_answer(user_text)
    if ans:
        parts.append("FAQ_MATCH:\n" + ans)

    # краткий список тарифов, если вопрос про тарифы в целом
    if "тариф" in norm(user_text):
        lines = []
        for t in get_tariffs():
            title = t.get("title")
            price = t.get("price_rub")
            if title and price:
                lines.append(f"- {title}: {price} ₽")
        if lines:
            parts.append("TARIFFS_LIST:\n" + "\n".join(lines))

    return "\n\n".join(parts).strip()


def build_system_prompt(state: Dict[str, Any]) -> str:
    prof = state.get("profile", {}) or {}
    name = prof.get("first_name") or "пожалуйста"
    sex = prof.get("sex")

    rules = f"""
Вы — «{ASSISTANT_NAME}», ассистент куратора Юлии в онлайн-школе {PROJECT_NAME}.
Общение ТОЛЬКО на «Вы». Тон дружелюбный, тактичный, живой. Без давления.

КРИТИЧЕСКИ ВАЖНО:
1) Факты берите ТОЛЬКО из предоставленного контекста KNOWLEDGE_SNIPPET.
2) Если в KNOWLEDGE_SNIPPET нет ответа — скажите, что уточните у куратора Юлии, и предложите оставить контакт.
3) Не выдумывайте курсы/тарифы/цены/условия.
4) Не обещайте гарантированный доход.

ФОРМАТ:
- 1–6 коротких абзацев, списки уместны.
- В конце 1 вопрос (следующий шаг).
- Обращайтесь к клиенту по имени: {name}.
- Согласуйте род: {("женский" if sex=="f" else "мужской" if sex=="m" else "неизвестен — избегайте родовых форм или уточните")}.

ЗАДАЧА:
Понять цель клиента, подобрать 1–3 варианта из контекста, мягко вести к выбору и оформлению заявки.
""".strip()
    return rules


# =========================
# LEAD (INTERNAL CHAT)
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


def format_lead(state: Dict[str, Any], user_text: str) -> str:
    prof = state.get("profile", {}) or {}
    chosen = state.get("chosen", {}) or {}
    now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    return (
        "🟩 ЗАЯВКА (INSTART)\n"
        f"Имя клиента: {prof.get('first_name') or '—'}\n"
        f"Пол: {prof.get('sex') or '—'}\n"
        f"Фамилия Имя: {prof.get('fio') or '—'}\n"
        f"Телефон: {prof.get('phone') or '—'}\n"
        f"Email: {prof.get('email') or '—'}\n"
        f"Курс/Тариф: {chosen.get('title') or '—'}\n"
        f"Источник: Telegram\n"
        f"Запрос/цель: {state.get('goal') or user_text[:200]}\n"
        f"Детали/возражения: {state.get('notes') or '—'}\n"
        f"Дата/время: {now_str}\n"
        f"User ID: {state.get('user_id')}\n"
    )


async def send_internal_lead(text: str) -> None:
    try:
        await bot.send_message(INTERNAL_CHAT_ID_INT, text)
    except Exception as e:
        log.exception("Failed to send lead to internal chat: %s", e)


# =========================
# SCRIPTED ANSWERS FROM YAML
# =========================
def format_project_info() -> str:
    proj = get_project()
    if not proj:
        return "В базе сейчас нет заполненной информации о проекте. Я уточню у куратора Юлии и вернусь к Вам."

    desc = proj.get("description") or ""
    mission = proj.get("mission") or ""
    founded = (proj.get("founded") or {}).get("date") if isinstance(proj.get("founded"), dict) else None
    license_info = proj.get("license") or {}
    lic_number = license_info.get("license_number") if isinstance(license_info, dict) else None
    lic_date = license_info.get("license_date") if isinstance(license_info, dict) else None

    lines = []
    if isinstance(desc, str) and desc.strip():
        lines.append(desc.strip())
    if isinstance(mission, str) and mission.strip():
        lines.append(f"Миссия: {mission.strip()}")
    if founded:
        lines.append(f"Основан: {founded}")
    if lic_number and lic_date:
        lines.append(f"Лицензия: № {lic_number} от {lic_date}")

    # что дальше — вопрос
    lines.append("Подскажите, пожалуйста, что Вам сейчас важнее: подработка, новая профессия или развитие в проекте?")
    return "\n\n".join(lines).strip()


def format_tariffs_list() -> str:
    lines = ["Актуальные тарифы:"]
    for t in get_tariffs():
        title = t.get("title")
        price = t.get("price_rub")
        if title and price:
            lines.append(f"• {title} — {price} ₽")
    if len(lines) == 1:
        return "В базе пока не вижу списка тарифов. Я уточню у куратора Юлии."
    lines.append("\nКакой тариф хотите посмотреть подробнее? (можно написать название, например «Премиум»)")
    return "\n".join(lines)


def format_tariff_detail(t: Dict[str, Any]) -> str:
    title = t.get("title", "Тариф")
    price = t.get("price_rub")
    about = t.get("short_about") or ""
    who_for = t.get("who_for") or []
    main_courses = t.get("main_courses") or []
    advantages = t.get("advantages") or []

    lines = [f"Тариф «{title}»"]
    if price:
        lines.append(f"Цена: {price} ₽.")
    if isinstance(about, str) and about.strip():
        lines.append(about.strip())

    if isinstance(who_for, list) and who_for:
        lines.append("Кому подходит:")
        for x in who_for[:5]:
            lines.append(f"• {x}")

    if isinstance(main_courses, list) and main_courses:
        lines.append("Что внутри (примеры направлений):")
        for x in main_courses[:8]:
            lines.append(f"• {x}")

    if isinstance(advantages, list) and advantages:
        lines.append("Плюсы:")
        for x in advantages[:5]:
            lines.append(f"• {x}")

    lines.append("Хотите, помогу понять, подходит ли Вам этот тариф? Сколько времени готовы уделять в неделю?")
    return "\n".join(lines)


def format_course_detail(c: Dict[str, Any]) -> str:
    title = c.get("title", "Курс")
    price = c.get("price") if isinstance(c.get("price"), dict) else {}
    price_with = price.get("with_chat_rub")
    price_without = price.get("without_chat_rub")

    sd = c.get("short_description") or ""
    suitable_for = c.get("suitable_for") or []
    results = c.get("results_after_course") or []
    notes = c.get("important_notes") or []

    lines = [f"Курс «{title}»"]
    if price_with and price_without and price_with != price_without:
        lines.append(f"Цена: с чатом — {price_with} ₽, без чата — {price_without} ₽.")
    elif price_with:
        lines.append(f"Цена: {price_with} ₽.")
    elif price_without:
        lines.append(f"Цена: {price_without} ₽.")

    if isinstance(sd, str) and sd.strip():
        lines.append(sd.strip())

    if isinstance(suitable_for, list) and suitable_for:
        lines.append("Кому подходит:")
        for x in suitable_for[:5]:
            lines.append(f"• {x}")

    if isinstance(results, list) and results:
        lines.append("Результат после курса:")
        for x in results[:5]:
            lines.append(f"• {x}")

    if isinstance(notes, list) and notes:
        lines.append("Важно:")
        for x in notes[:3]:
            lines.append(f"• {x}")

    lines.append("Рассматриваете этот курс для себя или хотите сравнить с ещё 1–2 вариантами?")
    return "\n".join(lines)


def format_guest_access() -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    Возвращает:
    - текст
    - список медиа к отправке: (type, file_id, caption)
    """
    ga = get_guest_access()
    if not ga:
        return ("Сейчас в базе нет данных по гостевому доступу. Я уточню у куратора Юлии.", [])

    title = ga.get("title") or "Гостевой доступ"
    desc = ga.get("description") or ""
    website = ga.get("website") or {}
    url = website.get("url") if isinstance(website, dict) else None

    guest_key = ga.get("guest_key") or {}
    key = guest_key.get("key") if isinstance(guest_key, dict) else None
    validity = guest_key.get("validity") if isinstance(guest_key, dict) else None

    lines = [f"{title}"]
    if isinstance(desc, str) and desc.strip():
        lines.append(desc.strip())
    if url:
        lines.append(f"Сайт: {url}")
    if key:
        if validity:
            lines.append(f"🔑 Ключ (действует {validity}): `{key}`")
        else:
            lines.append(f"🔑 Ключ: `{key}`")

    # медиа из guest_access + из глобального media
    to_send: List[Tuple[str, str, str]] = []

    promo = ga.get("promo_materials") or {}
    if isinstance(promo, dict):
        layout_id = promo.get("guest_access_layout_file_id")
        pres_id = promo.get("presentation_file_id")
        if layout_id:
            to_send.append(("photo", str(layout_id), "Отправляю макет по гостевому доступу 📎"))
        if pres_id:
            to_send.append(("video", str(pres_id), "Отправляю презентацию проекта 📎"))

    act = ga.get("activation_materials") or {}
    if isinstance(act, dict):
        instr_id = act.get("instruction_file_id")
        memo_id = act.get("memo_file_id")
        if memo_id:
            to_send.append(("photo", str(memo_id), "Памятка по регистрации и активации ключа ✅"))
        if instr_id:
            to_send.append(("video", str(instr_id), "Видео-инструкция по регистрации и активации ✅"))

    lines.append("Хотите, я подскажу 1–2 направления под Вашу цель, чтобы было проще выбрать?")
    return ("\n\n".join(lines), to_send)


def format_presentation_media() -> Optional[Tuple[str, str, str]]:
    """
    Берём презентацию из knowledge.media по ключу:
    'презентация_проекта_с_призывом_хочу_гостевой_ключ'
    """
    m = get_media_by_key("презентация_проекта_с_призывом_хочу_гостевой_ключ")
    if not m:
        return None
    mtype = m.get("type")
    fid = m.get("file_id")
    title = m.get("title") or "Презентация проекта"
    if not mtype or not fid:
        return None
    return (str(mtype), str(fid), str(title))


# =========================
# OPENAI (fallback)
# =========================
async def ai_answer(user_text: str, state: Dict[str, Any]) -> Optional[str]:
    """
    AI только когда:
    - есть клиент
    - и мы уже собрали релевантную выжимку
    """
    if not client:
        return None

    snippet = build_relevant_context(user_text, state)
    if not snippet:
        # если выжимки нет — лучше честно, чем галлюцинации
        return None

    sys = build_system_prompt(state)
    messages = [
        {"role": "system", "content": sys},
        {"role": "system", "content": f"KNOWLEDGE_SNIPPET:\n{snippet}"},
    ]

    # добавим историю (коротко)
    hist = state.get("history", []) or []
    for h in hist[-DEFAULT_HISTORY_TURNS * 2 :]:
        if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str):
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": user_text})

    def _call():
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=350,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        log.exception("OpenAI error: %s", e)
        return None


# =========================
# COMMANDS
# =========================
@dp.message(Command("myid"))
async def cmd_myid(message: Message):
    await message.answer(f"Ваш user_id: {message.from_user.id}\nchat_id: {message.chat.id}")


@dp.message(Command("reload"))
async def cmd_reload(message: Message):
    global knowledge
    knowledge = load_knowledge()
    await message.answer("knowledge.yaml перечитан ✅")


# =========================
# START
# =========================
@dp.message(CommandStart())
async def start(message: Message):
    await send_typing(message.chat.id)

    state = await db_get_user(message.from_user.id)
    state["stage"] = Stage.ASK_NAME
    add_history(state, "assistant", "start")
    await db_save_user(state)

    await message.answer(
        "Здравствуйте! 😊\n\n"
        f"Я {ASSISTANT_NAME} — помощница куратора Юлии в онлайн-школе {PROJECT_NAME}.\n"
        "Помогу подобрать курс и тариф под Вашу цель.\n\n"
        "Как я могу к Вам обращаться?"
    )


# =========================
# CALLBACKS (если понадобится)
# =========================
@dp.callback_query()
async def cb_any(cb: CallbackQuery):
    await cb.answer()


# =========================
# MAIN TEXT HANDLER
# =========================
@dp.message(F.text)
async def on_text(message: Message):
    await send_typing(message.chat.id)

    user_text = (message.text or "").strip()
    if not user_text:
        return

    state = await db_get_user(message.from_user.id)
    prof = state.get("profile", {}) or {}

    # --- ASK NAME ---
    if state.get("stage") == Stage.ASK_NAME:
        name = extract_name(user_text)
        if not name:
            await message.answer("Подскажите, пожалуйста, как я могу к Вам обращаться? 🙂")
            return

        prof["first_name"] = name
        sex = guess_sex_by_name(name)
        prof["sex"] = sex
        state["profile"] = prof
        state["stage"] = Stage.DISCOVERY

        await db_save_user(state)

        await message.answer(
            f"{name}, очень приятно познакомиться! 😊\n\n"
            "Подскажите, пожалуйста, что Вам сейчас ближе?\n"
            "1) Подработка\n"
            "2) Новая онлайн-профессия\n"
            "3) Развитие в проекте (партнёрство/кураторство)\n\n"
            "Можно просто цифрой или словами."
        )
        return

    # --- DISCOVERY: ответ на 1/2/3 или словами ---
    if state.get("stage") == Stage.DISCOVERY:
        t = norm(user_text)
        goal = None
        if t in {"1", "подработка"} or "подработ" in t:
            goal = "подработка"
        elif t in {"2"} or "профес" in t:
            goal = "новая профессия"
        elif t in {"3"} or "партнер" in t or "партн" in t or "куратор" in t:
            goal = "развитие в проекте"

        if goal:
            state["goal"] = goal
            state["stage"] = Stage.NORMAL
            await db_save_user(state)

            # мягкая связка с YAML
            if goal == "подработка":
                await message.answer(
                    "Поняла Вас 🙂\n\n"
                    "Чтобы подсказать самый удобный старт, уточню один момент:\n"
                    "Сколько времени в неделю Вы реально готовы уделять? (примерно)"
                )
            elif goal == "новая профессия":
                # можно подсказать 2-3 популярные направления из guest_access.preview
                preview = get_guest_access().get("popular_directions_preview") if isinstance(get_guest_access(), dict) else None
                extra = ""
                if isinstance(preview, list) and preview:
                    extra = "\n\nНапример, у нас популярны направления:\n" + "\n".join([f"• {x}" for x in preview[:5]])
                await message.answer(
                    "Отлично 🙂 Освоение новой профессии — самый сильный вариант для роста дохода.\n"
                    "Подскажите, пожалуйста, какое направление Вам ближе: нейросети, маркетплейсы, дизайн/инфографика, продвижение, тексты?"
                    + extra
                )
            else:
                await message.answer(
                    "Поняла 🙂\n\n"
                    "Если рассматривать развитие в проекте (партнёрство/кураторство), важно понять стартовые условия.\n"
                    "Подскажите, пожалуйста: у Вас уже есть блог/соцсеть или начинаем с нуля?"
                )
            return

        # если не распознали — попросим ещё раз, но вежливо
        await message.answer(
            "Подскажите, пожалуйста, какой вариант Вам ближе — 1, 2 или 3?\n"
            "Можно цифрой 🙂"
        )
        return

    # --- COMMON: FAQ quick ---
    faq_ans = find_faq_answer(user_text)
    if faq_ans:
        add_history(state, "user", user_text)
        add_history(state, "assistant", faq_ans)
        await db_save_user(state)
        await message.answer(faq_ans + "\n\nПодскажите, пожалуйста, что для Вас сейчас важнее при выборе: срок, бюджет или поддержка?")
        return

    # --- PRESENTATION ---
    if is_presentation_request(user_text):
        media = format_presentation_media()
        if media:
            mtype, fid, title = media
            await message.answer("Сейчас отправлю презентацию проекта 📎")
            await send_media_file_id(message, state, mtype, fid, title)
            await db_save_user(state)
        else:
            await message.answer(
                "Сейчас не вижу презентацию в базе 🙈\n"
                "Скажите, пожалуйста, что именно хотите узнать про INSTART: подработка, профессия или партнёрство?"
            )
        return

    # --- GUEST ACCESS ---
    if is_guest_request(user_text):
        text, media_list = format_guest_access()
        await message.answer(text, parse_mode="Markdown")

        # отправим материалы по одному (без повторов)
        for (mtype, fid, cap) in media_list:
            await send_media_file_id(message, state, mtype, fid, cap)

        await db_save_user(state)
        return

    # --- SCHOOL/PROJECT INFO ---
    if is_school_request(user_text):
        answer = format_project_info()
        add_history(state, "user", user_text)
        add_history(state, "assistant", answer)
        await db_save_user(state)
        await message.answer(answer)
        return

    # --- TARIFFS LIST ---
    if is_tariffs_list_request(user_text):
        answer = format_tariffs_list()
        add_history(state, "user", user_text)
        add_history(state, "assistant", answer)
        await db_save_user(state)
        await message.answer(answer)
        return

    # --- TARIFF DETAIL ---
    t = find_tariff(user_text)
    if t:
        detail = format_tariff_detail(t)
        # запомним выбор
        state["chosen"] = {"type": "tariff", "id": t.get("id"), "title": t.get("title")}
        state["stage"] = Stage.CHOSEN

        await message.answer(detail)

        # если есть media_refs (как у тарифа Премиум)
        media_refs = t.get("media_refs")
        if isinstance(media_refs, dict):
            # отправим первый попавшийся файл
            for _, v in media_refs.items():
                if isinstance(v, dict) and v.get("type") and v.get("file_id"):
                    await message.answer("Сейчас отправлю макет по этому тарифу 📎")
                    await send_media_file_id(message, state, str(v["type"]), str(v["file_id"]), str(v.get("title") or "Материалы"))
                    break

        await db_save_user(state)
        return

    # --- COURSE DETAIL ---
    c = find_course(user_text)
    if c:
        detail = format_course_detail(c)
        state["chosen"] = {"type": "course", "id": c.get("id"), "title": c.get("title")}
        state["stage"] = Stage.CHOSEN

        await message.answer(detail)

        # курс имеет media (как в вашем YAML)
        media = c.get("media")
        if isinstance(media, dict) and media.get("type") and media.get("file_id"):
            await message.answer("Сейчас отправлю макет/материалы по этому курсу 📎")
            await send_media_file_id(message, state, str(media["type"]), str(media["file_id"]), str(media.get("title") or "Материалы"))

        await db_save_user(state)
        return

    # --- AFTER "этот вариант" ---
    if norm(user_text) in {"этот вариант", "этот", "да этот", "для себя", "рассматриваю для себя", "рассматриваю"}:
        chosen = state.get("chosen", {}) or {}
        if chosen.get("title"):
            await message.answer(
                f"Поняла Вас 🙂 Тогда ориентируемся на «{chosen['title']}».\n\n"
                "Подскажите, пожалуйста, что для Вас важнее при выборе:\n"
                "• быстрее начать\n"
                "• больше направлений внутри\n"
                "• поддержка/чат\n"
                "• бюджет\n\n"
                "Что на первом месте?"
            )
            return

    # --- BUY INTENT ---
    if is_buy_intent(user_text):
        chosen = state.get("chosen", {}) or {}
        if not chosen.get("title"):
            await message.answer(
                "Поняла 🙂 Чтобы оформить заявку, сначала уточним выбор.\n"
                "Напишите, пожалуйста, какой курс или тариф Вы хотите (название) — и я оформлю дальше."
            )
            return

        state["stage"] = Stage.LEAD_COLLECT
        await db_save_user(state)

        await message.answer(
            "Хорошо 🙂 Чтобы оформить заявку, напишите одним сообщением:\n"
            "1) Фамилия Имя\n"
            "2) Телефон\n"
            "3) E-mail\n"
            f"4) Выбранный курс/тариф: {chosen.get('title')}\n\n"
            "Я передам заявку, и куратор Юлия свяжется с Вами."
        )
        return

    # --- LEAD COLLECT ---
    if state.get("stage") == Stage.LEAD_COLLECT:
        # вытащим ФИО как первую строку/первые 2 слова
        fio = None
        # если есть перенос строк — первая строка
        first_line = (user_text.splitlines()[0] or "").strip()
        words = first_line.split()
        if len(words) >= 2:
            fio = f"{words[0]} {words[1]}"
        elif len(words) == 1:
            fio = words[0]

        phone = extract_phone(user_text)
        email = extract_email(user_text)

        if fio:
            prof["fio"] = fio
        if phone:
            prof["phone"] = phone
        if email:
            prof["email"] = email

        state["profile"] = prof

        missing = []
        if not prof.get("fio"):
            missing.append("Фамилия Имя")
        if not prof.get("phone"):
            missing.append("телефон")
        if not prof.get("email") or not looks_like_email(prof.get("email")):
            missing.append("email")

        if missing:
            await db_save_user(state)
            await message.answer("Мне не хватает: " + ", ".join(missing) + " 🙂 Напишите, пожалуйста.")
            return

        # отправляем заявку во внутренний чат
        lead_text = format_lead(state, user_text)
        await send_internal_lead(lead_text)

        # ответ пользователю — согласуем род
        sex = prof.get("sex")
        sent_word = verb_variant(sex, "передала", "передала")  # Лиза = женский образ; оставляем "передала"
        await message.answer(
            f"Спасибо! 😊 Я {sent_word} заявку.\n"
            "Куратор Юлия свяжется с Вами и подскажет дальнейшие шаги."
        )

        state["stage"] = Stage.NORMAL
        await db_save_user(state)
        return

    # --- AI fallback (только если есть выжимка) ---
    add_history(state, "user", user_text)
    await db_save_user(state)

    ai = await ai_answer(user_text, state)
    if ai:
        add_history(state, "assistant", ai)
        await db_save_user(state)
        await message.answer(ai)
        return

    # --- если нет данных в YAML и AI не помог ---
    await message.answer(
        "Сейчас у меня нет точной информации по этому вопросу в базе 🙈\n"
        "Я могу уточнить у куратора Юлии.\n\n"
        "Подскажите, пожалуйста, как удобнее получить ответ — телефон или e-mail?"
    )


# =========================
# WEBHOOK / AIOHTTP APP
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
    try:
        await bot.session.close()
    except Exception:
        pass


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
