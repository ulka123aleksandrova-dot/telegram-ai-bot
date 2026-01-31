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
    1) dict (ваш текущий формат: project/media/guest_access/faq/...)
    2) list (старый формат: - id: ..., - id: ...)
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
    """Безопасный доступ по точечному пути: 'guest_access.website.url'"""
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
# MEDIA from YAML (knowledge.media.*)
# =========================
def media_get(key: str) -> Optional[dict]:
    media = knowledge.get("media")
    if isinstance(media, dict):
        m = media.get(key)
        if isinstance(m, dict):
            return m
    return None


async def send_media_by_key(message: Message, st: "UserState", key: str, caption_override: Optional[str] = None) -> bool:
    m = media_get(key)
    if not m:
        return False

    mtype = m.get("type")
    fid = m.get("file_id")
    title = m.get("title") or ""
    caption = caption_override or title

    if not fid:
        return False

    # не слать повторно
    if fid in st.sent_media_file_ids:
        return False

    if mtype == "photo":
        await message.answer_photo(photo=fid, caption=caption[:1024] if caption else None)
    elif mtype == "video":
        await message.answer_video(video=fid, caption=caption[:1024] if caption else None)
    elif mtype == "document":
        await message.answer_document(document=fid, caption=caption[:1024] if caption else None)
    else:
        return False

    st.sent_media_file_ids.add(fid)
    return True


# =========================
# CATALOG: courses/tariffs/items
# =========================
def collect_catalog() -> List[dict]:
    """
    Собираем всё, что может быть курсом/тарифом:
    - knowledge.courses: [...]
    - knowledge.tariffs: [...]
    - knowledge.items: [...] (если вдруг остался)
    """
    out: List[dict] = []

    for key in ["courses", "tariffs", "items"]:
        v = knowledge.get(key)
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    out.append(it)

    return out


CATALOG: List[dict] = []
ALIAS_INDEX: Dict[str, List[dict]] = {}

def rebuild_index() -> None:
    global CATALOG, ALIAS_INDEX
    CATALOG = collect_catalog()
    ALIAS_INDEX = {}

    for it in CATALOG:
        title = str(it.get("title") or "").strip()
        aliases = it.get("aliases") or []
        _id = str(it.get("id") or "").strip()
        keys = set()

        if title:
            keys.add(normalize_text(title))
        if _id:
            keys.add(normalize_text(_id))

        if isinstance(aliases, list):
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    keys.add(normalize_text(a))

        # спец-ключи для "тариф 1/2/3"
        t = normalize_text(title)
        m = re.search(r"\bтариф\s*(\d+)\b", t)
        if m:
            keys.add(f"тариф {m.group(1)}")

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
        _id = it.get("id") or it.get("title") or id(it)
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
    WAIT_RECEIPT = "wait_receipt"
    CONFIRM_RECEIPT = "confirm_receipt"

@dataclass
class UserProfile:
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    sex: Optional[str] = None
    goal: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

@dataclass
class UserState:
    stage: str = Stage.ASK_NAME
    last_seen: float = field(default_factory=lambda: time.time())
    history: List[dict] = field(default_factory=list)
    profile: UserProfile = field(default_factory=UserProfile)
    chosen_item_id: Optional[str] = None
    chosen_item_title: Optional[str] = None
    chosen_item_price: Optional[int] = None
    pending_receipt_file_id: Optional[str] = None
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
    Принимает:
    - "марина"
    - "Меня зовут Марина"
    - "привет, меня зовут марина. хочу..."
    """
    if not text:
        return None, None

    t = text.strip()

    m = re.search(r"(?:меня\s+зовут|я)\s+([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]+)", t, re.IGNORECASE)
    if m:
        first = m.group(1).strip()
        return first[:1].upper() + first[1:], None

    words = re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]+", t)
    words = [w for w in words if len(w) >= 2]

    if len(words) == 1 and len(t.split()) <= 3:
        first = words[0]
        return first[:1].upper() + first[1:], None

    if len(words) >= 2 and len(t.split()) <= 4:
        f = words[0]
        l = words[1]
        return f[:1].upper() + f[1:], l[:1].upper() + l[1:]

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
    return "презентац" in normalize_text(text)

def is_project_request(text: str) -> bool:
    t = normalize_text(text)
    return any(w in t for w in ["о школе", "о проекте", "про инстарт", "что такое", "расскажи о школе", "расскажи про проект"])

def is_tariff_or_course_request(text: str) -> bool:
    t = normalize_text(text)
    return any(w in t for w in ["тариф", "курс", "обучение", "нейросет", "маркетплейс", "вайлдберриз", "озон", "инфограф"])

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


async def typing_loop(chat_id: int, stop_event: asyncio.Event) -> None:
    try:
        while not stop_event.is_set():
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4)
    except Exception:
        return


async def send_admin(text: str) -> None:
    try:
        await bot.send_message(ADMIN_CHAT_ID_INT, text)
    except Exception as e:
        log.exception("Failed to send admin message: %s", e)


# =========================
# PROMPT
# =========================
def build_system_prompt(uid: int) -> str:
    disclaim = kget("faq", None)
    # Берём формулировку про доход из faq если она есть, иначе дефолт
    income_disclaimer = "Гарантий дохода нет — результат зависит от усилий, времени и выбранного направления."

    # Попробуем найти FAQ про доход
    faq = knowledge.get("faq")
    if isinstance(faq, list):
        for item in faq:
            if isinstance(item, dict):
                q = normalize_text(item.get("q", ""))
                if "доход" in q or "гарант" in q:
                    a = item.get("a")
                    if isinstance(a, str) and a.strip():
                        income_disclaimer = a.strip()
                        break

    return f"""
Вы — “{ASSISTANT_NAME}”, ассистент куратора {OWNER_NAME} в онлайн-школе {PROJECT_NAME}.
Вы одновременно профессиональный менеджер по продажам (не проговаривать клиенту).

ВАЖНО:
- Общение только на «Вы».
- Все факты о школе, курсах, тарифах, цене, бонусах, ссылках и медиа — ТОЛЬКО из knowledge.yaml.
- Если в knowledge.yaml нет нужной информации — НЕ выдумывайте: предложите оставить контакт для уточнения у куратора.
- Не обещайте гарантированный доход. Формулировка: {income_disclaimer}

СТИЛЬ:
- Дружелюбно, тактично, живо. Без давления.
- Обычно 1–6 коротких абзацев.
- В конце: 1 уточняющий вопрос.
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
        f"Я {ASSISTANT_NAME} — помощница куратора {OWNER_NAME} в онлайн-школе {PROJECT_NAME}.\n"
        "Помогу подобрать курс и тариф под Вашу цель.\n\n"
        "Как я могу к Вам обращаться?"
    )


# =========================
# PHOTO: чек
# =========================
@dp.message(F.photo)
async def on_photo(message: Message):
    uid = message.from_user.id
    st = user_state.setdefault(uid, UserState())
    st.last_seen = time.time()

    if st.stage != Stage.WAIT_RECEIPT:
        await message.answer(
            "Вижу фото 🙂\n"
            "Если это чек — сначала выберите курс/тариф, и я оформлю заявку ✅"
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
        await cb.message.answer("Хорошо 🙂 Тогда продолжим. Что хотите выбрать — курс или тариф?")
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

    # 1) Имя
    if st.stage == Stage.ASK_NAME:
        first, last = extract_name(text)
        if first:
            st.profile.first_name = first
            st.profile.last_name = last
            st.profile.sex = guess_sex_by_name(first)
            st.stage = Stage.QUALIFY

            # вопросы из YAML если есть, иначе дефолт
            dq = kget("sales_script.discovery_questions", None)
            if isinstance(dq, list) and dq:
                await message.answer(f"{first}, очень приятно познакомиться! 😊\n\n{dq[0]}")
            else:
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

    # 2) Выбор цели цифрой
    if st.stage == Stage.QUALIFY:
        t = normalize_text(text)
        if t in {"1", "2", "3"}:
            st.profile.goal = t
            st.stage = Stage.NORMAL
            if t == "1":
                await message.answer("Поняла 🙂 Подработку хотите без блога или готовы вести соцсети? (можно коротко)")
            elif t == "2":
                await message.answer("Отлично 🙂 Какое направление Вам ближе: нейросети, маркетплейсы, дизайн/инфографика, SMM? (можно 1–2 варианта)")
            else:
                await message.answer("Поняла 🙂 Вас интересует партнёрство/кураторство. Подскажите, есть ли уже опыт в онлайн-сфере или начинаем с нуля?")
            return

        # если человек не цифрой — просто продолжаем как обычный диалог
        st.stage = Stage.NORMAL

    # 3) Презентация
    if is_presentation_request(text):
        await message.answer("Сейчас отправлю презентацию проекта 📎")
        ok = await send_media_by_key(
            message, st,
            "презентация_проекта_с_призывом_хочу_гостевой_ключ",
            caption_override="Презентация проекта INSTART 📎"
        )
        if not ok:
            await message.answer("Похоже, презентацию уже отправляла ранее 🙂 Если нужно — напомню, что в ней есть.")
        return

    # 4) Гостевой доступ
    if is_guest_request(text):
        site_url = kget("guest_access.website.url", "")
        key = kget("guest_access.guest_key.key", "")
        validity = kget("guest_access.guest_key.validity", "")

        lines = ["Конечно 🙂"]
        if site_url:
            lines.append(f"\nСайт: {site_url}")
        if key:
            if validity:
                lines.append(f"\n🔑 Гостевой ключ (действует {validity}): `{key}`")
            else:
                lines.append(f"\n🔑 Гостевой ключ: `{key}`")

        # шаги
        steps = kget("guest_access.registration_instructions.steps", [])
        if isinstance(steps, list) and steps:
            short_steps = steps[:4]
            lines.append("\nКоротко как начать:\n- " + "\n- ".join(short_steps))

        await message.answer("\n".join(lines), parse_mode="Markdown")

        # промо-материалы из media
        await message.answer("Отправляю материалы по гостевому доступу ✅")
        await send_media_by_key(message, st, "макет_по_гостевому_доступу", caption_override="Макет по гостевому доступу ✅")
        await send_media_by_key(message, st, "памятка_по_регистрации_и_активации_ключа", caption_override="Памятка по регистрации ✅")
        await send_media_by_key(message, st, "инструкция_как_зарегистрироваться_и_активировать_к", caption_override="Видео-инструкция по активации ✅")

        await message.answer("Подскажите, пожалуйста, какая цель у Вас сейчас основная: подработка или новая профессия?")
        return

    # 5) Про школу/проект (строго из YAML)
    if is_project_request(text):
        desc = kget("project.description", "")
        mission = kget("project.mission", "")
        founded = kget("project.founded.date", "")
        license_num = kget("project.license.license_number", "")
        license_date = kget("project.license.license_date", "")

        parts = []
        if isinstance(desc, str) and desc.strip():
            parts.append(desc.strip())
        if mission:
            parts.append(f"Миссия: {mission}")
        if founded:
            parts.append(f"Проект основан: {founded}")
        if license_num and license_date:
            parts.append(f"Лицензия: № {license_num} от {license_date}")

        if not parts:
            await message.answer(
                "В базе есть INSTART, но сейчас не вижу заполненного описания проекта 🙈\n"
                "Скажите, пожалуйста, Вас больше интересует подработка, новая профессия или партнёрство?"
            )
            return

        await message.answer("\n\n".join(parts))
        await message.answer("Подскажите, пожалуйста, какое направление Вам сейчас ближе: нейросети, маркетплейсы, дизайн/инфографика, SMM?")
        return

    # 6) Курсы/тарифы по запросу (из каталога)
    item = find_one_item(text, types=["course", "tariff"])
    if item:
        title = item.get("title", "Без названия")
        typ = normalize_text(item.get("type", ""))
        short_desc = item.get("short_description") or item.get("description")

        # цена (разные форматы)
        price_text = ""
        price = item.get("price")
        if isinstance(price, dict):
            pw = price.get("with_chat_rub") or price.get("with_chat")
            p0 = price.get("without_chat_rub") or price.get("without_chat")
            if pw and p0 and pw != p0:
                price_text = f"Цена: с чатом — {pw} ₽, без чата — {p0} ₽."
            elif pw:
                price_text = f"Цена: {pw} ₽."
            elif p0:
                price_text = f"Цена: {p0} ₽."
        elif isinstance(price, (int, float, str)) and str(price).strip():
            price_text = f"Цена: {price} ₽."

        lines = []
        if typ == "tariff":
            lines.append(f"**Тариф:** {title}")
        else:
            lines.append(f"**Курс:** {title}")
        if price_text:
            lines.append(price_text)
        if isinstance(short_desc, str) and short_desc.strip():
            lines.append("\n" + short_desc.strip())

        await message.answer("\n".join(lines), parse_mode="Markdown")

        # медиа из карточки (если в ваших курсах/тарифах есть media:{type,file_id})
        media = item.get("media")
        if isinstance(media, dict) and media.get("file_id") and media.get("type"):
            fid = media.get("file_id")
            if fid not in st.sent_media_file_ids:
                await message.answer("Сейчас отправлю макет/материалы по этому варианту 📎")
                mtype = media.get("type")
                cap = (media.get("title") or f"Материалы по «{title}»")[:1024]
                if mtype == "photo":
                    await message.answer_photo(photo=fid, caption=cap)
                elif mtype == "video":
                    await message.answer_video(video=fid, caption=cap)
                elif mtype == "document":
                    await message.answer_document(document=fid, caption=cap)
                st.sent_media_file_ids.add(fid)

        await message.answer("Подскажите, пожалуйста: хотите оформить заявку на этот вариант или сравним ещё 1–2?")
        return

    # 7) Если спросили про маркетплейсы, а точного совпадения нет — покажем варианты из каталога
    if "маркетплейс" in normalize_text(text) or "вайлдберриз" in normalize_text(text) or "озон" in normalize_text(text):
        matches = find_items_by_query("маркетплейс", types=["course", "tariff"])
        if matches:
            titles = [m.get("title") for m in matches if m.get("title")]
            titles = titles[:6]
            await message.answer("Вот что нашла по маркетплейсам в базе 🙂\n\n- " + "\n- ".join(titles))
            await message.answer("Какой вариант Вам ближе? Напишите название — и я пришлю описание/макет.")
            return

    # 8) Покупка
    if BUY_INTENT_RE.search(text):
        st.stage = Stage.BUY_COLLECT
        await message.answer(
            "Хорошо 🙂 Чтобы оформить заявку, напишите одним сообщением:\n"
            "1) Фамилия Имя\n"
            "2) Телефон\n"
            "3) E-mail\n"
            "4) Выбранный курс/тариф (название)\n\n"
            "Если ещё не выбрали — напишите цель и направление, я предложу 1–3 варианта."
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

    # 9) OpenAI fallback (ТОЛЬКО если не нашли в YAML)
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
        return (resp.choices[0].message.content or "").strip()

    try:
        sys = build_system_prompt(uid)
        msgs = [{"role": "system", "content": sys}]
        msgs.extend(st.history[-HISTORY_MAX_TURNS * 2 :])
        msgs.append({"role": "user", "content": text})

        answer = await asyncio.to_thread(call_openai_sync, msgs)

        elapsed = time.time() - start_ts
        if elapsed < 1.5:
            await asyncio.sleep(1.5 - elapsed)

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
