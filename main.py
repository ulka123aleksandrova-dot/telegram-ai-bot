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
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me")

ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")            # chat_id куда слать заявки/чеки (можно пустым)
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
        log.warning("knowledge.yaml не найден рядом с main.py")
        return {}
    except Exception as e:
        log.exception("Ошибка чтения knowledge.yaml: %s", e)
        return {}

    if data is None:
        return {}

    # Если вдруг в корне список — завернём, чтобы не падать
    if isinstance(data, list):
        return {"items": data}

    if not isinstance(data, dict):
        raise RuntimeError("knowledge.yaml должен быть YAML-словарём (mapping) или списком (list).")

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
    gender: Optional[str] = None  # "female" | "male" | None
    phone: Optional[str] = None
    email: Optional[str] = None


@dataclass
class UserState:
    stage: str = Stage.ASK_NAME
    chosen_tariff_title: Optional[str] = None
    chosen_tariff_price: Optional[int] = None
    last_seen: float = field(default_factory=lambda: time.time())
    history: List[dict] = field(default_factory=list)  # [{"role":"user","content":...}]
    profile: UserProfile = field(default_factory=UserProfile)
    pending_receipt_file_id: Optional[str] = None
    sent_media_keys: set = field(default_factory=set)  # чтобы не отправлять одно и то же повторно


user_state: Dict[int, UserState] = {}


def cleanup_states(now: float) -> None:
    dead = [uid for uid, st in user_state.items() if now - st.last_seen > STATE_TTL_SECONDS]
    for uid in dead:
        user_state.pop(uid, None)


def add_history(uid: int, role: str, content: str) -> None:
    st = user_state.setdefault(uid, UserState())
    st.history.append({"role": role, "content": content})
    if len(st.history) > HISTORY_MAX_TURNS * 2:
        st.history = st.history[-HISTORY_MAX_TURNS * 2:]


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


NAME_RE = re.compile(
    r"(?:меня\s+зовут|я)\s+([A-Za-zА-Яа-яЁё\-]+)(?:\s+([A-Za-zА-Яа-яЁё\-]+))?",
    re.IGNORECASE
)

def extract_name(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = (text or "").strip()

    # 1) ищем "меня зовут ..." или "я ..."
    m = NAME_RE.search(t)
    if m:
        return m.group(1), m.group(2)

    # 2) если прислали два слова (имя фамилия)
    m2 = TWO_WORDS_RE.match(t)
    if m2:
        return m2.group(1), m2.group(2)

    # 3) если одно слово (имя)
    if re.fullmatch(r"[A-Za-zА-Яа-яЁё\-]{2,}", t):
        return t, None

    return None, None

def guess_gender_by_name(first_name: str) -> Optional[str]:
    if not first_name:
        return None
    n = first_name.strip().lower()

    # очень грубая эвристика (достаточно для бота)
    female_endings = ("а", "я")
    male_exceptions = {"никита", "илья", "кузьма", "фома", "миша", "саша", "женя"}
    female_exceptions = {"любовь"}

    if n in female_exceptions:
        return "female"
    if n in male_exceptions:
        return "male"

    if n.endswith(female_endings):
        return "female"
    return "male"


def polite_ready_phrase(gender: Optional[str]) -> str:
    # для формулировок “готов/готова” — но у нас общение только на "Вы",
    # поэтому используем нейтрально
    return "готовы"


BUY_INTENT_RE = re.compile(r"\b(купить|оплат(ить|а)|готов(ы)? купить|беру тариф|хочу тариф|оформим|оформить)\b", re.IGNORECASE)


def is_guest_request(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in ["гост", "демо", "пробн", "ключ"])


def is_presentation_request(text: str) -> bool:
    t = (text or "").lower()
    return "презентац" in t


def is_tariff_question(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in ["тариф", "цена", "стоим", "сколько"])


def is_about_project(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in [
        "instart",
        "инстарт",
        "про instart",
        "про инстарт",
        "про школу",
        "о школе",
        "что такое instart",
        "что такое инстарт",
        "расскаж",
        "узнать подробнее",
        "подробнее про",
    ])


# =========================
# HELPERS: tariffs/media from YOUR YAML
# =========================
def tariffs_list() -> List[dict]:
    t = kget("tariffs", [])
    return t if isinstance(t, list) else []


def tariffs_brief() -> str:
    lines = []
    for t in tariffs_list():
        title = t.get("title")
        price = t.get("price_rub") or t.get("price") or t.get("price_without_chat_rub")
        if title and price:
            lines.append(f"• {title} — {price} ₽")
    return "\n".join(lines) if lines else "Пока не вижу тарифы в базе knowledge.yaml."


def find_tariff_by_title(text: str) -> Optional[dict]:
    q = (text or "").strip().lower()

    # точное совпадение по названию
    for t in tariffs_list():
        if str(t.get("title", "")).strip().lower() == q:
            return t

    # "тариф 1/2/3..."
    m = re.search(r"\bтариф\s*(\d)\b", q)
    if m:
        idx = int(m.group(1)) - 1
        arr = tariffs_list()
        if 0 <= idx < len(arr):
            return arr[idx]
    return None


def media_get(key: str) -> Optional[dict]:
    media = kget("media", {})
    if isinstance(media, dict) and key in media and isinstance(media[key], dict):
        return media[key]
    return None


async def send_media_by_key(message: Message, key: str, caption_override: Optional[str] = None) -> bool:
    st = user_state.setdefault(message.from_user.id, UserState())
    if key in st.sent_media_keys:
        return False  # не отправляем повторно

    m = media_get(key)
    if not m:
        return False

    mtype = m.get("type")
    fid = m.get("file_id")
    caption = caption_override or m.get("caption") or m.get("title") or ""
    if not fid:
        return False

    if mtype == "photo":
        await message.answer("Сейчас отправлю материал 📎")
        await message.answer_photo(photo=fid, caption=caption[:1024] if caption else None)
        st.sent_media_keys.add(key)
        return True
    if mtype == "video":
        await message.answer("Сейчас отправлю материал 📎")
        await message.answer_video(video=fid, caption=caption[:1024] if caption else None)
        st.sent_media_keys.add(key)
        return True
    if mtype == "document":
        await message.answer("Сейчас отправлю материал 📎")
        await message.answer_document(document=fid, caption=caption[:1024] if caption else None)
        st.sent_media_keys.add(key)
        return True

    return False


# =========================
# HELPERS: admin
# =========================
async def send_admin(text: str) -> None:
    if not ADMIN_CHAT_ID_INT:
        log.info("ADMIN_CHAT_ID не задан — заявку некуда отправлять.")
        return
    try:
        await bot.send_message(ADMIN_CHAT_ID_INT, text)
        log.info("Admin notified OK")
    except Exception as e:
        log.exception("Failed to send admin message: %s", e)


# =========================
# HELPERS: typing loop
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
BASE_PROMPT = """
Вы — “Лиза”, ассистент куратора Юлии в онлайн-школе INSTART и одновременно профессиональный менеджер по продажам (НЕ проговаривайте это клиенту).
Главная цель — помочь человеку разобраться, подобрать подходящий курс/тариф и мягко довести до оформления заявки на покупку.

ВАЖНО:
— Общение только на «Вы».
— Тон дружелюбный, тактичный, живой. Без давления. Без манипуляций.
— Не пишите «простыни»: обычно 1–6 коротких абзацев, списки уместны.

ИСТОЧНИК ЗНАНИЙ:
— Все факты о школе, курсах, тарифах, условиях, длительности, цене, бонусах, форматах, ссылках, медиа — ТОЛЬКО из knowledge.yaml.
— Если данных в knowledge.yaml нет — НЕ выдумывайте. Скажите, что уточните у куратора Юлии, и предложите оставить контакты для уточнения.

ПОВЕДЕНИЕ:
— Задавайте вопросы по одному.
— Если клиент «хочу узнать про INSTART» — коротко расскажите о проекте и задайте 1 уточняющий вопрос про цель (подработка/профессия/партнерство).
— Персональные данные (ФИО/телефон/email) спрашивайте ТОЛЬКО когда клиент уже выбрал курс/тариф и готов оформить заявку.
""".strip()


def build_system_prompt(uid: int) -> str:
    st = user_state.setdefault(uid, UserState())

    project_desc = kget("project.description", "")
    mission = kget("project.mission", "")
    disclaim = kget("project.disclaimers.income", "Доход не гарантируется и зависит от усилий.")
    pay_phone = kget("instructions.payment.phone", "")
    pay_bank = kget("instructions.payment.bank", "")
    guest_key = kget("guest_access.key", "")

    name_line = f"Имя клиента: {st.profile.first_name or 'не указано'}."

    facts = []
    if project_desc:
        facts.append(f"Описание проекта: {project_desc}")
    if mission:
        facts.append(f"Миссия: {mission}")
    facts_block = "\n".join(facts).strip()

    tariffs_block = tariffs_brief()

    payment_block = ""
    if pay_phone and pay_bank:
        payment_block = f"Реквизиты оплаты: {pay_phone} (банк {pay_bank})."

    guest_block = f"Гостевой ключ (если нужен): {guest_key}" if guest_key else ""

    return "\n\n".join([
        BASE_PROMPT,
        name_line,
        f"Проект: {PROJECT_NAME}. Куратор: {OWNER_NAME}.",
        f"Дисклеймер по доходу: {disclaim}",
        "Тарифы (кратко):\n" + tariffs_block,
        guest_block,
        payment_block,
        "Отвечайте кратко и по делу. В конце — 1 уточняющий вопрос."
    ]).strip()


# =========================
# COMMANDS (тестовые)
# =========================
@dp.message(Command("myid"))
async def cmd_myid(message: Message):
    await message.answer(f"Ваш user_id: {message.from_user.id}\nТекущий chat_id: {message.chat.id}")


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
    uid = message.from_user.id
    cleanup_states(time.time())

    st = user_state.setdefault(uid, UserState())
    st.stage = Stage.ASK_NAME
    st.last_seen = time.time()

    await message.answer(
        "Здравствуйте! 😊\n\n"
        f"Я {ASSISTANT_NAME} — ассистент куратора {OWNER_NAME} в онлайн-школе {PROJECT_NAME}.\n"
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
            "Если это чек — пожалуйста, сначала напишите, что Вы выбрали курс/тариф и готовы оформить покупку. "
            "Я подскажу следующий шаг ✅"
        )
        return

    photo = message.photo[-1]
    st.pending_receipt_file_id = photo.file_id
    st.stage = Stage.CONFIRM_RECEIPT

    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Да, это чек", callback_data="receipt_yes")
    kb.button(text="❌ Нет, это не чек", callback_data="receipt_no")
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
        await cb.message.answer("Хорошо 🙂 Продолжим. Хотите — помогу подобрать тариф под Вашу цель.")
        return

    fid = st.pending_receipt_file_id
    st.pending_receipt_file_id = None
    st.stage = Stage.NORMAL

    await cb.message.answer("Принято ✅ Я передам информацию куратору Юлии, чтобы подтвердить оплату.")

    lead = (
        "✅ ПРИШЁЛ ЧЕК ОБ ОПЛАТЕ\n"
        f"ФИО: {(st.profile.first_name or '')} {(st.profile.last_name or '')}\n"
        f"Тариф: {st.chosen_tariff_title or 'не указан'} — {st.chosen_tariff_price or '—'} ₽\n"
        f"Телефон: {st.profile.phone or 'не указан'}\n"
        f"Email: {st.profile.email or 'не указан'}\n"
        f"User ID: {uid}"
    )
    await send_admin(lead)

    if fid and ADMIN_CHAT_ID_INT:
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

    # ---- 1) Стадия запроса имени (умное извлечение из длинной фразы)
    if st.stage == Stage.ASK_NAME:
        first, last = extract_name(text)
        if first:
            st.profile.first_name = first
            st.profile.last_name = last
            st.profile.gender = guess_gender_by_name(first)
            st.stage = Stage.QUALIFY

            # Если в этом же сообщении спрашивают про проект — отвечаем сразу по проекту
            if is_about_project(text):
                desc = kget("project.description", "")
                if desc:
                    await message.answer(
                        f"{first}, очень приятно познакомиться 😊\n\n"
                        f"{desc}\n\n"
                        "Подскажите, пожалуйста, Вы рассматриваете подработку или полноценную онлайн-профессию?"
                    )
                else:
                    await message.answer(
                        f"{first}, очень приятно познакомиться 😊\n\n"
                        f"{PROJECT_NAME} — это онлайн-школа по востребованным онлайн-направлениям.\n\n"
                        "Подскажите, пожалуйста, какая цель у Вас сейчас ближе: подработка или новая профессия?"
                    )
                return

            await message.answer(
                f"{first}, очень приятно познакомиться 😊\n\n"
                "Чтобы подсказать лучший старт, подскажите, пожалуйста, что Вам сейчас важнее:\n"
                "— подработка,\n"
                "— новая онлайн-профессия,\n"
                "— или развитие в проекте (партнёрство/кураторство)?"
            )
        else:
            await message.answer("Подскажите, пожалуйста, как я могу к Вам обращаться? (Можно просто имя)")
        return

    # ---- 2) Отдельная обработка: вопросы про INSTART (в любой стадии)
    if is_about_project(text):
        desc = kget("project.description", "")
        if desc:
            await message.answer(
                f"{desc}\n\n"
                "Подскажите, пожалуйста, Вы хотите подработку или хотите освоить новую онлайн-профессию?"
            )
        else:
            await message.answer(
                f"{PROJECT_NAME} — это онлайн-школа по востребованным онлайн-направлениям 🙂\n\n"
                "Подскажите, пожалуйста, для какой цели Вы рассматриваете обучение?"
            )
        st.stage = Stage.QUALIFY
        return

    # ---- 3) Гостевой доступ
    if is_guest_request(text):
        guest_key = kget("guest_access.key")
        if guest_key:
            await message.answer(
                "Конечно 🙂\n\n"
                f"🔑 Ваш гостевой ключ: `{guest_key}`\n\n"
                "Хотите — я подскажу, как активировать ключ на сайте (коротко, по шагам)?",
                parse_mode="Markdown",
            )
        else:
            await message.answer(
                "Сейчас в базе не вижу заполненный гостевой ключ 🙈\n"
                "Если хотите — я передам запрос куратору Юлии. Подскажите, пожалуйста, для какой цели Вам гостевой доступ?"
            )
        return

    # ---- 4) Презентация
    if is_presentation_request(text):
        # ключ медиа зависит от вашего knowledge.yaml
        # если у вас он в media: {презентация_проекта: {...}} — поменяйте строку ниже
        pres_key = "презентация_проекта"
        ok = await send_media_by_key(message, pres_key, caption_override="Презентация проекта INSTART 📎")
        if not ok:
            await message.answer(
                "Сейчас не вижу презентацию в базе 🙈\n"
                "Подскажите, пожалуйста, какая у Вас цель: подработка / профессия / партнёрство?"
            )
        return

    # ---- 5) Тарифы/цены (быстрый ответ)
    if is_tariff_question(text):
        await message.answer(
            "Вот актуальные тарифы и цены 🙂\n\n"
            f"{tariffs_brief()}\n\n"
            "Подскажите, пожалуйста, какая цель у Вас сейчас ближе?"
        )
        return

    # ---- 6) Готовность купить → сбор данных
    if BUY_INTENT_RE.search(text):
        st.stage = Stage.BUY_COLLECT
        await message.answer(
            "Поняла Вас 🙂 Давайте оформим заявку.\n\n"
            "Напишите, пожалуйста, одним сообщением:\n"
            "• Фамилия Имя\n"
            "• Телефон\n"
            "• E-mail\n"
            "• Выбранный курс/тариф (название или «тариф 1/2/3…»)"
        )
        return

    if st.stage == Stage.BUY_COLLECT:
        # пытаемся вытащить имя/фамилию, телефон, email
        first, last = extract_name(text)
        if first and not st.profile.first_name:
            st.profile.first_name = first
            st.profile.gender = guess_gender_by_name(first)
        if last and not st.profile.last_name:
            st.profile.last_name = last

        phone = extract_phone(text)
        email = extract_email(text)
        if phone:
            st.profile.phone = normalize_phone(phone)
        if email:
            st.profile.email = email.strip()

        t = find_tariff_by_title(text)
        if t:
            st.chosen_tariff_title = t.get("title")
            st.chosen_tariff_price = t.get("price_rub") or t.get("price")

        if not st.chosen_tariff_title:
            await message.answer(
                "Подскажите, пожалуйста, какой тариф Вы выбрали?\n\n"
                f"{tariffs_brief()}\n\n"
                "Можно написать «тариф 1/2/3…» или точное название."
            )
            return

        missing = []
        if not st.profile.first_name or not st.profile.last_name:
            missing.append("Фамилия Имя")
        if not st.profile.phone or len(re.sub(r"\D", "", st.profile.phone)) < 10:
            missing.append("телефон")
        if not st.profile.email or not looks_like_email(st.profile.email):
            missing.append("e-mail")

        if missing:
            await message.answer("Мне не хватает: " + ", ".join(missing) + " 🙂 Напишите, пожалуйста.")
            return

        # Формируем заявку для админа
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        gender = st.profile.gender or "не определён"

        lead_text = (
            "🟩 ЗАЯВКА НА ПОКУПКУ (INSTART)\n"
            f"Имя клиента: {st.profile.first_name}\n"
            f"Пол: {gender}\n"
            f"Фамилия Имя: {st.profile.last_name} {st.profile.first_name}\n"
            f"Телефон: {st.profile.phone}\n"
            f"Email: {st.profile.email}\n"
            f"Курс/Тариф: {st.chosen_tariff_title} — {st.chosen_tariff_price} ₽\n"
            "Источник: Telegram\n"
            f"Краткий запрос/цель: (не указано)\n"
            f"Важные детали/возражения: (не указано)\n"
            f"Дата/время: {now_str}\n"
            f"User ID: {uid}"
        )
        await send_admin(lead_text)

        # Реквизиты (если есть в knowledge.yaml)
        pay_phone = kget("instructions.payment.phone", "")
        pay_bank = kget("instructions.payment.bank", "")

        if pay_phone and pay_bank:
            await message.answer(
                "Спасибо! Я передала заявку ✅\n\n"
                "Реквизиты для оплаты:\n"
                f"📞 Номер телефона: {pay_phone}\n"
                f"🏦 Банк: {pay_bank}\n\n"
                "После оплаты пришлите, пожалуйста, чек (фото) сюда в чат — и я передам его куратору Юлии для подтверждения 🙂"
            )
            st.stage = Stage.WAIT_RECEIPT
        else:
            await message.answer(
                "Спасибо! Я передала заявку ✅\n\n"
                "Куратор Юлия свяжется с Вами и подскажет дальнейшие шаги 🙂"
            )
            st.stage = Stage.NORMAL

        return

    # =========================
    # OpenAI fallback (когда нет чётких правил)
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
        max_tokens=220,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()

    try:
        sys = build_system_prompt(uid)
        msgs = [{"role": "system", "content": sys}]
        msgs.extend(st.history[-HISTORY_MAX_TURNS * 2:])
        msgs.append({"role": "user", "content": text})

        answer = await asyncio.to_thread(call_openai_sync, msgs)

        elapsed = time.time() - start_ts
        if elapsed < 1.5:
            await asyncio.sleep(1.5 - elapsed)

        parts = split_answer(answer, max_chars=850)
        if not parts:
            parts = ["Извините, я не совсем поняла запрос 🙈 Подскажите, пожалуйста, что именно Вы хотите узнать?"]
        for p in parts:
            await message.answer(p)

        add_history(uid, "assistant", answer)
        st.stage = Stage.NORMAL

    except Exception as e:
        log.exception("OpenAI error: %s", e)
        await message.answer("Сейчас есть техническая перегрузка 🙈 Пожалуйста, попробуйте ещё раз через минуту.")
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
    # На старте: ставим webhook
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
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
        secret_token=WEBHOOK_SECRET,
    ).register(app, path=WEBHOOK_PATH)

    setup_application(app, dp, bot=bot)

    # Railway
    web.run_app(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()


