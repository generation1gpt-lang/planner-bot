import os, asyncio, json, logging, sqlite3, tempfile
from datetime import datetime, date
from contextlib import asynccontextmanager

import telebot
from telebot.async_telebot import AsyncTeleBot
from telebot.types import (
    InlineKeyboardMarkup, InlineKeyboardButton,
    WebAppInfo, ForceReply
)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from groq import Groq

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BOT_TOKEN    = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
APP_URL      = os.environ.get("APP_URL", "")   # https://yourapp.railway.app

bot          = AsyncTeleBot(BOT_TOKEN)
groq_client  = Groq(api_key=GROQ_API_KEY)
scheduler    = AsyncIOScheduler()

# ── DB ──────────────────────────────────────────────────────────────────────
def init_db():
    c = sqlite3.connect("planner.db")
    c.executescript("""
    CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        note TEXT DEFAULT '',
        type TEXT DEFAULT 'timed',
        task_date TEXT, task_time TEXT,
        prio TEXT DEFAULT 'med',
        done INTEGER DEFAULT 0,
        notified INTEGER DEFAULT 0,
        ai_generated INTEGER DEFAULT 0,
        created_at TEXT DEFAULT(datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS checklists(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS checklist_items(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        checklist_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        done INTEGER DEFAULT 0
    );
    """)
    c.commit(); c.close()

def db(): return sqlite3.connect("planner.db")


# ── UTILS ────────────────────────────────────────────────────────────────────
def task_row_to_dict(row):
    cols = ["id","user_id","title","note","type","task_date","task_time",
            "prio","done","notified","ai_generated","created_at"]
    return dict(zip(cols, row))

def open_app_kb():
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("📋 Открыть планировщик", web_app=WebAppInfo(url=APP_URL))
    ]])


# ── SCHEDULER ────────────────────────────────────────────────────────────────
async def send_reminders():
    from datetime import timezone, timedelta
    tz = timezone(timedelta(hours=3))
    now_dt = datetime.now(tz)
    now_date = now_dt.date().isoformat()
    now_time = now_dt.strftime("%H:%M")
    c = db()
    rows = c.execute(
        "SELECT id,user_id,title,note FROM tasks "
        "WHERE done=0 AND notified=0 AND type='timed' "
        "AND task_date=? AND task_time=?",
        (now_date, now_time)
    ).fetchall()
    for rid, uid, title, note in rows:
        text = f"⏰ *Напоминание*\n\n*{title}*"
        if note: text += f"\n_{note}_"
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Выполнено", callback_data=f"done_{rid}"),
            InlineKeyboardButton("📋 Открыть", web_app=WebAppInfo(url=APP_URL)),
        ]])
        try:
            await bot.send_message(uid, text, parse_mode="Markdown", reply_markup=kb)
            c.execute("UPDATE tasks SET notified=1 WHERE id=?", (rid,))
        except Exception as e:
            log.error(f"Reminder error uid={uid}: {e}")
    c.commit(); c.close()


# ── AI HELPERS ───────────────────────────────────────────────────────────────
async def transcribe(path: str) -> str:
    with open(path, "rb") as f:
        r = groq_client.audio.transcriptions.create(
            file=(os.path.basename(path), f),
            model="whisper-large-v3",
            language="ru",
        )
    return r.text


async def parse_task(text: str) -> dict:
    today = date.today().isoformat()
    now_time = datetime.now().strftime("%H:%M")
    now_h, now_m = datetime.now().hour, datetime.now().minute
    prompt = f"""Извлеки задачу. Сегодня {today}, сейчас {now_time}.
Верни ТОЛЬКО JSON без markdown:
{{"title":"краткое действие","note":"детали или пустая строка","task_date":"YYYY-MM-DD или null","task_time":"HH:MM или null","prio":"high|med|low"}}
- "через N минут" → вычисли {now_h}:{now_m:02d} + N минут = точное HH:MM
- "через час" → +60 минут от сейчас
- без времени → task_time: null
- prio high если срочно/важно
Текст: "{text}"
"""
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        max_tokens=200,
    )
    raw = r.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except Exception:
        return {"title": text[:100], "note": "", "task_date": None, "task_time": None, "prio": "med"}


async def ai_insight(user_id: int) -> str:
    today = date.today().isoformat()
    c = db()
    rows = c.execute(
        "SELECT title,task_time,prio FROM tasks "
        "WHERE user_id=? AND done=0 AND (task_date=? OR type='anytime') LIMIT 10",
        (user_id, today)
    ).fetchall(); c.close()
    if not rows:
        return "На сегодня задач нет — добавь что-нибудь! 🌟"
    task_list = "\n".join(f"- {r[0]}{' ('+r[1]+')' if r[1] else ''} [{r[2]}]" for r in rows)
    r = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role":"user","content":
            f"Задачи пользователя:\n{task_list}\n\n"
            "Дай короткий (2 предложения) мотивирующий совет по приоритетам на русском."}],
        max_tokens=120,
    )
    return r.choices[0].message.content.strip()


async def breakdown_task(title: str, user_id: int) -> list[str]:
    r = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role":"user","content":
            f'Разбей задачу "{title}" на 4-5 конкретных шагов. '
            'Верни JSON массив строк (без markdown): ["шаг 1","шаг 2",...]'}],
        max_tokens=250,
    )
    raw = r.choices[0].message.content.strip()
    try:
        steps = json.loads(raw)
        return steps if isinstance(steps, list) else []
    except Exception:
        return [f"Подготовиться к: {title}", "Выполнить основное", "Проверить результат"]


# ── KEYBOARD HELPERS ─────────────────────────────────────────────────────────
def main_kb(user_id):
    """Inline keyboard with main actions sent after /start."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Открыть планировщик", web_app=WebAppInfo(url=APP_URL))],
        [
            InlineKeyboardButton("➕ Добавить текстом", callback_data="add_text"),
            InlineKeyboardButton("✨ AI-анализ", callback_data="ai_insight"),
        ],
        [InlineKeyboardButton("📊 Задачи на сегодня", callback_data="list_today")],
    ])


# ── BOT HANDLERS ─────────────────────────────────────────────────────────────
@bot.message_handler(commands=["start", "help"])
async def cmd_start(msg):
    await bot.send_message(
        msg.chat.id,
        "👋 *Привет! Я твой AI-планировщик.*\n\n"
        "🎙 *Голосовое* → создам задачу автоматически\n"
        "✏️ *Текст* → напиши что угодно, разберу сам\n"
        "📋 *Приложение* → полный интерфейс с таймлайном\n"
        "⏰ *Уведомления* → буду напоминать в назначенное время\n\n"
        "_Просто отправь голосовое или напиши задачу прямо в чат!_",
        parse_mode="Markdown",
        reply_markup=main_kb(msg.from_user.id),
    )

@bot.message_handler(commands=["today"])
async def cmd_today(msg):
    await list_today_for(msg.chat.id, msg.from_user.id)

@bot.message_handler(commands=["priorities"])
async def cmd_priorities(msg):
    wait = await bot.send_message(msg.chat.id, "🤖 Анализирую...")
    text = await ai_insight(msg.from_user.id)
    await bot.edit_message_text(f"✨ *AI-анализ*\n\n{text}", msg.chat.id, wait.message_id, parse_mode="Markdown")


# ── VOICE ────────────────────────────────────────────────────────────────────
@bot.message_handler(content_types=["voice"])
async def handle_voice(msg):
    wait = await bot.send_message(msg.chat.id, "🎙 Распознаю голосовое...")
    try:
        fi = await bot.get_file(msg.voice.file_id)
        url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{fi.file_path}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
        tmp = f"/tmp/v_{msg.message_id}.ogg"
        open(tmp,"wb").write(resp.content)

        text = await transcribe(tmp)
        os.remove(tmp)

        await bot.edit_message_text(
            f"📝 *Распознано:*\n_{text}_\n\n⚙️ Создаю задачу...",
            msg.chat.id, wait.message_id, parse_mode="Markdown"
        )

        task = await parse_task(text)
        _save_task(msg.from_user.id, task, ai_generated=True)

        reply = _task_confirm_text(task)
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Верно", callback_data="ok"),
            InlineKeyboardButton("📋 Открыть", web_app=WebAppInfo(url=APP_URL)),
        ]])
        await bot.edit_message_text(reply, msg.chat.id, wait.message_id,
            parse_mode="Markdown", reply_markup=kb)
    except Exception as e:
        log.error(e)
        await bot.edit_message_text("❌ Ошибка. Попробуй ещё раз.",
            msg.chat.id, wait.message_id)


# ── TEXT → TASK ───────────────────────────────────────────────────────────────
@bot.message_handler(func=lambda m: m.text and not m.text.startswith("/"))
async def handle_text(msg):
    wait = await bot.send_message(msg.chat.id, "⚙️ Создаю задачу...")
    try:
        task = await parse_task(msg.text)
        _save_task(msg.from_user.id, task)
        reply = _task_confirm_text(task)
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Ок", callback_data="ok"),
            InlineKeyboardButton("📋 Открыть", web_app=WebAppInfo(url=APP_URL)),
        ]])
        await bot.edit_message_text(reply, msg.chat.id, wait.message_id,
            parse_mode="Markdown", reply_markup=kb)
    except Exception as e:
        log.error(e)
        await bot.edit_message_text("❌ Не удалось создать задачу.",
            msg.chat.id, wait.message_id)


# ── CALLBACKS ─────────────────────────────────────────────────────────────────
@bot.callback_query_handler(func=lambda c: True)
async def handle_cb(call):
    uid = call.from_user.id
    d   = call.data

    if d == "ok":
        await bot.answer_callback_query(call.id, "✅")
        await bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)

    elif d == "add_text":
        await bot.answer_callback_query(call.id)
        await bot.send_message(uid,
            "✏️ Напиши задачу в любом формате, например:\n"
            "_«Встреча с Иваном завтра в 14:00»_\n"
            "_«Позвонить врачу»_\n"
            "_«Срочно: сдать отчёт»_",
            parse_mode="Markdown")

    elif d == "ai_insight":
        await bot.answer_callback_query(call.id)
        wait = await bot.send_message(uid, "🤖 Анализирую...")
        text = await ai_insight(uid)
        await bot.edit_message_text(f"✨ *AI-анализ приоритетов*\n\n{text}",
            uid, wait.message_id, parse_mode="Markdown",
            reply_markup=open_app_kb())

    elif d == "list_today":
        await bot.answer_callback_query(call.id)
        await list_today_for(uid, uid)

    elif d.startswith("done_"):
        task_id = int(d.split("_")[1])
        c = db(); c.execute("UPDATE tasks SET done=1 WHERE id=? AND user_id=?", (task_id, uid)); c.commit(); c.close()
        await bot.answer_callback_query(call.id, "✅ Выполнено!")
        await bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)

    elif d.startswith("breakdown_"):
        task_id = int(d.split("_")[1])
        c = db()
        row = c.execute("SELECT title FROM tasks WHERE id=?", (task_id,)).fetchone()
        c.close()
        if row:
            wait = await bot.send_message(uid, "✂️ Разбиваю на шаги...")
            steps = await breakdown_task(row[0], uid)
            cl_c = db()
            cl_id = cl_c.execute("INSERT INTO checklists(user_id,name) VALUES(?,?)", (uid, row[0])).lastrowid
            for s in steps:
                cl_c.execute("INSERT INTO checklist_items(checklist_id,text) VALUES(?,?)", (cl_id, s))
            cl_c.commit(); cl_c.close()
            steps_text = "\n".join(f"{i+1}. {s}" for i,s in enumerate(steps))
            await bot.edit_message_text(
                f"✅ *Чеклист создан: {row[0]}*\n\n{steps_text}",
                uid, wait.message_id, parse_mode="Markdown",
                reply_markup=open_app_kb())


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _save_task(user_id, task: dict, ai_generated=False):
    c = db()
    c.execute(
        "INSERT INTO tasks(user_id,title,note,type,task_date,task_time,prio,ai_generated) VALUES(?,?,?,?,?,?,?,?)",
        (user_id, task["title"], task.get("note",""),
         "timed" if task.get("task_date") else "anytime",
         task.get("task_date"), task.get("task_time"),
         task.get("prio","med"), 1 if ai_generated else 0)
    )
    c.commit(); c.close()

def _task_confirm_text(task: dict) -> str:
    prio_emoji = {"high":"🔴","med":"🟡","low":"🟢"}.get(task.get("prio","med"),"🟡")
    t = f"✅ *Задача добавлена*\n\n📌 {task['title']}"
    if task.get("task_date"): t += f"\n📅 {task['task_date']}"
    if task.get("task_time"): t += f"  ⏰ {task['task_time']}"
    if task.get("note"): t += f"\n📝 {task['note']}"
    t += f"\n{prio_emoji} Приоритет: {task.get('prio','med')}"
    return t

async def list_today_for(chat_id, user_id):
    today = date.today().isoformat()
    c = db()
    rows = c.execute(
        "SELECT id,title,task_time,prio,done FROM tasks "
        "WHERE user_id=? AND (task_date=? OR type='anytime') ORDER BY task_time",
        (user_id, today)
    ).fetchall(); c.close()

    if not rows:
        await bot.send_message(chat_id, "📭 На сегодня задач нет.", reply_markup=open_app_kb())
        return

    prio_e = {"high":"🔴","med":"🟡","low":"🟢"}
    lines = []
    for rid, title, ttime, prio, done in rows:
        check = "✅" if done else "◻️"
        time_s = f" {ttime}" if ttime else ""
        lines.append(f"{check}{prio_e.get(prio,'🟡')}{time_s} {title}")

    text = f"📋 *Задачи на сегодня*\n\n" + "\n".join(lines)
    await bot.send_message(chat_id, text, parse_mode="Markdown", reply_markup=open_app_kb())


# ── FASTAPI ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scheduler.add_job(send_reminders, "cron", minute="*")
    scheduler.start()
    if APP_URL:
        await bot.set_webhook(url=f"{APP_URL}/webhook")
        log.info(f"Webhook set: {APP_URL}/webhook")
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def serve_app():
    with open("index.html") as f: return f.read()

@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()
    update = telebot.types.Update.de_json(data)
    await bot.process_new_updates([update])
    return {"ok": True}

# ── TASKS REST API ─────────────────────────────────────────────────────────────
@app.get("/api/tasks/{user_id}")
async def api_get_tasks(user_id: int):
    c = db()
    tasks = [task_row_to_dict(r) for r in c.execute("SELECT * FROM tasks WHERE user_id=? ORDER BY task_date,task_time", (user_id,)).fetchall()]
    cls_rows = c.execute("SELECT * FROM checklists WHERE user_id=?", (user_id,)).fetchall()
    checklists = []
    for cl in cls_rows:
        items = c.execute("SELECT * FROM checklist_items WHERE checklist_id=?", (cl[0],)).fetchall()
        checklists.append({"id":cl[0],"name":cl[2],"items":[{"id":i[0],"text":i[2],"done":bool(i[3])} for i in items]})
    c.close()
    return {"tasks": tasks, "checklists": checklists}

@app.post("/api/tasks/{user_id}")
async def api_create_task(user_id: int, req: Request):
    d = await req.json()
    c = db()
    cid = c.execute(
        "INSERT INTO tasks(user_id,title,note,type,task_date,task_time,prio) VALUES(?,?,?,?,?,?,?)",
        (user_id,d["title"],d.get("note",""),d.get("type","timed"),d.get("task_date"),d.get("task_time"),d.get("prio","med"))
    ).lastrowid
    c.commit(); c.close()
    return {"id": cid}

@app.patch("/api/tasks/{task_id}")
async def api_update_task(task_id: int, req: Request):
    d = await req.json()
    valid = ["title","note","type","task_date","task_time","prio","done","notified"]
    fields = {k:v for k,v in d.items() if k in valid}
    if not fields: return {"ok": False}
    c = db()
    c.execute(f"UPDATE tasks SET {','.join(k+'=?' for k in fields)} WHERE id=?", (*fields.values(), task_id))
    c.commit(); c.close()
    return {"ok": True}

@app.delete("/api/tasks/{task_id}")
async def api_delete_task(task_id: int):
    c = db(); c.execute("DELETE FROM tasks WHERE id=?", (task_id,)); c.commit(); c.close()
    return {"ok": True}

@app.post("/api/ai/insight/{user_id}")
async def api_insight(user_id: int):
    return {"insight": await ai_insight(user_id)}

@app.post("/api/ai/breakdown")
async def api_breakdown(req: Request):
    d = await req.json()
    return {"steps": await breakdown_task(d["title"], d.get("user_id", 0))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
