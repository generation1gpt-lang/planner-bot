import os, json, logging, tempfile
from datetime import datetime, date, timezone, timedelta
from contextlib import asynccontextmanager

import telebot
from telebot.async_telebot import AsyncTeleBot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from groq import Groq
import asyncpg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BOT_TOKEN    = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
APP_URL      = os.environ.get("APP_URL", "")
DATABASE_URL = os.environ["DATABASE_URL"]

bot         = AsyncTeleBot(BOT_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)
scheduler   = AsyncIOScheduler()
pool: asyncpg.Pool = None

TZ = timezone(timedelta(hours=3))

# ── DB ───────────────────────────────────────────────────────────────────────
async def init_db():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL, ssl="require")
    async with pool.acquire() as c:
        await c.execute("""
        CREATE TABLE IF NOT EXISTS tasks(
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            title TEXT NOT NULL,
            note TEXT DEFAULT '',
            type TEXT DEFAULT 'timed',
            task_date TEXT, task_time TEXT,
            prio TEXT DEFAULT 'med',
            done BOOLEAN DEFAULT FALSE,
            notified BOOLEAN DEFAULT FALSE,
            ai_generated BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS checklists(
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checklist_items(
            id SERIAL PRIMARY KEY,
            checklist_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            done BOOLEAN DEFAULT FALSE
        );
        """)

def open_app_kb():
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("📋 Открыть планировщик", web_app=WebAppInfo(url=APP_URL))
    ]])

def now_local():
    return datetime.now(TZ)

# ── SCHEDULER ────────────────────────────────────────────────────────────────
async def send_reminders():
    now = now_local()
    now_date = now.strftime("%Y-%m-%d")
    now_time = now.strftime("%H:%M")
    async with pool.acquire() as c:
        rows = await c.fetch(
            "SELECT id,user_id,title,note FROM tasks "
            "WHERE done=FALSE AND notified=FALSE AND type='timed' "
            "AND task_date=$1 AND task_time=$2",
            now_date, now_time
        )
        for row in rows:
            text = f"⏰ *Напоминание*\n\n*{row['title']}*"
            if row['note']: text += f"\n_{row['note']}_"
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Выполнено", callback_data=f"done_{row['id']}"),
                InlineKeyboardButton("📋 Открыть", web_app=WebAppInfo(url=APP_URL)),
            ]])
            try:
                await bot.send_message(row['user_id'], text, parse_mode="Markdown", reply_markup=kb)
                await c.execute("UPDATE tasks SET notified=TRUE WHERE id=$1", row['id'])
                log.info(f"Reminder sent uid={row['user_id']} task={row['title']}")
            except Exception as e:
                log.error(f"Reminder error: {e}")

# ── AI ───────────────────────────────────────────────────────────────────────
async def transcribe(path: str) -> str:
    with open(path, "rb") as f:
        r = groq_client.audio.transcriptions.create(
            file=(os.path.basename(path), f),
            model="whisper-large-v3",
            language="ru",
        )
    return r.text

async def parse_task(text: str) -> dict:
    now = now_local()
    today = now.strftime("%Y-%m-%d")
    now_str = now.strftime("%H:%M")
    now_h, now_m = now.hour, now.minute
    prompt = f"""Извлеки задачу. Сегодня {today}, сейчас {now_str} (UTC+3).
Верни ТОЛЬКО JSON без markdown:
{{"title":"краткое действие","note":"детали или пустая строка","task_date":"YYYY-MM-DD или null","task_time":"HH:MM или null","prio":"high|med|low"}}
- "через N минут" → {now_h}:{now_m:02d} + N минут = точное HH:MM
- "через час" → +60 минут
- без времени → task_time: null
- prio: high если срочно, low если мелочь, иначе med
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
    today = now_local().strftime("%Y-%m-%d")
    async with pool.acquire() as c:
        rows = await c.fetch(
            "SELECT title,task_time,prio FROM tasks "
            "WHERE user_id=$1 AND done=FALSE AND (task_date=$2 OR type='anytime') LIMIT 10",
            user_id, today
        )
    if not rows:
        return "На сегодня задач нет — добавь что-нибудь! 🌟"
    task_list = "\n".join(f"- {r['title']}{' ('+r['task_time']+')' if r['task_time'] else ''} [{r['prio']}]" for r in rows)
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":
            f"Задачи:\n{task_list}\n\nДай короткий (2 предложения) мотивирующий совет по приоритетам на русском."}],
        max_tokens=120,
    )
    return r.choices[0].message.content.strip()

async def breakdown_task(title: str) -> list:
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":
            f'Разбей задачу "{title}" на 4-5 шагов. Верни JSON массив строк без markdown: ["шаг 1",...]'}],
        max_tokens=250,
    )
    raw = r.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try:
        steps = json.loads(raw.strip())
        return steps if isinstance(steps, list) else []
    except Exception:
        return [f"Подготовиться: {title}", "Выполнить основное", "Проверить результат"]

# ── BOT ──────────────────────────────────────────────────────────────────────
def main_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Открыть планировщик", web_app=WebAppInfo(url=APP_URL))],
        [InlineKeyboardButton("➕ Добавить текстом", callback_data="add_text"),
         InlineKeyboardButton("✨ AI-анализ", callback_data="ai_insight")],
        [InlineKeyboardButton("📊 Задачи на сегодня", callback_data="list_today")],
    ])

@bot.message_handler(commands=["start","help"])
async def cmd_start(msg):
    await bot.send_message(msg.chat.id,
        "👋 *Привет! Я твой AI-планировщик.*\n\n"
        "🎙 *Голосовое* → создам задачу автоматически\n"
        "✏️ *Текст* → напиши что угодно, разберу сам\n"
        "📋 *Приложение* → полный интерфейс\n"
        "⏰ *Уведомления* → напомню в нужное время\n\n"
        "_Просто отправь голосовое или напиши задачу!_",
        parse_mode="Markdown", reply_markup=main_kb())

@bot.message_handler(commands=["today"])
async def cmd_today(msg):
    await list_today_for(msg.chat.id, msg.from_user.id)

@bot.message_handler(commands=["priorities"])
async def cmd_priorities(msg):
    wait = await bot.send_message(msg.chat.id, "🤖 Анализирую...")
    text = await ai_insight(msg.from_user.id)
    await bot.edit_message_text(f"✨ *AI-анализ*\n\n{text}", msg.chat.id, wait.message_id, parse_mode="Markdown")

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
            msg.chat.id, wait.message_id, parse_mode="Markdown")
        task = await parse_task(text)
        await save_task(msg.from_user.id, task, ai_generated=True)
        reply = task_confirm_text(task)
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Верно", callback_data="ok"),
            InlineKeyboardButton("📋 Открыть", web_app=WebAppInfo(url=APP_URL)),
        ]])
        await bot.edit_message_text(reply, msg.chat.id, wait.message_id,
            parse_mode="Markdown", reply_markup=kb)
    except Exception as e:
        log.error(e)
        await bot.edit_message_text("❌ Ошибка. Попробуй ещё раз.", msg.chat.id, wait.message_id)

@bot.message_handler(func=lambda m: m.text and not m.text.startswith("/"))
async def handle_text(msg):
    wait = await bot.send_message(msg.chat.id, "⚙️ Создаю задачу...")
    try:
        task = await parse_task(msg.text)
        await save_task(msg.from_user.id, task)
        reply = task_confirm_text(task)
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Ок", callback_data="ok"),
            InlineKeyboardButton("📋 Открыть", web_app=WebAppInfo(url=APP_URL)),
        ]])
        await bot.edit_message_text(reply, msg.chat.id, wait.message_id,
            parse_mode="Markdown", reply_markup=kb)
    except Exception as e:
        log.error(e)
        await bot.edit_message_text("❌ Не удалось создать задачу.", msg.chat.id, wait.message_id)

@bot.callback_query_handler(func=lambda c: True)
async def handle_cb(call):
    uid = call.from_user.id
    d   = call.data
    if d == "ok":
        await bot.answer_callback_query(call.id, "✅")
        await bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)
    elif d == "add_text":
        await bot.answer_callback_query(call.id)
        await bot.send_message(uid, "✏️ Напиши задачу, например:\n_«Встреча завтра в 14:00»_\n_«Позвонить врачу»_", parse_mode="Markdown")
    elif d == "ai_insight":
        await bot.answer_callback_query(call.id)
        wait = await bot.send_message(uid, "🤖 Анализирую...")
        text = await ai_insight(uid)
        await bot.edit_message_text(f"✨ *AI-анализ приоритетов*\n\n{text}", uid, wait.message_id,
            parse_mode="Markdown", reply_markup=open_app_kb())
    elif d == "list_today":
        await bot.answer_callback_query(call.id)
        await list_today_for(uid, uid)
    elif d.startswith("done_"):
        task_id = int(d.split("_")[1])
        async with pool.acquire() as c:
            await c.execute("UPDATE tasks SET done=TRUE WHERE id=$1 AND user_id=$2", task_id, uid)
        await bot.answer_callback_query(call.id, "✅ Выполнено!")
        await bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)

# ── HELPERS ──────────────────────────────────────────────────────────────────
async def save_task(user_id, task: dict, ai_generated=False):
    async with pool.acquire() as c:
        await c.execute(
            "INSERT INTO tasks(user_id,title,note,type,task_date,task_time,prio,ai_generated) VALUES($1,$2,$3,$4,$5,$6,$7,$8)",
            user_id, task["title"], task.get("note",""),
            "timed" if task.get("task_date") or task.get("task_time") else "anytime",
            task.get("task_date"), task.get("task_time"),
            task.get("prio","med"), ai_generated
        )

def task_confirm_text(task: dict) -> str:
    prio_e = {"high":"🔴","med":"🟡","low":"🟢"}.get(task.get("prio","med"),"🟡")
    t = f"✅ *Задача добавлена*\n\n📌 {task['title']}"
    if task.get("task_date"): t += f"\n📅 {task['task_date']}"
    if task.get("task_time"): t += f"  ⏰ {task['task_time']}"
    if task.get("note"): t += f"\n📝 {task['note']}"
    t += f"\n{prio_e} Приоритет: {task.get('prio','med')}"
    return t

async def list_today_for(chat_id, user_id):
    today = now_local().strftime("%Y-%m-%d")
    async with pool.acquire() as c:
        rows = await c.fetch(
            "SELECT id,title,task_time,prio,done FROM tasks "
            "WHERE user_id=$1 AND (task_date=$2 OR type='anytime') ORDER BY task_time NULLS LAST",
            user_id, today
        )
    if not rows:
        await bot.send_message(chat_id, "📭 На сегодня задач нет.", reply_markup=open_app_kb())
        return
    prio_e = {"high":"🔴","med":"🟡","low":"🟢"}
    lines = [f"{'✅' if r['done'] else '◻️'}{prio_e.get(r['prio'],'🟡')}{' '+r['task_time'] if r['task_time'] else ''} {r['title']}" for r in rows]
    await bot.send_message(chat_id, f"📋 *Задачи на сегодня*\n\n" + "\n".join(lines),
        parse_mode="Markdown", reply_markup=open_app_kb())

# ── FASTAPI ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    scheduler.add_job(send_reminders, "cron", minute="*")
    scheduler.start()
    if APP_URL:
        await bot.set_webhook(url=f"{APP_URL}/webhook")
        log.info(f"Webhook: {APP_URL}/webhook")
    yield
    scheduler.shutdown()
    await pool.close()

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

@app.get("/api/tasks/{user_id}")
async def api_get_tasks(user_id: int):
    async with pool.acquire() as c:
        tasks = await c.fetch("SELECT * FROM tasks WHERE user_id=$1 ORDER BY task_date NULLS LAST, task_time NULLS LAST", user_id)
        cls = await c.fetch("SELECT * FROM checklists WHERE user_id=$1", user_id)
        checklists = []
        for cl in cls:
            items = await c.fetch("SELECT * FROM checklist_items WHERE checklist_id=$1", cl['id'])
            checklists.append({"id":cl['id'],"name":cl['name'],
                "items":[{"id":i['id'],"text":i['text'],"done":i['done']} for i in items]})
    return {
        "tasks": [{"id":t['id'],"user_id":t['user_id'],"title":t['title'],"note":t['note'],
            "type":t['type'],"task_date":t['task_date'],"task_time":t['task_time'],
            "prio":t['prio'],"done":t['done'],"ai_generated":t['ai_generated']} for t in tasks],
        "checklists": checklists
    }

@app.post("/api/tasks/{user_id}")
async def api_create_task(user_id: int, req: Request):
    d = await req.json()
    async with pool.acquire() as c:
        row = await c.fetchrow(
            "INSERT INTO tasks(user_id,title,note,type,task_date,task_time,prio) VALUES($1,$2,$3,$4,$5,$6,$7) RETURNING id",
            user_id, d["title"], d.get("note",""), d.get("type","timed"),
            d.get("task_date"), d.get("task_time"), d.get("prio","med")
        )
    return {"id": row["id"]}

@app.patch("/api/tasks/{task_id}")
async def api_update_task(task_id: int, req: Request):
    d = await req.json()
    valid = ["title","note","type","task_date","task_time","prio","done","notified"]
    fields = {k:v for k,v in d.items() if k in valid}
    if not fields: return {"ok": False}
    sets = ", ".join(f"{k}=${i+2}" for i,k in enumerate(fields))
    async with pool.acquire() as c:
        await c.execute(f"UPDATE tasks SET {sets} WHERE id=$1", task_id, *fields.values())
    return {"ok": True}

@app.delete("/api/tasks/{task_id}")
async def api_delete_task(task_id: int):
    async with pool.acquire() as c:
        await c.execute("DELETE FROM tasks WHERE id=$1", task_id)
    return {"ok": True}

@app.post("/api/ai/insight/{user_id}")
async def api_insight(user_id: int):
    return {"insight": await ai_insight(user_id)}

@app.post("/api/ai/breakdown")
async def api_breakdown(req: Request):
    d = await req.json()
    return {"steps": await breakdown_task(d["title"])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
