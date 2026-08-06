import os
import json
import re
import sqlite3
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from fastapi import FastAPI, Header, HTTPException, Request
from langchain_groq import ChatGroq
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request as GoogleAuthRequest

app = FastAPI()

# Configurable via environment variables
# details baked in -- anyone cloning it configures their own values.
LOCAL_TZ = ZoneInfo(os.getenv("TIMEZONE", "America/New_York"))
TIMEZONE_NAME = os.getenv("TIMEZONE", "America/New_York")
SELF_NAME = os.getenv("SELF_NAME", "Me")

# ------------------------------
# Common Sense Defaults
# ------------------------------

DEFAULT_EVENT_TIMES = {
    "breakfast": ("08:00", 60),
    "coffee": ("09:00", 45),
    "brunch": ("11:00", 90),
    "lunch": ("12:00", 60),

    "gym": ("18:00", 90),
    "workout": ("18:00", 90),

    "basketball": ("19:00", 120),
    "soccer": ("19:00", 120),
    "tennis": ("18:00", 90),

    "dinner": ("18:30", 90),
    "ice cream": ("20:00", 60),
    "dessert": ("20:00", 60),
    "drinks": ("20:00", 120),
    "movie": ("19:00", 150),
    "concert": ("20:00", 180),

    "meeting": ("10:00", 60),
    "call": ("15:00", 30)
}

# --- Database Setup for Memory & Event Tracking ---
def init_db():
    conn = sqlite3.connect("memory.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            sender TEXT, 
            role TEXT, 
            content TEXT, 
            event_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_history_sender_ts
        ON history(sender, timestamp)
    """)

    # New table: tracks every parsed invite from the moment it's understood,
    # not just the ones you happen to confirm right away. This is what lets
    # a missed notification be checked/confirmed later instead of vanishing.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_invites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            title TEXT,
            activity TEXT,
            location TEXT,
            time TEXT,
            duration INTEGER,
            confidence INTEGER,
            status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pending_status
        ON pending_invites(status)
    """)
    conn.commit()
    conn.close()

def save_message(sender, role, content, event_id=None):
    conn = sqlite3.connect("memory.db")
    conn.execute(
        "INSERT INTO history (sender, role, content, event_id) VALUES (?, ?, ?, ?)",
        (sender, role, content, event_id)
    )
    conn.commit()
    conn.close()

def get_history(sender):
    conn = sqlite3.connect("memory.db")
    cursor = conn.execute("""
        SELECT content FROM history 
        WHERE sender = ? 
        AND role = 'User'
        AND timestamp > datetime('now', '-6 hours')
        ORDER BY timestamp ASC
    """, (sender,))
    rows = cursor.fetchall()
    conn.close()
    return "\n".join([f"User: {r[0]}" for r in rows])

def get_last_event_id(sender, current_activity):
    """
    Looks back through the last 6 hours of context to find the last modified event
    ID, but ONLY if the logged record for that event actually mentions the same
    activity we're currently processing (e.g. "lunch" won't match a stored
    "movie" event).
    """
    conn = sqlite3.connect("memory.db")
    cursor = conn.execute("""
        SELECT event_id FROM history 
        WHERE sender = ? 
        AND event_id IS NOT NULL 
        AND content LIKE ?
        AND timestamp > datetime('now', '-6 hours')
        ORDER BY timestamp DESC LIMIT 1
    """, (sender, f"%{current_activity}%"))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def save_pending_invite(analysis):
    """Saves a freshly-parsed invitation as 'pending' the moment it's
    understood -- BEFORE you've said yes or no. This is what makes it
    possible to check back later if you miss the notification."""
    conn = sqlite3.connect("memory.db")
    cursor = conn.execute("""
        INSERT INTO pending_invites
            (sender, title, activity, location, time, duration, confidence, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
    """, (
        analysis["sender"], analysis["title"], analysis["activity"],
        analysis["location"], analysis["time"], analysis["duration"],
        analysis["confidence"]
    ))
    conn.commit()
    pending_id = cursor.lastrowid
    conn.close()
    return pending_id


def mark_pending_status(pending_id, status):
    conn = sqlite3.connect("memory.db")
    conn.execute(
        "UPDATE pending_invites SET status = ? WHERE id = ?",
        (status, pending_id)
    )
    conn.commit()
    conn.close()


def get_pending_invites():
    conn = sqlite3.connect("memory.db")
    cursor = conn.execute("""
        SELECT id, sender, activity, time, location FROM pending_invites
        WHERE status = 'pending'
        ORDER BY created_at ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_pending_invite_by_id(pending_id):
    conn = sqlite3.connect("memory.db")
    cursor = conn.execute("""
        SELECT sender, title, activity, location, time, duration, confidence
        FROM pending_invites
        WHERE id = ?
    """, (pending_id,))
    row = cursor.fetchone()
    conn.close()
    return row

init_db()

# --- Calendar & Time Utils ---
def get_google_service(name, version):
    creds_dict = json.loads(os.getenv("GOOGLE_TOKEN"))
    creds = Credentials.from_authorized_user_info(creds_dict)
    
    # Check if access token has expired and refresh it automatically
    if creds.expired and creds.refresh_token:
        creds.refresh(GoogleAuthRequest())
        # Update your local env string so subsequent requests use the valid token
        os.environ["GOOGLE_TOKEN"] = creds.to_json()
        
    return build(name, version, credentials=creds)

def parse_dt(iso_str):
    if iso_str is None:
        return None

    if iso_str.endswith("Z"):
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone(LOCAL_TZ)

    dt = datetime.fromisoformat(iso_str)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=LOCAL_TZ)

    return dt.astimezone(LOCAL_TZ)

def check_conflicts(start_time_str, duration_mins):
    service = get_google_service('calendar', 'v3')

    start_dt = parse_dt(start_time_str)
    end_dt = start_dt + timedelta(minutes=duration_mins)

    events_result = service.events().list(
        calendarId='primary',
        timeMin=start_dt.isoformat(),
        timeMax=end_dt.isoformat(),
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])
    return [e['summary'] for e in events]

def build_weekly_digest_text():
    """
    Looks at your actual Google Calendar (not just events this app created)
    and builds a friendly, day-grouped summary of everything in the next
    7 days.
    """
    service = get_google_service('calendar', 'v3')

    now = datetime.now(LOCAL_TZ)
    week_end = now + timedelta(days=7)

    events_result = service.events().list(
        calendarId='primary',
        timeMin=now.isoformat(),
        timeMax=week_end.isoformat(),
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])

    if not events:
        return "You have nothing on your calendar for the next 7 days."

    lines = []
    current_day_label = None

    for event in events:
        start_raw = event['start'].get('dateTime') or event['start'].get('date')

        if 'T' in start_raw:
            start_dt = parse_dt(start_raw)
            time_label = start_dt.strftime("%I:%M %p").lstrip("0")
        else:
            # All-day event -- Google gives just a date, no time.
            start_dt = datetime.fromisoformat(start_raw).replace(tzinfo=LOCAL_TZ)
            time_label = "All day"

        day_label = start_dt.strftime("%A, %b %d")
        if day_label != current_day_label:
            lines.append(f"\n{day_label}")
            current_day_label = day_label

        summary = event.get('summary', '(No title)')
        location = event.get('location')

        line = f"  • {time_label} — {summary}"
        if location:
            line += f" @ {location}"
        lines.append(line)

    return "This week:\n" + "\n".join(lines).strip()


def send_digest_email(body_text):
    """
    Sends the digest via Gmail SMTP. Requires these env vars on Render:
      SMTP_EMAIL         - the Gmail address sending FROM
      SMTP_APP_PASSWORD  - a Gmail "App Password" (not your normal password)
      DIGEST_EMAIL_TO    - optional; defaults to SMTP_EMAIL if not set

    If these aren't configured, this just logs a message and skips sending
    rather than crashing the request.
    """
    smtp_email = os.getenv("SMTP_EMAIL")
    smtp_password = os.getenv("SMTP_APP_PASSWORD")
    digest_to = os.getenv("DIGEST_EMAIL_TO", smtp_email)

    if not smtp_email or not smtp_password:
        print("[INFO] SMTP_EMAIL/SMTP_APP_PASSWORD not set -- skipping digest email.")
        return

    msg = MIMEText(body_text)
    msg["Subject"] = "Your Week Ahead"
    msg["From"] = smtp_email
    msg["To"] = digest_to

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_email, smtp_password)
        server.sendmail(smtp_email, [digest_to], msg.as_string())


@app.get("/digest")
async def weekly_digest(x_auth_token: str = Header(None)):
    if x_auth_token != os.getenv("SECRET_TOKEN"):
        raise HTTPException(status_code=401)

    digest_text = build_weekly_digest_text()

    # Try to email it too, but don't let an email failure break the
    # notification response your Shortcut is waiting on.
    try:
        send_digest_email(digest_text)
    except Exception as e:
        print(f"[WARN] Failed to send digest email: {e}")

    return {
        "notification_title": "This Week",
        "message": digest_text
    }


def create_calendar_event(sender, title, activity, location, time_str, duration):
    """
    Creates (or patches) the actual Google Calendar event. Pulled out into
    its own function so both the immediate-confirm flow (webhook) and the
    confirm-a-pending-one-later flow (/pending/{id}/confirm) can share the
    exact same logic instead of duplicating it.
    """
    service = get_google_service("calendar", "v3")
    start_dt = parse_dt(time_str)
    end_dt = start_dt + timedelta(minutes=duration)

    clean_activity = re.sub(r'[?.,!]', '', activity.strip().title()).strip().title()

    event_body = {
        "summary": f"{clean_activity} w/ {sender}",
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": TIMEZONE_NAME
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": TIMEZONE_NAME
        }
    }

    if location:
        event_body["location"] = location

    existing_event_id = get_last_event_id(sender, title)

    if existing_event_id:
        try:
            updated_event = service.events().patch(
                calendarId="primary",
                eventId=existing_event_id,
                body=event_body
            ).execute()
            new_event_id = updated_event["id"]
        except Exception:
            new_event = service.events().insert(
                calendarId="primary",
                body=event_body
            ).execute()
            new_event_id = new_event["id"]

            save_message(
                sender,
                "Assistant",
                f"Re-created event '{title}' due to patch failure: {new_event_id}",
                event_id=new_event_id
            )
            return new_event_id
    else:
        new_event = service.events().insert(
            calendarId="primary",
            body=event_body
        ).execute()
        new_event_id = new_event["id"]

    save_message(
        sender,
        "Assistant",
        f"Created/Updated event '{title}': {new_event_id}",
        event_id=new_event_id
    )
    return new_event_id


@app.get("/pending")
async def list_pending(x_auth_token: str = Header(None)):
    if x_auth_token != os.getenv("SECRET_TOKEN"):
        raise HTTPException(status_code=401)

    rows = get_pending_invites()

    if not rows:
        # Deliberately quiet when there's nothing outstanding, so a daily
        # automation calling this doesn't nag you every single day.
        return {"has_pending": False, "message": "Nothing pending -- you're all caught up."}

    lines = []
    for pending_id, sender, activity, time_str, location in rows:
        dt_obj = parse_dt(time_str)
        friendly_time = dt_obj.strftime("%a %I:%M %p") if dt_obj else "time unknown"
        line = f"#{pending_id}: {activity} w/ {sender} — {friendly_time}"
        if location:
            line += f" @ {location}"
        lines.append(line)

    return {
        "has_pending": True,
        "notification_title": "Pending Invites",
        "message": "Still waiting on your yes/no:\n" + "\n".join(lines)
    }


@app.post("/pending/{pending_id}/confirm")
async def confirm_pending(pending_id: int, x_auth_token: str = Header(None)):
    if x_auth_token != os.getenv("SECRET_TOKEN"):
        raise HTTPException(status_code=401)

    row = get_pending_invite_by_id(pending_id)
    if not row:
        raise HTTPException(status_code=404, detail="No pending invite with that ID")

    sender, title, activity, location, time_str, duration, confidence = row

    new_event_id = create_calendar_event(
        sender=sender,
        title=title,
        activity=activity,
        location=location,
        time_str=time_str,
        duration=duration
    )
    mark_pending_status(pending_id, "confirmed")

    return {"status": "success", "event_id": new_event_id}


@app.post("/pending/{pending_id}/dismiss")
async def dismiss_pending(pending_id: int, x_auth_token: str = Header(None)):
    if x_auth_token != os.getenv("SECRET_TOKEN"):
        raise HTTPException(status_code=401)

    row = get_pending_invite_by_id(pending_id)
    if not row:
        raise HTTPException(status_code=404, detail="No pending invite with that ID")

    mark_pending_status(pending_id, "dismissed")
    return {"status": "dismissed"}


@app.post("/webhook")
async def handle_webhook(request: Request, x_auth_token: str = Header(None)):
    if x_auth_token != os.getenv("SECRET_TOKEN"):
        raise HTTPException(status_code=401)
    
    data = await request.json()
    text = data.get("text", "")
    sender = data.get("sender", "Unknown")
    confirmed = data.get("confirmed", False)

    # This webhook fires when you text yourself (e.g. a self-reminder) --
    # in that case some SMS bridges echo the sender field back as your own
    # message text instead of a real name. Detect that and swap in the
    # configured SELF_NAME instead.
    if sender.lower() == text.lower():
        sender = SELF_NAME

    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    
    now = datetime.now(LOCAL_TZ)
    current_context = now.strftime("%A, %B %d, %Y %I:%M %p")
    conversation_history = get_history(sender)
    
    # FIX: this is an f-string (f"""..."""), so every literal { or }
    # must be doubled ({{ / }}) or Python tries to evaluate it as an
    # expression -- which caused a second SyntaxError here, since the JSON
    # example below isn't valid Python. The braces are now escaped.
    prompt = f"""
    Current Context: Today is {current_context}.
    Recent Conversation History with {sender} (Last 6 Hours):
    {conversation_history}
 
    New Message from {sender}: "{text}"
 
    Logic:
    1. Use History to resolve follow-ups or corrections.
    2. Resolve relative dates using Current Context. If the user mentions "tomorrow", determine the exact YYYY-MM-DD date.
    3. If the message explicitly gives a specific time (e.g., "8pm"), return the full ISO8601 string in the "time" field.
    4. If a date is known or inferred (like tomorrow) but NO specific time/hour is given, return the date as "YYYY-MM-DD" in the "time" field, or return null if completely unknown.
    Do NOT invent a random hour like 00:00 or 12:00 AM.
    5. Durations:
    Meals = 60 minutes
    Sports = 90 minutes
    Calls = 30 minutes
 
    6. If the new message is about a completely different activity or separate event than what is discussed in the history, ignore the history and treat this as a brand-new event.
    7. Lower confidence if you must infer missing information.
    8. Classify as "invitation" any message describing a plan or activity involving the sender and recipient together -- this includes actual invitations phrased as questions ("Want to grab dinner at 7?") AND plain statements announcing an already-decided plan ("We are going to dinner at 7", "I'll pick you up at 6", "Movie starts at 8"). The message does not need to ask permission or use question phrasing to count as an "invitation". Only use "none" for messages that don't describe any calendar-worthy plan at all.
 
    Return ONLY JSON:

    {{
    "type": "invitation/reminder/none",

    "activity": "ONLY the activity name. Examples: 'Basketball', 'Ice Cream', 'Dinner', 'Coffee'. Never include dates, times, locations, or words like 'tomorrow', 'next Friday', 'at 7pm', etc.",

    "title": "Natural language summary",

    "location": "null if unknown",

    "time": "ISO8601_or_YYYY-MM-DD_or_null",

    "duration": int,

    "confidence": int
    }}
    """
    
    res = llm.invoke(prompt).content
 
    print("\n========== RAW AI RESPONSE ==========")
    print(res)
    print("=====================================\n")
 
    # Save the user's text to history for context
    save_message(sender, "User", text)
 
    try:
        match = re.search(r'\{.*\}', res, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in AI response")
        analysis = json.loads(match.group())
        if not isinstance(analysis, dict):
            raise ValueError("Parsed JSON was not an object")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[WARN] Failed to parse AI response as JSON: {e}")
        return {
            "type": "none",
            "error": "Sorry, I couldn't understand that message well enough to make an event."
        }
 
    analysis["type"] = analysis.get("type") or "none"
    analysis["title"] = analysis.get("title") or ""
    analysis["activity"] = analysis.get("activity") or analysis["title"]

    # The AI is told to return "null if unknown" -- that comes back as the
    # literal string "null" (or sometimes an empty string), not Python's
    # None. Normalize all of those to a real None so the rest of the code
    # can just check "if analysis['location']:".
    raw_location = analysis.get("location")
    if raw_location and raw_location.strip().lower() not in ("null", "none", ""):
        analysis["location"] = raw_location.strip()
    else:
        analysis["location"] = None
    analysis["time"] = analysis.get("time")
    analysis["duration"] = analysis.get("duration") or 0
    analysis["confidence"] = analysis.get("confidence", 50)
 
    if isinstance(analysis.get("confidence"), (int, float)) and analysis["confidence"] <= 1.0:
        analysis["confidence"] = int(analysis["confidence"] * 100)
    
    # ------------------------------
    # Apply common-sense defaults
    # ------------------------------
    if analysis["time"] is None or len(analysis["time"]) <= 10:
        activity = analysis["title"].lower()
        inferred_date = analysis["time"] if (analysis["time"] and len(analysis["time"]) == 10) else None
 
        for key in DEFAULT_EVENT_TIMES:
            if re.search(r'\b' + re.escape(key) + r'\b', activity):
                default_time, default_duration = DEFAULT_EVENT_TIMES[key]
                date_only = inferred_date if inferred_date else datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
 
                analysis["time"] = f"{date_only}T{default_time}:00"
 
                if analysis["duration"] == 0:
                    analysis["duration"] = default_duration
 
                analysis["confidence"] = min(analysis.get("confidence", 85), 85)
                break
                
    if analysis["type"] == "none":
        return analysis
 
    if not analysis["time"]:
        analysis["type"] = "none"
        analysis["error"] = "I couldn't figure out a time for this, so I didn't create anything."
        return analysis
 
    # Check calendar conflicts
    conflicts = check_conflicts(analysis["time"], analysis["duration"])
    analysis["conflicts_str"] = ", ".join(conflicts) if conflicts else "None"
    analysis["sender"] = sender
 
    dt_obj = parse_dt(analysis["time"])
    friendly_time = dt_obj.strftime("%A at %I:%M %p")
 
    analysis["notification_title"] = f"New {analysis['type'].capitalize()}"

    # Only add a location line to the notification if we actually found one.
    location_line = f"📍 {analysis['location']}\n" if analysis["location"] else ""

    analysis["message"] = (
        f"{sender} has invited you to {analysis['activity']}\n"
        f"📅 {friendly_time}\n"
        f"{location_line}"
        f"⚠️ Conflicts: {analysis['conflicts_str']}\n"
        f"🎯 Confidence: {analysis['confidence']}%\n\n"
        f"Add to calendar?"
    )
 
    print("\n========== FINAL ANALYSIS SENT TO SHORTCUT ==========")
    print(json.dumps(analysis, indent=2))
    print("====================================================\n")

    # Save this as a pending invite the moment it's understood -- BEFORE
    # checking whether it was confirmed. This is what makes it possible to
    # check /pending later and still act on it, even if the notification
    # for THIS exact request gets missed.
    if analysis["type"] == "invitation":
        pending_id = save_pending_invite(analysis)
        analysis["pending_id"] = pending_id

    if not confirmed:
        return analysis
 
    # -------------------------------
    # Calendar Creation / Update
    # -------------------------------
    if analysis["type"] == "invitation":
        new_event_id = create_calendar_event(
            sender=sender,
            title=analysis["title"],
            activity=analysis["activity"],
            location=analysis["location"],
            time_str=analysis["time"],
            duration=analysis["duration"]
        )

        # This invite is no longer "waiting on you" -- mark it confirmed so
        # it drops off the /pending list.
        if "pending_id" in analysis:
            mark_pending_status(analysis["pending_id"], "confirmed")

    return {**analysis, "status": "success"}
