from __future__ import annotations

import ssl
import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional

import certifi
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv  
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aqylai")

GH_TOKEN = os.getenv("GH_MODELS_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
GH_MODEL = os.getenv("GH_MODELS_MODEL", "openai/gpt-4o-mini")
GH_ORG = os.getenv("GH_MODELS_ORG") 
GH_BASE = (os.getenv("GH_MODELS_BASE") or "https://models.github.ai").rstrip("/")

VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() != "false"
VERIFY_SSL_MODE = os.getenv("VERIFY_SSL_MODE", "certifi").lower()  

def build_verify() -> Any:
    """
    httpx verify parameter:
      - False: disable verification (NOT recommended)
      - str path: CA bundle path
      - ssl.SSLContext: custom context
    """
    if not VERIFY_SSL:
        return False
    if VERIFY_SSL_MODE == "system":
        return ssl.create_default_context()
    return certifi.where()

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5173,http://localhost:5173",
)
ALLOWED_ORIGINS = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]

if not GH_TOKEN:
    raise RuntimeError("No token found. Set GH_MODELS_TOKEN (or GITHUB_TOKEN/GITHUB_PAT).")

def gh_url() -> str:
    if GH_ORG:
        return f"{GH_BASE}/orgs/{GH_ORG}/inference/chat/completions"
    return f"{GH_BASE}/inference/chat/completions"

app = FastAPI(title="AqylAI Tutor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Role = Literal["system", "developer", "user", "assistant"]

class ChatMsg(BaseModel):
    role: Role
    content: str

class AIRequest(BaseModel):
    taskText: Optional[str] = None
    expectedAnswer: Optional[str] = None
    userAnswer: Optional[str] = None
    userMessage: Optional[str] = None
    history: List[ChatMsg] = Field(default_factory=list)
    lang: str = "en"

class AIResponse(BaseModel):
    correct: Optional[bool] = None
    reply: str
    short_hint: Optional[str] = None
    steps: List[str] = Field(default_factory=list)
    final_answer: Optional[str] = None

def norm(s: str) -> str:
    return re.sub(r"\s+", "", s).strip().lower()

def clamp_history(history: List[ChatMsg], max_items: int = 14) -> List[ChatMsg]:
    return history[-max_items:] if len(history) > max_items else history

def safe_json_parse(content: str) -> Dict[str, Any]:
    c = (content or "").strip()
    c = re.sub(r"^\s*```(?:json)?\s*", "", c)
    c = re.sub(r"\s*```\s*$", "", c)

    try:
        return json.loads(c)
    except Exception:
        pass

    m = re.search(r"\{.*\}", c, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return {
        "correct": None,
        "reply": c[:2000] if c else "Empty model response",
        "short_hint": None,
        "steps": [],
        "final_answer": None,
    }

def is_asking_final(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    patterns = [
        r"\bsolve\b", r"\bgive me the answer\b", r"\bfinal answer\b",
        r"\bреши\b", r"\bдай ответ\b", r"\bответ\b", r"\bреши полностью\b",
        r"\bшешіп бер\b", r"\bшығарып бер\b", r"\bтолық жауап\b",
    ]
    return any(re.search(p, t) for p in patterns)

def looks_like_attempt(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if re.search(r"[=]", t) and re.search(r"[0-9]", t):
        return True
    if re.search(r"\(.*\)\(.*\)", t):
        return True
    if re.search(r"\b(x|u|y)\s*=\s*[-+]?\d", t, flags=re.I):
        return True
    return False

ZPD_SYSTEM = """
You are a tutor using Lev Vygotsky's Zone of Proximal Development (ZPD).
You must help the student with ONE short hint only.
Never output step-by-step solutions.
""".strip()

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model": GH_MODEL,
        "base": GH_BASE,
        "org": GH_ORG,
        "verify_ssl": VERIFY_SSL,
        "verify_ssl_mode": VERIFY_SSL_MODE,
        "cors_origins": ALLOWED_ORIGINS,
    }

@app.post("/api/ai", response_model=AIResponse)
async def ai(req: AIRequest) -> AIResponse:
    correct: Optional[bool] = None
    if req.expectedAnswer and req.userAnswer:
        correct = (norm(req.expectedAnswer) == norm(req.userAnswer))

    asked_final = is_asking_final(req.userMessage or "")
    has_attempt = bool((req.userAnswer or "").strip())
    for m in req.history:
        if m.role == "user" and looks_like_attempt(m.content):
            has_attempt = True
            break
    allow_final = bool(asked_final and has_attempt)

    lang = (req.lang or "en").lower().strip()

    context_lines = [
        f"LANG={lang}",
        f"TASK={req.taskText or '—'}",
        f"EXPECTED={req.expectedAnswer or '—'}",
        f"STUDENT_ANSWER={req.userAnswer or '—'}",
        f"STUDENT_QUESTION={req.userMessage or '—'}",
        f"SERVER_CORRECT={correct}",
        f"ALLOW_FINAL={allow_final}",
        "You MUST return a valid json object only (no markdown, no extra text).",
        "Put ONE short hint sentence into key 'reply'.",
        "Set 'steps' to an empty array []. Set 'short_hint' to null. Set 'final_answer' to null.",
        "Keys: correct, reply, short_hint, steps, final_answer.",
    ]

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": ZPD_SYSTEM},
        {"role": "developer", "content": "\n".join(context_lines)},
    ]

    for m in clamp_history(req.history):
        messages.append({"role": m.role, "content": m.content})

    user_msg = req.userMessage or ("Please give me one short hint." if lang == "en" else "Маған бір қысқа подсказка берші.")
    messages.append({"role": "user", "content": user_msg})

    payload: Dict[str, Any] = {
        "model": GH_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 220,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GH_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }

    verify_arg: Any = build_verify()

    try:
        async with httpx.AsyncClient(timeout=45, verify=verify_arg, trust_env=True) as client:
            r = await client.post(gh_url(), headers=headers, json=payload)

        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"GitHub Models error {r.status_code}: {r.text}")

        data = r.json()
        content = data["choices"][0]["message"]["content"]
        obj = safe_json_parse(content)

        reply_text = (str(obj.get("reply", "") or "")).strip()
        if not reply_text:
            reply_text = "Бір қысқа подсказка: есепті жарты+жарты логикасымен ойлап көр."

        return AIResponse(
            correct=obj.get("correct", correct),
            reply=reply_text,
            short_hint=None,
            steps=[],
            final_answer=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("AI error")
        fallback = (
            "Sorry — server error occurred while contacting GitHub Models.\n"
            "Try again, or check token/model/SSL settings."
        )
        return AIResponse(
            correct=correct,
            reply=fallback,
            short_hint=str(e),
            steps=[],
            final_answer=None,
        )
