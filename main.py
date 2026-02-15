from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GitHub Models (из окружения/.env)
GH_TOKEN = os.getenv("GH_MODELS_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
GH_MODEL = os.getenv("GH_MODELS_MODEL", "openai/gpt-4o-mini")
GH_ORG = os.getenv("GH_MODELS_ORG")  # только если доступ через org


def gh_url() -> str:
    if GH_ORG:
        return f"https://models.github.ai/orgs/{GH_ORG}/inference/chat/completions"
    return "https://models.github.ai/inference/chat/completions"


class AIRequest(BaseModel):
    taskText: Optional[str] = None
    userAnswer: Optional[str] = None
    expectedAnswer: Optional[str] = None
    lang: Optional[str] = "kk"
    extra: Optional[str] = None


def norm(s: Optional[str]) -> str:
    return (s or "").strip().replace(" ", "").lower()


def ensure_schema(d: Dict[str, Any]) -> Dict[str, Any]:
    d.setdefault("correct", False)
    d.setdefault("steps", [])
    d.setdefault("short_hint", "")
    d.setdefault("final_answer", "")
    d.setdefault("error_type", "logic")
    d.setdefault("next_task_suggestion", "")
    return d


def extract_json(text: str) -> str:
    if not text:
        return text
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e < s:
        return text
    return text[s:e + 1]


def parse_llm_output(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}

    # 1) JSON целиком
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) JSON внутри текста
    try:
        obj = json.loads(extract_json(raw))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) fallback: текст -> steps
    lines: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("-•* \t")
        lines.append(line)

    steps = lines[:8]
    return {
        "steps": steps,
        "short_hint": steps[0] if steps else raw[:120],
        "final_answer": "",
        "error_type": "logic",
        "next_task_suggestion": "",
    }


async def call_github_models(messages: List[Dict[str, str]]) -> str:
    if not GH_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="GH_MODELS_TOKEN not set. Put GitHub PAT (Models: Read) into backend/.env or env vars.",
        )

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GH_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": GH_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }

    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(gh_url(), headers=headers, json=payload)

        # если 400 из-за response_format — пробуем без него
        if r.status_code == 400:
            payload2 = dict(payload)
            payload2.pop("response_format", None)
            r = await client.post(gh_url(), headers=headers, json=payload2)

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"raw": r.text}
        raise HTTPException(status_code=502, detail={"github_models": err})

    data = r.json()
    return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()


@app.post("/api/ai")
async def ai_help(data: AIRequest):
    language = "Kazakh" if (data.lang or "kk") == "kk" else "Russian"

    system_msg = f"""
You are a professional math tutor for grades 7–11.

STRICT RULES:
- Return ONLY a JSON object (no markdown, no code fences).
- No LaTeX/backslashes "\\" in values. Use plain text math: x^2, (1/2), sqrt(2).
- Keep steps clear and short.

Language: {language}

Return JSON:
{{
  "correct": true or false,
  "steps": ["step 1", "step 2"],
  "short_hint": "short hint",
  "final_answer": "answer",
  "error_type": "formula | arithmetic | logic | none",
  "next_task_suggestion": "harder variation if correct"
}}
""".strip()

    user_msg = f"""
Task: {data.taskText}
Student answer: {data.userAnswer}
Expected answer (if known): {data.expectedAnswer}
Student question: {data.extra}
""".strip()

    # детерминированная проверка, если expectedAnswer есть
    deterministic_correct: Optional[bool] = None
    if data.expectedAnswer is not None:
        deterministic_correct = norm(data.userAnswer) == norm(data.expectedAnswer)

    try:
        raw = await call_github_models(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        )
        result = parse_llm_output(raw)
    except Exception as e:
        print("GITHUB MODELS ERROR:", repr(e))
        result = {
            "correct": False,
            "steps": [],
            "short_hint": "GitHub Models жауап бермеді немесе қате қайтарды",
            "final_answer": data.expectedAnswer or "",
            "error_type": "logic",
            "next_task_suggestion": "",
        }

    result = ensure_schema(result)

    if deterministic_correct is not None:
        result["correct"] = deterministic_correct
        if deterministic_correct:
            result["error_type"] = "none"
            if not result.get("final_answer"):
                result["final_answer"] = data.expectedAnswer or ""

    return result


@app.get("/api/task")
def generate_task(level: int = 1):
    level = max(1, min(level, 4))

    if level == 1:
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        return {"topic": "arithmetic", "level": level, "q": f"{a} + {b} =", "a": str(a + b)}

    if level == 2:
        x = random.randint(2, 10)
        b = random.randint(1, 10)
        res = 2 * x + b
        return {"topic": "linear", "level": level, "q": f"Solve: 2x + {b} = {res}", "a": str(x)}

    if level == 3:
        t = random.randint(1, 10)
        return {"topic": "algebra", "level": level, "q": f"Simplify: (x+{t})(x-{t})", "a": f"x^2 - {t*t}"}

    return {"topic": "calculus", "level": level, "q": "Find derivative of x^2", "a": "2x"}


@app.get("/")
def health():
    return {"status": "backend works", "engine": "github-models"}
