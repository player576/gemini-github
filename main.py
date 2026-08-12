import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GEMINI_API_KEY or not GITHUB_TOKEN:
    raise RuntimeError("Set GEMINI_API_KEY and GITHUB_TOKEN in .env")

app = FastAPI(title="Gemini GitHub AI")

GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "X-GitHub-Api-Version": "2026-03-10",
}

class ChatRequest(BaseModel):
    message: str
    repository: str

@app.get("/")
def root():
    return {"status": "ok", "name": "Gemini GitHub AI"}

def get_readme(repository: str) -> str:
    url = f"https://api.github.com/repos/{repository}/readme"
    response = requests.get(url, headers=GITHUB_HEADERS, timeout=20)
    response.raise_for_status()
    data = response.json()

    import base64
    content = data.get("content", "")
    return base64.b64decode(content).decode("utf-8", errors="replace")

def ask_gemini(prompt: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
    )
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    response = requests.post(
        url,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        repo_url = f"https://api.github.com/repos/{request.repository}"
        repo_response = requests.get(repo_url, headers=GITHUB_HEADERS, timeout=20)
        repo_response.raise_for_status()
        repo = repo_response.json()

        try:
            readme = get_readme(request.repository)
        except Exception:
            readme = "README not found."

        prompt = f"""You are a GitHub coding assistant.
Repository: {repo['full_name']}
Repository description: {repo.get('description') or 'none'}
README:\n{readme[:12000]}

User request:\n{request.message}

Answer in Russian. Do not claim to have changed files. In this first version you can only analyze the repository information provided above."""

        answer = ask_gemini(prompt)
        return {"answer": answer}
    except requests.HTTPError as e:
        detail = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
