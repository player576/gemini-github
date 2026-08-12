import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from github import Github

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GEMINI_API_KEY or not GITHUB_TOKEN:
    raise RuntimeError("Set GEMINI_API_KEY and GITHUB_TOKEN in .env")

app = FastAPI(title="Gemini GitHub AI")
gemini = genai.Client(api_key=GEMINI_API_KEY)
github = Github(GITHUB_TOKEN)

class ChatRequest(BaseModel):
    message: str
    repository: str

@app.get("/")
def root():
    return {"status": "ok", "name": "Gemini GitHub AI"}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        repo = github.get_repo(request.repository)
        readme = ""
        try:
            readme = repo.get_readme().decoded_content.decode("utf-8")
        except Exception:
            readme = "README not found."

        prompt = f"""You are a GitHub coding assistant.
Repository: {repo.full_name}
Repository description: {repo.description or 'none'}
README:\n{readme[:12000]}

User request:\n{request.message}

Answer in Russian. Do not claim to have changed files. In this first version you can only analyze the repository information provided above."""

        response = gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return {"answer": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
