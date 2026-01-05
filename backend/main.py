from dotenv import load_dotenv
load_dotenv()
from starlette.middleware.sessions import SessionMiddleware

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse , JSONResponse
from pathlib import Path
from pydantic import BaseModel

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

from ai_news.agent_with_tools import create_agent, get_news_repsonse
import os

from moviepy.editor import AudioFileClip
from groq import Groq

from speech_to_text.src.podcast.speech_to_text import audio_to_text
from speech_to_text.src.podcast.embedding import store_embeddings
from speech_to_text.src.podcast.question_answer import (
    query_vector_database,
    transcript_chat_completion,
)

# login/signup
#------------------------------
from starlette.middleware.sessions import SessionMiddleware

from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import SessionLocal, engine
from models import Base, User
from auth import hash_password, verify_password, create_reset_token, verify_reset_token
from email_utils import send_reset_email
from oauth import oauth
#------------------------------

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama


os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


#------------------------------
Base.metadata.create_all(bind=engine)




def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




# =========================================================
# CLIENTS
# =========================================================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# (LangChain reads Pinecone automatically)
assert os.getenv("PINECONE_API_KEY"), "PINECONE_API_KEY missing"


# =========================================================
# GLOBAL PODCAST MEMORY (🔥 IMPORTANT)
# =========================================================
PODCAST_STORE = {
    "docsearch": None,
    "transcription": None
}


# -------------------------
# APP INIT
# -------------------------
app = FastAPI(title="AI-NEXUS API")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Session Middleware (REQUIRED FOR OAUTH)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET"),  
    https_only=False  # set True in production
)



BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
# BASE_DIR = Path(__file__).resolve().parent
# FRONTEND_DIR = BASE_DIR / "frontend"
templates = Jinja2Templates(directory=str(FRONTEND_DIR))

# -------------------------
# STATIC FILES
# -------------------------
app.mount(
    "/assets",
    StaticFiles(directory=FRONTEND_DIR / "assets"),
    name="assets"
)

app.mount(
    "/media",
    StaticFiles(directory=BASE_DIR / "media"),
    name="media"
)

UPLOAD_DIR = BASE_DIR / "uploaded_files"
CHUNK_DIR = BASE_DIR / "chunks"

UPLOAD_DIR.mkdir(exist_ok=True)
CHUNK_DIR.mkdir(exist_ok=True)

def load_html(file_name: str):
    path = FRONTEND_DIR / file_name
    return HTMLResponse(path.read_text(encoding="utf-8"))

# -------------------------
# IMPORT PODCAST LOGIC
# -------------------------
from speech_to_text.src.podcast.speech_to_text import audio_to_text
from speech_to_text.src.podcast.embedding import store_embeddings
from speech_to_text.src.podcast.question_answer import (
    query_vector_database,
    transcript_chat_completion,
)

from langchain_core.documents import Document



# -------------------------
# ROUTES (PAGES)
# -------------------------
@app.get("/")
def index():
    return load_html("index.html")

@app.get("/home")
def home():
    return load_html("home.html")

@app.get("/news")
def news():
    return load_html("ai_news.html")

@app.get("/image-captioning")
def image_captioning():
    return load_html("image_captioning.html")

@app.get("/podcast")
def podcast():
    return load_html("podcast.html")

@app.get("/text-to-image")
def text_to_image():
    return load_html("text_to_image.html")

@app.get("/about")
def about():
    return load_html("about.html")

@app.get("/contact")
def contact():
    return load_html("contact.html")

@app.get("/sign-in")
def sign_in(request: Request):
    return templates.TemplateResponse("sign_in.html", {"request": request})

@app.get("/signup")
def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/reset-password/{token}")
def reset_form(request: Request, token: str):
    return templates.TemplateResponse("reset_password.html", {"request": request, "token": token})

@app.get("/nexusgpt", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("NexusGpt.html", {"request": request})


# -------------------------
# IMAGE CAPTIONING
# -------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

@app.post("/api/image-caption")
async def image_caption(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return {"caption": caption}

# -------------------------
# AI NEWS API
# -------------------------
agent = create_agent()

class NewsRequest(BaseModel):
    prompt: str

@app.post("/api/generate-news")
def generate_news(req: NewsRequest):
    news = get_news_repsonse(agent, req.prompt)
    return {"news": news}
import re
import uuid
import math
from pathlib import Path
from gtts import gTTS
from pydantic import BaseModel
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips
)

VIDEO_TEMPLATE = BASE_DIR / "news.mp4"
AUDIO_DIR = BASE_DIR / "media/audio"
VIDEO_DIR = BASE_DIR / "media/video"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


class NewsMediaRequest(BaseModel):
    prompt: str

def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)


def format_news_text(raw_text: str):
    """
    Returns:
    - headline (string)
    - bullet_points (list)
    - speech_text (string, clean for TTS)
    """
    raw_text = remove_emojis(raw_text)
    # Remove markdown symbols
    text = re.sub(r'[*_`#]', '', raw_text)

    # Remove dates like (2026-01-01)
    text = re.sub(r'\(\d{4}-\d{2}-\d{2}\)', '', text)

    parts = [p.strip() for p in text.split('-') if p.strip()]

    headline = parts[0]
    bullet_points = parts[1:]

    # Speech should be clean & natural
    speech_text = headline + ". " + " ".join(bullet_points)

    return headline, bullet_points, speech_text


@app.post("/api/news-with-media")
def generate_news_with_media(req: NewsMediaRequest):

    # 1️⃣ Generate AI news
    news_text = get_news_repsonse(agent, req.prompt)

    headline, bullet_points, speech_text = format_news_text(news_text)

    uid = uuid.uuid4().hex

    # 2️⃣ Text → Speech
    audio_path = AUDIO_DIR / f"{uid}.mp3"
    tts = gTTS(speech_text)
    tts.save(str(audio_path))

    # 3️⃣ Sync audio with looping anchor video
    video = VideoFileClip(str(VIDEO_TEMPLATE))
    audio = AudioFileClip(str(audio_path))

    loops_required = math.ceil(audio.duration / video.duration)
    looped_video = concatenate_videoclips([video] * loops_required)

    final_video = looped_video.subclip(0, audio.duration)
    final_video = final_video.set_audio(audio)

    video_path = VIDEO_DIR / f"{uid}.mp4"
    final_video.write_videofile(
        str(video_path),
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None
    )

    return {
        "headline": headline,
        "points": bullet_points,
        "audio_url": f"/media/audio/{uid}.mp3",
        "video_url": f"/media/video/{uid}.mp4"
    }
 
 
 
 
 
# =========================================================
# PODCAST TRANSCRIPTION (🔥 FIXED)
# =========================================================
@app.post("/api/podcast/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    path = UPLOAD_DIR / file.filename
    path.write_bytes(await file.read())

    audio = AudioFileClip(str(path))
    parts = []

    for start in range(0, int(audio.duration), 60):
        chunk = audio.subclip(start, min(start + 60, audio.duration))
        chunk_path = CHUNK_DIR / f"{uuid.uuid4().hex}.mp3"
        chunk.write_audiofile(str(chunk_path), logger=None)

        parts.append(audio_to_text(str(chunk_path)))

    combined_text = " ".join(parts)

    # 🔥 STORE EMBEDDINGS ONCE
    docs = [Document(page_content=combined_text)]
    PODCAST_STORE["docsearch"] = store_embeddings(docs)
    PODCAST_STORE["transcription"] = combined_text

    return {"transcription": combined_text}


# =========================================================
# PODCAST Q&A (🔥 FIXED)
# =========================================================
class PodcastQuestion(BaseModel):
    question: str

@app.post("/api/podcast/ask")
def ask_question(payload: PodcastQuestion):

    if PODCAST_STORE["docsearch"] is None:
        return {"answer": "Please upload and transcribe a podcast first."}

    relevant = query_vector_database(
        PODCAST_STORE["docsearch"],
        payload.question
    )

    if not relevant:
        return {"answer": "No relevant information found in the podcast."}

    answer = transcript_chat_completion(
        client,
        relevant,
        payload.question
    )

    return {"answer": answer}



# ---------- SIGNUP ----------

@app.post("/signup")
def signup_user(
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    if password != confirm_password:
        return HTMLResponse("Passwords do not match", 400)

    if db.query(User).filter_by(email=email).first():
        return HTMLResponse("User already exists", 400)

    user = User(
        email=email,
        hashed_password=hash_password(password),
        provider="local",
    )
    db.add(user)
    db.commit()
    return RedirectResponse("/sign-in", 302)


# ---------- LOGIN ----------

@app.post("/sign-in")
def login_user(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
   user = db.query(User).filter_by(email=email).first()

   if not user:
       return HTMLResponse("Invalid credentials", 401)
   
   if user.provider != "local":
       return HTMLResponse("Please login using Google/Facebook", 400)
   
   if not verify_password(password, user.hashed_password):
       return HTMLResponse("Invalid credentials", 401)

   return RedirectResponse(url="/home", status_code=302)

# ---------- FORGOT PASSWORD ----------

@app.post("/forgot-password")
def forgot_password(email: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=email).first()
    if not user:
        return HTMLResponse("No account found", 404)

    token = create_reset_token(email)
    reset_link = f"http://localhost:8000/reset-password/{token}"

    send_reset_email(email, reset_link)
    return HTMLResponse("Reset link sent. Check console.")


@app.get("/reset-password/{token}", response_class=HTMLResponse)
def reset_form(request: Request, token: str):
    return templates.TemplateResponse(
        "reset_password.html", {"request": request, "token": token}
    )


@app.post("/reset-password/{token}")
def reset_password(
    token: str,
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    email = verify_reset_token(token)
    user = db.query(User).filter_by(email=email).first()
    user.hashed_password = hash_password(password)
    db.commit()
    return RedirectResponse("/sign-in", 302)


# ---------- GOOGLE OAUTH ----------

@app.get("/auth/google")
async def google_login(request: Request):
    return await oauth.google.authorize_redirect(request, "http://localhost:8000/auth/google/callback")

@app.get("/auth/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    token = await oauth.google.authorize_access_token(request)
    user_info = token["userinfo"]

    email = user_info["email"]
    user = db.query(User).filter_by(email=email).first()

    if not user:
        user = User(email=email, provider="google")
        db.add(user)
        db.commit()

    return RedirectResponse("/home")


# ---------- FACEBOOK OAUTH ----------

# @app.get("/auth/facebook")
# async def facebook_login(request: Request):
#     return await oauth.facebook.authorize_redirect(request, "http://localhost:8000/auth/facebook/callback")

# @app.get("/auth/facebook/callback")
# async def facebook_callback(request: Request, db: Session = Depends(get_db)):
#     token = await oauth.facebook.authorize_access_token(request)
#     resp = await oauth.facebook.get("me?fields=id,email")
#     email = resp.json().get("email")

#     user = db.query(User).filter_by(email=email).first()
#     if not user:
#         user = User(email=email, provider="facebook")
#         db.add(user)
#         db.commit()

#     return RedirectResponse("/home")


# =========================
# LangChain setup
# =========================
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are NexusGPT, a professional enterprise-grade AI assistant.

RESPONSE RULES (STRICT):
1. Always respond in clean, professional Markdown.
2. If the answer contains code:
   - Start with a clear title.
   - Provide a short professional summary.
   - Provide the code in a fenced code block with correct language.
   - Follow with a structured explanation using bullet points.
   - End with usage or execution steps if applicable.
3. If the answer is NOT code:
   - Use headings.
   - Use bullet points or numbered sections.
   - Keep tone formal, concise, and well-organized.
4. Never mix code inside paragraphs.
5. Never include emojis.
6. Never include casual language.
7. Output must look suitable for official documentation or a technical report.
"""
        ),
        ("user", "{question}")
    ]
)


llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser



# =========================
# Chat API Route
# =========================
@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    user_question = data.get("question")

    if not user_question:
        return JSONResponse({"error": "Question is required"}, status_code=400)

    response = chain.invoke({"question": user_question})
    return {"answer": response}