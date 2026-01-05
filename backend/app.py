import streamlit as st
import os
import base64
from PIL import Image
import requests
from io import BytesIO
import re

# ==== NEWS IMPORTS ====
from ai_news.agent_with_tools import create_agent, get_news_repsonse


# ==== SPEECH-TO-TEXT IMPORTS ====

from moviepy.editor import AudioFileClip, VideoFileClip,concatenate_videoclips
from dotenv import load_dotenv
from groq import Groq
from speech_to_text.src.podcast.speech_to_text import audio_to_text
from speech_to_text.src.podcast.embedding import store_embeddings
from speech_to_text.src.podcast.question_answer import query_vector_database, transcript_chat_completion
from langchain_core.documents import Document

load_dotenv()

pinecone_api = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

# ==== IMAGE CAPTIONING MODEL (BLIP) ====
from transformers import BlipProcessor, BlipForConditionalGeneration

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


# -----------------------------------------
#           MAIN APP CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="AI-NEXUS",
    page_icon="🤖",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("AI-NEXUS")
app_choice = st.sidebar.radio(
    "Choose a Module",
    ["AI News Reporter", "Image Captioning", "Speech to Text Q&A", "Text to Image Generator"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Created by **Mehfuz Alam** ❤️")

# -----------------------------------------
#               1️⃣ AI NEWS
# -----------------------------------------

import streamlit as st
from ai_news.agent_with_tools import create_agent, get_news_repsonse
from gtts import gTTS
import requests
import json
import os
from dotenv import load_dotenv
import math



# -------------------------
# AI News Reporter Module
# -------------------------
video_file = "news.mp4"
if app_choice == "AI News Reporter":
    st.title("🎙️ AI News Reporter with AI Anchor")
    st.write("Generate news and play it with a lip-synced AI anchor.")

    # Initialize agent
    if "agent" not in st.session_state:
        st.session_state.agent = create_agent()

    # Initialize news_text in session_state
    if "news_text" not in st.session_state:
        st.session_state.news_text = ""

    # News query input
    news_query = st.text_area("Enter your news topic", "Tell me about breaking news in 2025.")

    # Fetch AI news text
    if st.button("🔍 Get News"):
        if not news_query:
            st.error("Enter a news topic first!")
        else:
            with st.spinner("Fetching news..."):
                try:
                    news_text = get_news_repsonse(st.session_state.agent, news_query)
                    st.session_state.news_text = news_text
                    st.markdown(f"**News Text:**\n{news_text}")
                except Exception as e:
                    st.error(f"Error fetching news: {e}")

    # Generate AI anchor video
    if st.button("Play News with AI Voice"):
        if not st.session_state.news_text:
            st.error("Please generate news first!")
        else:
            if not news_query:
                st.error("Enter a news topic first!")
            else:
                with st.spinner("Fetching news..."):
                    try:
                        news_text = get_news_repsonse(st.session_state.agent, news_query)
                        st.session_state.news_text = news_text
                        st.markdown(f"**News Text:**\n{news_text}")
                        clean_news = re.sub(r'[*_`#>-]', '', st.session_state.news_text)
                        clean_news = re.sub(r'\s+', ' ', clean_news).strip()
            
                        # Convert to speech
                        tts = gTTS(clean_news)
                        audio_file = "news_audio.mp3"
                        tts.save(audio_file)
                
                        st.success("AI is now reading the news…")
                
                        # Play audio
                        audio_bytes = open(audio_file, "rb").read()
                        st.audio(audio_bytes, format="audio/mp3")
                        with st.spinner("Merging audio with anchor video..."):
                            video = VideoFileClip(video_file)       # Your 20 sec video
                        audio = AudioFileClip(audio_file)       # Full audio

                        video_dur = video.duration              # ~20 sec
                        audio_dur = audio.duration              # Can be 20 sec - many mins

                        # -----------------------------
                        # 🔁 LOOP VIDEO TO MATCH AUDIO
                        # -----------------------------
                        loops_required = math.ceil(audio_dur / video_dur)

                        looped_clips = [video] * loops_required
                        looped_video = concatenate_videoclips(looped_clips)

                        # Trim excess video to match exactly audio length
                        final_video_clip = looped_video.subclip(0, audio_dur)

                        # Attach audio
                        final_video = final_video_clip.set_audio(audio)

                        # Save final video
                        output_path = "final_synced_news.mp4"
                        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
                            # video = VideoFileClip(video_file)
                            # audio = AudioFileClip(audio_file)
        
                            # # Attach audio to video
                            # final_video = video.set_audio(audio)
        
                            # # Save final synced mp4
                            # output_path = "final_synced_news.mp4"
                            # final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

                        st.success("Your AI Anchor Video is Ready!")
                        st.video(output_path)
                    except Exception as e:
                        st.error(f"Error fetching news: {e}")
                        
                
            





   

# -----------------------------------------
#         2️⃣ IMAGE CAPTIONING (MERGED)
# -----------------------------------------
elif app_choice == "Image Captioning":
    st.title("🖼️ Image Captioning (BLIP Model)")
    st.write("Upload an image and get an AI-generated caption.")

    processor, model = load_blip_model()

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)

        if st.button("Generate Caption"):
            with st.spinner("AI is generating a caption..."):
                try:
                    inputs = processor(img, return_tensors="pt")
                    out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)

                    st.success("Caption Generated!")
                    st.write(f"### {caption}")

                except Exception as e:
                    st.error(f"Error generating caption: {str(e)}")

# -----------------------------------------
#       3️⃣ SPEECH TO TEXT + Q&A
# -----------------------------------------
elif app_choice == "Speech to Text Q&A":

    st.title("🎧 Podcast Q&A System")
    st.write("Upload MP3 → Transcribe → Ask Questions")

    load_dotenv()
    API_KEY =  os.getenv("GROQ_API_KEY")

    if not API_KEY:
        st.error("GROQ_API_KEY missing in .env")
        st.stop()

    client = Groq(api_key=API_KEY)

    os.makedirs("uploaded_files", exist_ok=True)
    os.makedirs("chunks", exist_ok=True)

    uploaded_file = st.file_uploader("Upload an MP3 file", type="mp3")

    if "transcriptions" not in st.session_state:
        st.session_state.transcriptions = []
    if "docsearch" not in st.session_state:
        st.session_state.docsearch = None

    if uploaded_file:
        file_path = f"uploaded_files/{uploaded_file.name}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        audio = AudioFileClip(file_path)
        chunk_len = 60  # seconds

        if not st.session_state.transcriptions:
            with st.spinner("Transcribing audio..."):
                for start in range(0, int(audio.duration), chunk_len):
                    end = min(start + chunk_len, int(audio.duration))
                    chunk_path = f"chunks/chunk_{start}.mp3"

                    audio.subclip(start, end).write_audiofile(chunk_path)

                    text = audio_to_text(chunk_path)
                    st.session_state.transcriptions.append(text)

                full_text = " ".join(st.session_state.transcriptions)
                st.write(full_text[:400] + "...")

                docs = [Document(page_content=full_text)]
                st.session_state.docsearch = store_embeddings(docs)

    user_q = st.text_input("Ask a question about the audio")

    if user_q and st.session_state.docsearch:
        with st.spinner("Thinking..."):
            rel = query_vector_database(st.session_state.docsearch, user_q)
            response = transcript_chat_completion(client, rel, user_q)
            st.write(f"**Answer:** {response}")

# -----------------------------------------
#           4️⃣ TEXT TO IMAGE
# -----------------------------------------
elif app_choice == "Text to Image Generator":

    st.title("🎨 Text to Image Generator")

    st.sidebar.subheader("Config")
    api_url = st.sidebar.text_input(
        "Backend API URL",
        "https://casemated-hazardously-elvera.ngrok-free.dev"
    )

    prompt = st.text_input("Enter your prompt", "A beautiful sunset over the mountains")

    if st.button("Generate Image"):
        if not api_url:
            st.error("Enter the API URL in sidebar first.")
        else:
            with st.spinner("Generating image..."):
                try:
                    res = requests.post(
                        f"{api_url}/generate",
                        json={"text": prompt}
                    )

                    if res.status_code == 200:
                        img_bytes = base64.b64decode(res.json()["image"])
                        image = Image.open(BytesIO(img_bytes))
                        st.image(image)
                    else:
                        st.error("Image generation failed.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# -----------------------------------------
#                 FOOTER
# -----------------------------------------
st.markdown("""
<hr>
<div style='text-align:center; color:gray'>
AI-NEXUS | Unified AI App<br>
Built with ❤️ by <b>Mehfuz Alam</b>
</div>
""", unsafe_allow_html=True)






















