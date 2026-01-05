import os
import streamlit as st
from moviepy.editor import AudioFileClip
from dotenv import load_dotenv
from groq import Groq
from podcast.speech_to_text import audio_to_text
from podcast.embedding import store_embeddings
from podcast.question_answer import query_vector_database, transcript_chat_completion
from langchain_core.documents import Document


# Load environment variables from .env file
load_dotenv()
API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# Ensure the API_KEY is loaded
if API_KEY is None:
    st.error("API key not found. Please set the GROQ_API_KEY in your .env file.")
    st.stop()

# Initialize the Groq client
client = Groq(api_key=API_KEY)


# Ensure output directories exist
mp3_file_folder = "uploaded_files"
mp3_chunk_folder = "chunks"
os.makedirs(mp3_file_folder, exist_ok=True)
os.makedirs(mp3_chunk_folder, exist_ok=True)

st.title("Podcast Q&A App")

# Upload audio file
uploaded_file = st.file_uploader("Upload an MP3 file", type="mp3")

# Session state to store the last processed file to avoid reprocessing
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = []
if "docsearch" not in st.session_state:
    st.session_state.docsearch = None

# When a new file is uploaded, reset the session state for transcriptions and embeddings
if uploaded_file is not None:
    # Check if the new file is different from the last processed file
    if uploaded_file.name != st.session_state.last_uploaded_file:
        st.session_state.transcriptions = []  # Reset transcriptions
        st.session_state.docsearch = None  # Reset embeddings
        st.session_state.last_uploaded_file = uploaded_file.name  # Update the last file

    # Save the uploaded file
    filepath = os.path.join(mp3_file_folder, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Optional: To save and chunk the audio file
    audio = AudioFileClip(filepath)
    chunk_length = 60  # seconds

    # Process and transcribe each chunk only if not already done
    if not st.session_state.transcriptions:
        for start in range(0, int(audio.duration), chunk_length):
            end = min(start + chunk_length, int(audio.duration))
            audio_chunk = audio.subclip(start, end)
            chunk_filename = os.path.join(mp3_chunk_folder, f"chunk_{start}.mp3")
            audio_chunk.write_audiofile(chunk_filename)

            # Process and transcribe each chunk using Groq
            transcription = audio_to_text(chunk_filename)
            st.session_state.transcriptions.append(transcription)  # Collect transcriptions

        # Combine all transcriptions into a single text
        combined_transcription = " ".join(st.session_state.transcriptions)
        st.write(f"Transcription: {combined_transcription[:500]}...")  # Show the first 500 characters

        # Generate embeddings and store in Pinecone
        documents = [Document(page_content=combined_transcription)]
        st.session_state.docsearch = store_embeddings(documents)

# User query
user_question = st.text_input("Ask a question about the podcast")
if user_question and st.session_state.docsearch:
    relevant_transcripts = query_vector_database(st.session_state.docsearch, user_question)
    response = transcript_chat_completion(client, relevant_transcripts, user_question)
    st.write(f"Response: {response}")