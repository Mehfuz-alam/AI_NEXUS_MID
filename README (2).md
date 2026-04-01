<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=220&section=header&text=AI%20NEXUS&fontSize=80&fontColor=ffffff&fontAlignY=38&desc=Integrated%20Multimodal%20AI%20System&descSize=24&descAlignY=62&descColor=a78bfa" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-Agents-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Stable Diffusion](https://img.shields.io/badge/Stable_Diffusion-Text_to_Image-FF6B35?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-7C3AED?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)]()

<br/>

> **"Unlocking Human Potential With Generative AI."**
>
> AI NEXUS is a full-stack, production-grade multimodal AI platform that unifies vision, language, speech, and generative intelligence into a single scalable architecture — built entirely from scratch as a deep-dive into real-world AI engineering.

<br/>

[🚀 Features](#-features) · [🖼️ Screenshots](#️-screenshots) · [🏗️ Architecture](#️-architecture) · [📦 Installation](#-installation) · [📚 Modules](#-module-breakdown) · [🔬 Research](#-applied-ai-research) · [🤝 Contributing](#-contributing)

</div>

---

## ✨ Features

AI NEXUS integrates **6 fully functional AI modules** into one unified platform:

| Module | Capability | Model / Tech |
|---|---|---|
| 🖼️ **Image Captioning** | Auto-generate natural language descriptions from images | BLIP |
| 🎨 **Image Generation** | Create images from text prompts | Stable Diffusion XL |
| 📰 **AI News Generator** | Fetch, summarize & broadcast news with voice + AI anchor video | LLM Agent + TTS + Video Synthesis |
| 🤖 **NexusGPT** | Conversational AI assistant with code generation | LangChain + Ollama |
| 🎙️ **Podcast Intelligence** | Transcribe audio & answer questions from spoken content | Whisper + RAG + Pinecone |
| 🐱 **Cat vs Dog Classifier** | Transfer learning-based binary image classification | CNN (ImageNet) |

---

## 🖼️ Screenshots

### Image Captioning — BLIP Vision Model
> Upload any image and get an instant AI-generated description.

![Image Captioning](screenshot_captioning.jpg)

---

### Image Generation — Stable Diffusion XL
> Type a prompt, select a model, and generate photorealistic images in seconds.

![Image Generation](screenshot_image_gen.jpg)

---

### AI News Generator — Agentic AI + TTS + AI Anchor
> Enter any topic. Get AI-written news articles, a voice narration, and a video AI anchor broadcast — all generated automatically.

![AI News](screenshot_news.jpg)

---

### NexusGPT — Conversational AI
> LangChain-powered local LLM assistant. Handles code generation, explanations, and multi-turn conversations.

![NexusGPT](screenshot_nexusgpt.jpg)

---

### Podcast Intelligence — Whisper + RAG
> Upload any audio file, get a full transcription, then ask questions grounded in the spoken content via RAG.

![Podcast Intelligence](screenshot_podcast.jpg)

---

## 🏗️ Architecture

### Activity Diagram
> Complete request lifecycle — from user input through FastAPI routing, AI model invocation, optional tool use (web search / Pinecone), optional media output (TTS / video), and final response delivery. All error paths surface gracefully to the user.

![Activity Diagram](activity_diagram.jpg)

**Three input paths:**
- **Image Upload** → routed to the Image module (captioning or classification)
- **Text Prompt** → routed to Text Generation, Chat, or News module
- **Audio Upload** → routed to Podcast Q&A (Whisper + RAG)

---

### Class Diagram
> Service-oriented architecture with clear separation between the FastAPI app, AI service classes, agent tools, and data stores.

![Class Diagram](class_diagram.jpg)

**Key classes:**

| Class | Role |
|---|---|
| `FastAPIApp` | Routes all requests, validates inputs |
| `ImageService` | BLIP image captioning + CNN classification |
| `DiffusionService` | Stable Diffusion XL image generation |
| `NewsService` | Generates news content and coordinates media output |
| `ChatService` | LangChain-based conversational chat agent |
| `PodcastService` | Whisper transcription + Pinecone RAG + Q&A |
| `Agent` | Base agent class with goal, tools[], and run() loop |
| `SearchTool` | DuckDuckGo web search integration |
| `MediaTool` | Text-to-Speech and video synthesis |
| `VectorDB` | Pinecone vector store for semantic retrieval |
| `LLM` | Shared LLM backend for generation tasks |

---

## ⚙️ System Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                          AI NEXUS FRONTEND                            │
│   Image Captioning │ Image Gen │ AI News │ NexusGPT │ Podcast Q&A   │
└────────────────────────────┬─────────────────────────────────────────┘
                             │  HTTP / Multipart
┌────────────────────────────▼─────────────────────────────────────────┐
│                      FastAPI Backend                                  │
│          routeRequest()  │  validateInput()  │  async handlers        │
└──────┬──────────┬────────────┬──────────┬──────────┬─────────────────┘
       │          │            │          │          │
  ┌────▼───┐ ┌───▼─────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼───────────┐
  │ Image  │ │Diffusion│ │  News  │ │  Chat  │ │   Podcast     │
  │Service │ │Service  │ │Service │ │Service │ │   Service     │
  └────┬───┘ └───┬─────┘ └───┬────┘ └───┬────┘ └───┬───────────┘
       │         │           │          │           │
      BLIP     SD-XL     LLM Agent   Ollama    Whisper + Pinecone
     / CNN               + TTS        LLM        RAG Pipeline
               + Video Synthesis
```

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA GPU (recommended for Stable Diffusion inference)
- [Ollama](https://ollama.com) installed locally
- Pinecone API key

### Clone & Setup

```bash
# Clone the repository
git clone https://github.com/Mehfuz-alam/AI_NEXUS_MID.git
cd AI_NEXUS_MID

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

```bash
cp .env.example .env
```

```env
# Pinecone Vector DB
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name

# Ollama (local LLM)
OLLAMA_BASE_URL=http://localhost:11434

# App Security
SECRET_KEY=your_secret_key
DEBUG=False
```

### Pull a Local LLM & Start

```bash
# Pull a model for NexusGPT
ollama pull llama3

# Start the FastAPI backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger API docs.

---

## 📚 Module Breakdown

<details>
<summary><b>🖼️ Image Captioning — BLIP</b></summary>
<br/>

Upload any image and BLIP generates a natural language description automatically. Supports JPEG, PNG, and WEBP.

**Example output:** `"a kitten with blue eyes laying on a green blanket"`

```
POST /api/vision/caption
Content-Type: multipart/form-data
Body: { image: <file> }
```

</details>

<details>
<summary><b>🎨 Image Generation — Stable Diffusion XL</b></summary>
<br/>

Generate photorealistic images from text prompts. Supports multiple models including `stabilityai/stable-diffusion-xl-base-1.0`.

**Example prompt:** `"white Tiger in 8k"`

```
POST /api/generate/image
Content-Type: application/json
Body: { "prompt": "...", "model": "stabilityai/stable-diffusion-xl-base-1.0", "steps": 30 }
```

</details>

<details>
<summary><b>📰 AI News Generator — Agent + TTS + AI Anchor</b></summary>
<br/>

Enter any topic. The agentic AI searches the web using DuckDuckGo, synthesizes a grounded news summary, generates a voice narration via TTS, and produces a video AI anchor broadcast.

**Example query:** `"Latest news of Nepal today in 3 lines"`

**Output:** ✍️ Text summary + 🔊 AI Voice audio + 🎥 AI Anchor video

```
POST /api/news/generate
Content-Type: application/json
Body: { "query": "Latest news on AI today" }
```

</details>

<details>
<summary><b>🤖 NexusGPT — LangChain + Ollama</b></summary>
<br/>

A fully local LLM-powered assistant with multi-turn conversation, code generation, explanations, and general Q&A. Built on LangChain with Ollama for local model inference — no cloud API needed.

**Example:** `"Write a python program to find sum of two numbers"` → Returns complete, documented code.

```
POST /api/chat
Content-Type: application/json
Body: { "message": "...", "session_id": "abc123" }
```

</details>

<details>
<summary><b>🎙️ Podcast Intelligence — Whisper + Pinecone RAG</b></summary>
<br/>

Upload any audio file (MP3, WAV, M4A). Whisper transcribes the spoken content in full. The transcript is embedded and stored in Pinecone. Ask any question and receive a grounded, context-aware answer via RAG.

**Example:**
- Upload: `Groq's AI Chip Breaks Speed Records.mp3`
- Ask: `"World Government Summit held in?"`
- Answer: `"The World Government Summit is held in Dubai."`

```
POST /api/podcast/transcribe    # Upload & transcribe audio
POST /api/podcast/ask           # Ask questions from content
```

</details>

<details>
<summary><b>🐱 Cat vs Dog Classifier — CNN Transfer Learning</b></summary>
<br/>

Binary image classifier using a CNN fine-tuned with transfer learning on ImageNet pretrained weights. Demonstrates domain adaptation from a general vision backbone to a custom classification task.

```
POST /api/vision/classify
Content-Type: multipart/form-data
Body: { image: <file> }
```

</details>

---

## 🔬 Applied AI Research

| Research Area | Where It's Applied in AI NEXUS |
|---|---|
| **Transformer Architectures** | BLIP (vision-language), Whisper (speech), LLM chat |
| **Diffusion Models** | Stable Diffusion XL inference + custom DDPM training pipeline |
| **Transfer Learning** | ImageNet pretrained CNN → Cat/Dog binary classifier |
| **Vector Search** | Dense semantic embeddings stored and retrieved via Pinecone |
| **Retrieval-Augmented Generation** | Podcast Q&A — semantic retrieval + LLM generation over transcripts |
| **Agentic Tool Use** | News agent with DuckDuckGo web search integration |
| **Text-to-Speech Synthesis** | Voice narration pipeline for AI news broadcasting |
| **Video Synthesis** | AI anchor video generation in the news module |

---

## 📁 Project Structure

```
AI_NEXUS_MID/
│
├── 📂 api/                        # FastAPI route handlers
│   ├── vision.py                  # Captioning & classification
│   ├── generative.py              # Stable Diffusion endpoints
│   ├── news.py                    # AI News + TTS + video
│   ├── chat.py                    # NexusGPT endpoints
│   └── podcast.py                 # Whisper + RAG endpoints
│
├── 📂 core/                       # AI service implementations
│   ├── blip_captioner.py          # BLIP image captioning
│   ├── cnn_classifier.py          # CNN transfer learning
│   ├── stable_diffusion.py        # Text-to-image generation
│   ├── ddpm_trainer.py            # Custom DDPM training
│   ├── nexusgpt.py                # LangChain + Ollama chat
│   ├── whisper_stt.py             # Whisper transcription
│   └── rag_pipeline.py            # Pinecone RAG pipeline
│
├── 📂 agents/                     # Agentic AI
│   ├── news_agent.py              # Web search + media agent
│   └── search_tool.py             # DuckDuckGo tool
│
├── 📂 media/                      # TTS + Video synthesis
│   ├── tts_engine.py
│   └── video_generator.py
│
├── 📂 auth/                       # Authentication & security
├── 📂 frontend/                   # Web frontend (HTML/CSS/JS)
├── 📂 database/                   # DB models & migrations
│
├── main.py                        # FastAPI application entry
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! This project was built as a deep learning platform and collaboration is encouraged.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add: description of feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👤 Author

**Mehfuz Alam**

> *"Building AI systems taught me that real-world AI is not just about models — it's about integrating multiple intelligent systems into one coherent architecture."*

[![GitHub](https://img.shields.io/badge/GitHub-Mehfuz--alam-181717?style=for-the-badge&logo=github)](https://github.com/Mehfuz-alam)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with 🧠 curiosity, ☕ coffee, and a relentless drive to go deeper into AI.**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" width="100%"/>

</div>
