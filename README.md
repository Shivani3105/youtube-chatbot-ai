# 📺 TubeTalk AI — Interactive YouTube Transcript Chatbot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)](https://www.langchain.com/)
[![Gemini](https://img.shields.io/badge/Model-Gemini_3.6_Flash-orange.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**TubeTalk AI** is a production-grade, full-stack **Retrieval-Augmented Generation (RAG)** web application that enables users to interactively chat with any YouTube video transcript in real-time.

Powered by **FastAPI**, **LangChain (LCEL)**, **FAISS Vector Database**, and **Google Gemini 3.6 Flash**, the application automatically fetches YouTube transcripts, splits text into vector chunks, and generates context-grounded answers to user queries with zero hallucination.

---

## 🌟 Key Features

- 🎥 **Dynamic YouTube URL & ID Parsing**: Supports standard watch links (`youtube.com/watch?v=...`), short links (`youtu.be/...`), YouTube Shorts, and raw video IDs.
- 🌐 **Multi-Language & Subtitle Fallbacks**: Automatically fetches English, Hindi, Spanish, French, German, or auto-generated captions.
- ⚡ **High-Speed Vector Search**: Uses `GoogleGenerativeAIEmbeddings` and Meta's **FAISS** vector store to index text chunks in milliseconds.
- 🛡️ **Grounded RAG Responses**: Answers are strictly grounded in video transcript context using custom prompt templates.
- 🎨 **Modern Glassmorphism UI**: Beautiful, responsive dark-mode interface built with HTML5, CSS3 (Google Fonts `Outfit` & `Inter`), and vanilla JavaScript.
- 💬 **Interactive Quick Action Prompts**: One-click chips for instant video summaries, key takeaways, and core topics.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Browser (Client UI)                  │
│               [index.html + style.css + app.js]             │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Async Fetch API / REST)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       server.py (FastAPI)                   │
│         • Exposes REST API Endpoints (/api/load-video)      │
│         • Serves Static HTML/CSS/JS Frontend Assets         │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Imports & Delegates)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       backend.py (AI Engine)                │
│   1. YouTubeTranscriptApi ➔ Downloads Video Captions        │
│   2. RecursiveCharacterTextSplitter ➔ Chunks text           │
│   3. GoogleGenerativeAIEmbeddings + FAISS ➔ Vector Index    │
│   4. ChatGoogleGenerativeAI (gemini-3.6-flash) ➔ RAG Chain   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
youtubeChatbot/
├── .env                  # Environment secrets (GEMINI_API_KEY)
├── .gitignore            # Git exclusion rules
├── app.js                # Frontend client JavaScript logic
├── backend.py            # Core AI & RAG Pipeline module
├── index.html            # User interface HTML layout
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies for deployment
├── server.py             # FastAPI Web Server & API routes
└── style.css             # Glassmorphism dark-mode styling
```

---

## 🛠️ Tech Stack & Libraries

- **Backend Framework:** FastAPI, Uvicorn, Pydantic
- **LLM & Embeddings:** Google Gemini 3.6 Flash (`ChatGoogleGenerativeAI`), `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2`)
- **Orchestration & Vector Store:** LangChain (LCEL), FAISS (`faiss-cpu`), `youtube-transcript-api`
- **Frontend:** Vanilla HTML5, CSS3 (CSS Variables, Flexbox/Grid), JavaScript (ES6+ Fetch API)
- **Deployment:** Render Cloud Web Service, GitHub

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/Shivani3105/youtube-chatbot-ai.git
cd youtube-chatbot-ai
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the Server
```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to `http://127.0.0.1:8000`.

---

## 🔌 API Endpoints Documentation

| Method | Endpoint | Description | Request Body Example |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/load-video` | Parses URL, fetches transcript, & builds FAISS index | `{"video_url": "https://youtu.be/..."}` |
| `POST` | `/api/query` | Executes similarity search & generates AI answer | `{"question": "Summarize this video"}` |
| `GET` | `/api/status` | Returns backend server health & video status | N/A |

---

## 🌐 Production Deployment

The project is configured for one-click deployment on **Render**:

1. Push code to GitHub repository.
2. Create a **New Web Service** on Render.
3. Set **Build Command:** `pip install -r requirements.txt`
4. Set **Start Command:** `uvicorn server:app --host 0.0.0.0 --port 10000`
5. Add `GEMINI_API_KEY` under **Environment Variables**.

---

## 📝 License

This project is licensed under the MIT License.
