import os
import re
import requests
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()

# Extract API Keys securely from environment variables
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
youtube_api_key = os.getenv("YOUTUBE_API_KEY") or gemini_key

if not gemini_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in .env file or server environment.")

os.environ["GEMINI_API_KEY"] = gemini_key
os.environ["GOOGLE_API_KEY"] = gemini_key


def extract_video_id(url_or_id: str) -> str:
    """Extract 11-character video ID from YouTube URL or ID."""
    url_or_id = url_or_id.strip()
    if len(url_or_id) == 11 and not ("/" in url_or_id or "." in url_or_id):
        return url_or_id
    match = re.search(r"(?:v=|\/|shorts\/)([0-9A-Za-z_-]{11})", url_or_id)
    if match:
        return match.group(1)
    raise ValueError(f"Invalid YouTube URL: {url_or_id}")


def format_docs(docs):
    """Format retrieved document chunks into context string."""
    return "\n\n".join(d.page_content for d in docs)


def fetch_official_youtube_api(video_id: str, api_key: str):
    """
    Official Google YouTube Data API v3 fallback.
    Fetches official video metadata, title, and detailed description.
    """
    try:
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if "items" in data and len(data["items"]) > 0:
                snippet = data["items"][0]["snippet"]
                title = snippet.get("title", "")
                desc = snippet.get("description", "")
                text = f"Video Title: {title}\n\nVideo Description:\n{desc}"
                if text.strip():
                    return text, "official_api"
    except Exception:
        pass
    return None, None


def fetch_transcript(video_id: str):
    """Fetch transcript with multi-language fallback & official YouTube Data API fallback."""
    # 1. Try standard YouTubeTranscriptApi
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id, languages=['en', 'hi', 'es', 'fr', 'de'])
        text = " ".join(chunk.text for chunk in fetched)
        if text.strip():
            return text, "transcript"
    except Exception:
        pass

    # 2. Try Official Google YouTube Data API v3 Fallback
    official_text, official_lang = fetch_official_youtube_api(video_id, youtube_api_key)
    if official_text:
        return official_text, official_lang

    raise ValueError("Could not retrieve transcript or metadata for this video.")


def build_rag_chain(video_url_or_id: str):
    """
    RAG Pipeline:
    1. Extract Video ID & Fetch Transcript
    2. Split text into chunks using RecursiveCharacterTextSplitter
    3. Build FAISS Vector Store using GoogleGenerativeAIEmbeddings (models/gemini-embedding-2)
    4. Construct LCEL Chain using ChatGoogleGenerativeAI (gemini-3.6-flash)
    """
    video_id = extract_video_id(video_url_or_id)
    transcript_text, lang = fetch_transcript(video_id)

    # 1. Text Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])

    # 2. Vector Store & Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=gemini_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 3. Prompt & LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2, google_api_key=gemini_key)
    prompt = PromptTemplate(
        template="""You are a helpful YouTube assistant. Answer only from the provided context.

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    # 4. LangChain LCEL RAG Chain
    chain = (
        RunnableParallel({'context': retriever | RunnableLambda(format_docs), 'question': RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )

    return {
        "video_id": video_id,
        "main_chain": chain,
        "chunk_count": len(chunks),
        "language": lang,
        "transcript_snippet": transcript_text[:300] + "..."
    }


def query_rag_chain(main_chain, question: str) -> str:
    """Invoke the RAG chain with a user question."""
    if not main_chain:
        raise ValueError("RAG Chain is not initialized.")
    return main_chain.invoke(question)