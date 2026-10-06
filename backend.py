import os
import re
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

# Extract API Key
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
os.environ["GEMINI_API_KEY"] = api_key
os.environ["GOOGLE_API_KEY"] = api_key


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


def fetch_transcript(video_id: str):
    """Fetch transcript supporting multi-language captions."""
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id, languages=['en', 'hi', 'es', 'fr', 'de'])
        text = " ".join(chunk.text for chunk in fetched)
        return text, "auto"
    except Exception as e:
        raise ValueError(f"Could not fetch transcript: {str(e)}")


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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 3. Prompt & LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2, google_api_key=api_key)
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