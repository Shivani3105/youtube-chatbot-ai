import os
import re
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        os.environ["GEMINI_API_KEY"] = env_key


def extract_video_id(url_or_id: str) -> str:
    """Extract YouTube 11-character video ID from various URL formats or raw ID."""
    url_or_id = url_or_id.strip()
    if len(url_or_id) == 11 and not ("/" in url_or_id or "." in url_or_id):
        return url_or_id
    
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/shorts\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
            
    raise ValueError(f"Invalid YouTube URL or Video ID: '{url_or_id}'")


def format_docs(retrieved_docs):
    """Format retrieved document chunks into context string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def fetch_transcript(video_id: str):
    """Fetch transcript for a given video ID supporting multi-language fallbacks."""
    ytt_api = YouTubeTranscriptApi()
    fetched = None
    lang_code = "auto"
    
    # Strategy 1: Try api.list() or list_transcripts()
    try:
        transcript_list = None
        if hasattr(ytt_api, 'list'):
            transcript_list = ytt_api.list(video_id)
        elif hasattr(ytt_api, 'list_transcripts'):
            transcript_list = ytt_api.list_transcripts(video_id)
        elif hasattr(YouTubeTranscriptApi, 'list_transcripts'):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        if transcript_list:
            try:
                transcript = transcript_list.find_transcript(['en', 'hi', 'es', 'fr', 'de', 'pt', 'ja', 'ko', 'zh', 'ar', 'ru'])
            except Exception:
                transcript = next(iter(transcript_list))
            
            fetched = transcript.fetch()
            lang_code = getattr(transcript, 'language_code', 'auto')
    except Exception:
        pass

    # Strategy 2: Direct fetch with language fallback
    if not fetched:
        pref_langs = ['en', 'hi', 'es', 'fr', 'de', 'pt', 'ja', 'ko', 'zh', 'ar', 'ru']
        try:
            if hasattr(ytt_api, 'fetch'):
                fetched = ytt_api.fetch(video_id, languages=pref_langs)
            elif hasattr(YouTubeTranscriptApi, 'get_transcript'):
                fetched = YouTubeTranscriptApi.get_transcript(video_id, languages=pref_langs)
        except Exception:
            if hasattr(ytt_api, 'fetch'):
                fetched = ytt_api.fetch(video_id)
            elif hasattr(YouTubeTranscriptApi, 'get_transcript'):
                fetched = YouTubeTranscriptApi.get_transcript(video_id)

    if not fetched:
        raise ValueError("Could not retrieve transcript for this video.")

    text_chunks = []
    for chunk in fetched:
        if hasattr(chunk, 'text'):
            text_chunks.append(chunk.text)
        elif isinstance(chunk, dict) and 'text' in chunk:
            text_chunks.append(chunk['text'])
        else:
            text_chunks.append(str(chunk))

    transcript_text = " ".join(text_chunks)
    if not transcript_text.strip():
        raise ValueError("Transcript is empty for this video.")

    return transcript_text, lang_code


def build_rag_chain(video_url_or_id: str):
    """
    Core RAG Pipeline:
    1. Extract Video ID
    2. Fetch Transcript
    3. Split into chunks using RecursiveCharacterTextSplitter
    4. Build FAISS Vector Store using GoogleGenerativeAIEmbeddings
    5. Construct LangChain LCEL RAG Chain with ChatGoogleGenerativeAI (gemini-3.6-flash)
    """
    video_id = extract_video_id(video_url_or_id)
    transcript_text, lang_code = fetch_transcript(video_id)

    # 1. Split transcript text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])

    # 2. Embeddings & Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 3. LLM & Prompt Template
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
    prompt = PromptTemplate(
        template="""You are a helpful YouTube Video assistant.

Answer only from the provided video context.
If the context is insufficient or not found in the transcript, state clearly that you don't know based on the transcript.

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    # 4. LCEL Parallel Chain & Parser
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    return {
        "video_id": video_id,
        "main_chain": main_chain,
        "chunk_count": len(chunks),
        "language": lang_code,
        "transcript_snippet": transcript_text[:300] + "..."
    }


def query_rag_chain(main_chain, question: str) -> str:
    """Invoke the RAG chain with a user question."""
    if not main_chain:
        raise ValueError("RAG Chain is not initialized.")
    return main_chain.invoke(question)