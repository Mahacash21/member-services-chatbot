import logging
import hashlib
import time
from datetime import datetime
from functools import lru_cache
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import subprocess





load_dotenv()

# ══════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[        
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Auto-build vector DB if not present (needed for cloud deployment)
if not os.path.exists("./db"):
    logger.info("Vector database not found — building now...")
    try:
        from ingest import ingest_docs
        ingest_docs()
        logger.info("Vector database built successfully")
    except Exception as e:
        logger.critical(f"Failed to build vector database: {e}")
        raise

# ══════════════════════════════════════════════════════
# LOAD TOOLS
# ══════════════════════════════════════════════════════
def load_vectorstore():
    try:
        embeddings  = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            "./db",
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")
        return vectorstore
    except FileNotFoundError:
        logger.critical("Vector store not found — run ingest.py first")
        return None
    except Exception as e:
        logger.critical(f"Failed to load vector store: {e}")
        return None

def load_llm():
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        logger.info("LLM loaded successfully")
        return llm
    except Exception as e:
        logger.critical(f"Failed to load LLM: {e}")
        return None

vectorstore = load_vectorstore()
llm         = load_llm()

# ══════════════════════════════════════════════════════
# INPUT VALIDATION
# ══════════════════════════════════════════════════════
def validate_question(question: str) -> tuple:
    """
    Returns (is_valid, error_message)
    """
    if not question or not question.strip():
        return False, "Question cannot be empty"

    if len(question) > 500:
        return False, "Question too long — please keep it under 500 characters"

    if len(question.strip()) < 3:
        return False, "Question too short — please be more specific"

    # Basic PHI detection — flag potential sensitive info
    phi_patterns = ['ssn', 'social security', 'date of birth',
                    'dob', 'member id', 'account number']
    question_lower = question.lower()
    for pattern in phi_patterns:
        if pattern in question_lower:
            logger.warning(f"Potential PHI detected in question — flagged")
            return False, "Please do not include personal information like SSN or member ID in your question. Call member services directly for account-specific questions."

    return True, None

# ══════════════════════════════════════════════════════
# RATE LIMITING
# ══════════════════════════════════════════════════════
# Simple in-memory rate limiter
# Production would use Redis
user_request_times = {}

def check_rate_limit(user_id: str, max_requests: int = 10,
                     window_seconds: int = 60) -> bool:
    """
    Returns True if request is allowed
    Returns False if rate limit exceeded
    """
    now = time.time()

    if user_id not in user_request_times:
        user_request_times[user_id] = []

    # Remove requests outside the time window
    user_request_times[user_id] = [
        t for t in user_request_times[user_id]
        if now - t < window_seconds
    ]

    # Check if under limit
    if len(user_request_times[user_id]) >= max_requests:
        logger.warning(f"Rate limit exceeded for user: {user_id}")
        return False

    # Record this request
    user_request_times[user_id].append(now)
    return True

# ══════════════════════════════════════════════════════
# RESPONSE CACHING
# ══════════════════════════════════════════════════════
response_cache = {}

def get_cache_key(question: str) -> str:
    """Create a hash key from the question"""
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

def get_cached_response(question: str):
    """Returns cached response if available and fresh"""
    key = get_cache_key(question)
    if key in response_cache:
        cached_time, answer = response_cache[key]
        # Cache valid for 1 hour
        if time.time() - cached_time < 3600:
            logger.info(f"Cache hit for question: {question[:50]}")
            return answer
    return None

def cache_response(question: str, answer: str):
    """Save response to cache"""
    key = get_cache_key(question)
    response_cache[key] = (time.time(), answer)

# ══════════════════════════════════════════════════════
# AUDIT LOGGING
# ══════════════════════════════════════════════════════
audit_logger = logging.getLogger("audit")
audit_handler = logging.FileHandler("audit.log")
audit_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(message)s'
))
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)

def log_audit(user_id: str, question: str, answer: str,
              cached: bool, response_time: float):
    """
    HIPAA requirement — log every interaction
    Note: In production never log actual PHI
    Store de-identified data only
    """
    audit_logger.info(
        f"user={user_id} | "
        f"question_length={len(question)} | "
        f"answer_length={len(answer)} | "
        f"cached={cached} | "
        f"response_time={response_time:.2f}s | "
        f"question_hash={get_cache_key(question)}"
    )

# ══════════════════════════════════════════════════════
# CORE FUNCTION — PRODUCTION VERSION
# ══════════════════════════════════════════════════════
def get_bot_response(question: str, history: list,
                     vectorstore, llm,
                     user_id: str = "anonymous") -> dict:
    """
    Production version of get_bot_response
    Returns dict with answer and metadata
    """
    start_time = time.time()

    # Step 1 — Validate vectorstore and llm
    if vectorstore is None or llm is None:
        logger.error("Vectorstore or LLM not loaded")
        return {
            "answer": "Service temporarily unavailable. Please try again later.",
            "error": True,
            "cached": False
        }

    # Step 2 — Validate input
    is_valid, error_msg = validate_question(question)
    if not is_valid:
        logger.warning(f"Invalid question from {user_id}: {error_msg}")
        return {
            "answer": error_msg,
            "error": True,
            "cached": False
        }

    # Step 3 — Check rate limit
    if not check_rate_limit(user_id):
        return {
            "answer": "You have sent too many messages. Please wait a moment before asking again.",
            "error": True,
            "cached": False
        }

    # Step 4 — Check cache
    cached_answer = get_cached_response(question)
    if cached_answer:
        response_time = time.time() - start_time
        log_audit(user_id, question, cached_answer, True, response_time)
        return {
            "answer": cached_answer,
            "error": False,
            "cached": True
        }

    # Step 5 — Search policy documents
    try:
        docs    = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        logger.info(f"Found {len(docs)} relevant policy chunks")
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return {
            "answer": "I'm having trouble searching the policy documents. Please try again.",
            "error": True,
            "cached": False
        }

    # Step 6 — Build history text
    history_text = ""
    for msg in history[-6:]:
        role          = "Member" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    # Step 7 — Build prompt
    prompt = f"""You are a helpful and empathetic member services assistant
for a healthcare insurance plan.

IMPORTANT RULES:
- Only answer using the policy context provided below
- If you cannot find the answer say exactly:
  "I don't have that information in your plan documents.
   Please call member services at 1-800-XXX-XXXX
   or visit the member portal for assistance."
- Never make up coverage details or dollar amounts
- Never ask for or reference personal information
- Keep answers clear and concise
- Use plain language — avoid medical jargon

Previous conversation:
{history_text}

Policy context:
{context}

Member question: {question}

Answer:"""

    # Step 8 — Call LLM
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer   = response.content
        logger.info(f"Response generated successfully")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {
            "answer": "I'm having trouble generating a response. Please try again in a moment.",
            "error": True,
            "cached": False
        }

    # Step 9 — Cache the response
    cache_response(question, answer)

    # Step 10 — Audit log
    response_time = time.time() - start_time
    log_audit(user_id, question, answer, False, response_time)
    logger.info(f"Request completed in {response_time:.2f}s")

    return {
        "answer":  answer,
        "error":   False,
        "cached":  False,
        "response_time": response_time
    }