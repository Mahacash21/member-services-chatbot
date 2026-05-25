import streamlit as st
import time
from member_chatbot import get_bot_response, vectorstore, llm

st.set_page_config(
    page_title="Member Services Assistant",
    page_icon="🏥",
    layout="centered"
)

# ── Simple session auth ────────────────────────────────
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🏥 Member Services Assistant")
        st.markdown("Please enter your member ID to continue.")
        member_id = st.text_input("Member ID", type="password")
        if st.button("Login", type="primary"):
            if member_id and len(member_id) >= 4:
                st.session_state.member_id    = member_id
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Please enter a valid member ID")
        return False
    return True

if not check_password():
    st.stop()

# ── Page header ────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    st.title("🏥 Member Services Assistant")
    st.markdown("Ask me anything about your healthcare benefits.")
with col2:
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

st.divider()

# ── Service status ─────────────────────────────────────
if vectorstore is None or llm is None:
    st.error("Service is temporarily unavailable. Please try again later.")
    st.stop()

# ── Initialize session state ───────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0

# ── Display metrics ────────────────────────────────────
col3, col4 = st.columns(2)
with col3:
    st.metric("Questions Asked", st.session_state.total_questions)
with col4:
    member_id = st.session_state.get("member_id", "anonymous")
    st.metric("Session", f"Member {member_id[:4]}***")

st.divider()

# ── Display previous messages ──────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("cached"):
            st.caption("⚡ Cached response")
        if message.get("response_time"):
            st.caption(f"⏱ {message['response_time']:.2f}s")

# ── Handle new question ────────────────────────────────
if question := st.chat_input("Ask about your benefits..."):

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({
        "role":    "user",
        "content": question
    })

    with st.chat_message("assistant"):
        with st.spinner("Searching your benefits..."):
            result = get_bot_response(
                question=question,
                history=st.session_state.messages,
                vectorstore=vectorstore,
                llm=llm,
                user_id=st.session_state.get("member_id", "anonymous")
            )

        answer = result["answer"]
        st.markdown(answer)

        if result.get("cached"):
            st.caption("⚡ Cached response")
        if result.get("response_time"):
            st.caption(f"⏱ {result['response_time']:.2f}s")

        if result.get("error"):
            st.error("There was an issue processing your request.")

    st.session_state.messages.append({
        "role":          "assistant",
        "content":       answer,
        "cached":        result.get("cached", False),
        "response_time": result.get("response_time")
    })

    st.session_state.total_questions += 1