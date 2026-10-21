import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

MEMORY_FILE = Path(__file__).parent / "chat_memory.json"

st.set_page_config(
    page_title="Gemma AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background:
        radial-gradient(
            circle at 75% 10%,
            rgba(91, 50, 190, 0.12),
            transparent 35%
        ),
        #080c19;

    color: #f1f3ff;
}

/* Remove excessive top whitespace */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* Main headings */
h1, h2, h3 {
    color: #f5f3ff !important;
    letter-spacing: -0.5px;
}

/* General text visibility */
.stApp p,
.stApp label,
.stApp .stMarkdown {
    color: #e0e5f5;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0c1122;
    border-right: 1px solid #202943;
}

section[data-testid="stSidebar"] * {
    color: #e7eaff;
}

/* Brand */
.brand {
    font-size: 27px;
    font-weight: 800;
    padding: 10px 0 25px 0;
    letter-spacing: -1px;
}

.brand span {
    color: #9b7bff;
}

/* Online status */
.status {
    color: #a8b2cc;
    font-size: 14px;
    padding-bottom: 22px;
}

.status span {
    color: #54e6a5;
}

/* Welcome section */
.welcome {
    text-align: center;
    padding: 65px 10px 35px 10px;
}

.orb {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    width: 78px;
    height: 78px;
    border-radius: 50%;
    background: linear-gradient(135deg, #a78bfa, #6039e8);
    box-shadow: 0 0 55px rgba(124, 78, 255, 0.35);
    font-size: 40px;
    margin-bottom: 22px;
}

.welcome h1 {
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 800;
    margin-bottom: 10px;
}

.gradient-text {
    background: linear-gradient(90deg, #a78bfa, #7c5cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.welcome p {
    color: #aab6d1 !important;
    font-size: 17px;
}

/* Suggestion cards */
div.stButton > button {
    background: #111a30;
    color: #f1f3ff !important;
    border: 1px solid #293657;
    border-radius: 14px;
    padding: 15px 14px;
    min-height: 66px;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: #1b2441;
    border-color: #9271ff;
    color: #ffffff !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #10182b;
    border: 1px solid #202c48;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 14px;
}

/* Make chat message text readable */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] div {
    color: #edf0ff;
    line-height: 1.75;
}

/* Code blocks */
[data-testid="stChatMessage"] pre {
    background: #080d19 !important;
    border: 1px solid #293657;
    border-radius: 10px;
}

[data-testid="stChatMessage"] code {
    color: #d7c8ff;
}

/* Message input container */
[data-testid="stChatInput"] {
    background: #11182b;
    border: 1px solid #51418c;
    border-radius: 18px;
    padding: 5px;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #9b7bff;
    box-shadow: 0 0 0 2px rgba(155, 123, 255, 0.13);
}

/* IMPORTANT: visible placeholder text */
[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #a78bfa !important;
    background: transparent !important;
    font-size: 16px !important;
}/* Chat input box */
[data-testid="stChatInput"] {
    background-color: #11182b !important;
    border: 1px solid #8b5cf6 !important;
    border-radius: 18px !important;
    padding: 8px !important;
}

/* Text typed by the user */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
    background-color: #11182b !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #a78bfa !important;
    font-size: 16px !important;
}

/* Placeholder: "Message Gemma... Ask anything" */
[data-testid="stChatInput"] textarea::placeholder {
    color: #c4b5fd !important;
    -webkit-text-fill-color: #c4b5fd !important;
    opacity: 1 !important;
}

/* Prevent a light background inside the input */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {
    background-color: #11182b !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #aeb9d3 !important;
    -webkit-text-fill-color: #aeb9d3 !important;
    opacity: 1 !important;
}

/* Sidebar divider */
hr {
    border-color: #26304b;
}

/* Footer */
.footer {
    text-align: center;
    color: #8491af;
    font-size: 12px;
    padding: 12px 0 0 0;
}

/* Small screens */
@media (max-width: 640px) {
    .block-container {
        padding: 1rem 0.8rem;
    }

    .welcome {
        padding-top: 35px;
    }

    .welcome h1 {
        font-size: 30px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. MEMORY FUNCTIONS
# ==========================================

def load_memory():
    if not MEMORY_FILE.exists():
        return []

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return [
                item for item in data
                if isinstance(item, dict)
                and item.get("role") in ("user", "assistant")
                and isinstance(item.get("content"), str)
            ]

    except (json.JSONDecodeError, OSError):
        pass

    return []


def save_memory(messages):
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            messages,
            file,
            ensure_ascii=False,
            indent=2
        )


# ==========================================
# 4. MODEL
# ==========================================

@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(
        model="gemma-4-26b-a4b-it"
    )


# ==========================================
# 5. SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = load_memory()

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ==========================================
# 6. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        '<div class="brand">✦ Gemma <span>AI</span></div>',
        unsafe_allow_html=True
    )

    if st.button("＋  New chat", use_container_width=True):
        st.session_state.messages = []
        save_memory([])
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("---")

    st.caption("YOUR WORKSPACE")

    st.markdown("✨  Gemma Assistant")
    st.markdown("🧠  Conversation Memory")
    st.markdown("⚡  Powered by LangChain")

    st.markdown("---")

    st.caption("RECENT MESSAGES")

    user_messages = [
        message for message in st.session_state.messages
        if message["role"] == "user"
    ]

    for message in reversed(user_messages[-5:]):
        label = message["content"].replace("\n", " ")

        if len(label) > 32:
            label = label[:32] + "..."

        st.caption("💬  " + label)

    st.markdown("---")

    st.caption("MODEL")
    st.markdown("🟢  Gemma 4")

    st.markdown(
        """
        <div style="
            position: relative;
            margin-top: 35px;
            padding: 12px;
            border: 1px solid #293657;
            border-radius: 12px;
            background: #11182b;
        ">
            <strong>✨ Gemma AI</strong><br>
            <span style="color:#9ba8c5;font-size:12px;">
                Your personal AI workspace
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 7. MAIN HEADER
# ==========================================

st.markdown(
    '<div class="status">🟢 &nbsp; Gemma 4 '
    '&nbsp; · &nbsp; <span>Online</span></div>',
    unsafe_allow_html=True
)


# ==========================================
# 8. WELCOME SCREEN
# ==========================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome">
            <div class="orb">✦</div>
            <h1>Your <span class="gradient-text">AI workspace.</span></h1>
            <p>
                Ask questions, explore ideas, and build something great.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "📚  Learn something new",
            use_container_width=True
        ):
            st.session_state.pending_prompt = (
                "Teach me an interesting concept about LLMs."
            )
            st.rerun()

        if st.button(
            "✨  Explore LLMs",
            use_container_width=True
        ):
            st.session_state.pending_prompt = (
                "Explain how LLMs work and how I can build "
                "projects using them."
            )
            st.rerun()

    with col2:
        if st.button(
            "💻  Write or debug code",
            use_container_width=True
        ):
            st.session_state.pending_prompt = (
                "Help me write and debug Python code."
            )
            st.rerun()

        if st.button(
            "💡  Brainstorm ideas",
            use_container_width=True
        ):
            st.session_state.pending_prompt = (
                "Give me some interesting AI project ideas."
            )
            st.rerun()


# ==========================================
# 9. DISPLAY CONVERSATION
# ==========================================

for message in st.session_state.messages:

    role = message["role"]

    with st.chat_message(
        role,
        avatar="assistant" if role == "assistant" else "user"
    ):
        st.markdown(message["content"])


# ==========================================
# 10. CHAT INPUT
# ==========================================

typed_prompt = st.chat_input(
    "Message Gemma... Ask anything"
)

user_input = typed_prompt

if user_input is None and st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


# ==========================================
# 11. GENERATE RESPONSE
# ==========================================

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    save_memory(st.session_state.messages)

    with st.chat_message("user", avatar="user"):
        st.markdown(user_input)

    # Keep the recent conversation as context.
    recent_messages = st.session_state.messages[-20:]

    conversation = "\n".join(
        f"{'User' if message['role'] == 'user' else 'Gemma'}: "
        f"{message['content']}"
        for message in recent_messages
    )

    prompt = f"""
You are Gemma, a helpful AI assistant.

Use the conversation history to understand the user's
questions and previous messages.

Be friendly, accurate, and clear.
Do not invent information about the user.

CONVERSATION HISTORY:
{conversation}

Respond to the latest user message.
"""

    with st.chat_message("assistant", avatar="assistant"):

        with st.spinner("Gemma is thinking..."):

            try:
                model = get_model()
                response = model.invoke(prompt)

                answer = (
                    response.text
                    or "I couldn't generate a text response."
                )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                save_memory(st.session_state.messages)

            except Exception as error:

                st.error(
                    "Gemma couldn't respond. "
                    "Please check your API key, model access, "
                    "and internet connection."
                )

                # Avoid displaying potentially sensitive error details.
                st.caption(f"Error type: {type(error).__name__}")

    st.rerun()


# ==========================================
# 12. FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
        Gemma can make mistakes. Check important information.
    </div>
    """,
    unsafe_allow_html=True
)