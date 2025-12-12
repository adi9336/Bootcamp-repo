# streamlit_app.py
import streamlit as st
import uuid
from typing import List, Tuple
import time
import traceback

# Import your agent functions from agent.py (must be in same repo)
# agent.py must provide create_agent() and chat(user_input, agent_executor)
from agent import create_agent, chat as agent_chat

# ========== Streamlit UI & Config ==========
st.set_page_config(page_title="Agentic AI Chat", layout="wide", page_icon="🤖")
st.title("🤖 Agentic AI Chat (Integrated)")

# Sidebar
st.sidebar.header("Settings")
st.sidebar.markdown(
    "This app runs your LangChain agent directly inside Streamlit. "
    "Make sure OPENAI_API_KEY and TAVILY_API_KEY are set in Streamlit Secrets."
)



# ========== Cached agent initializer ==========
# Use Streamlit cache_resource to initialize the agent once per app instance
@st.cache_resource(show_spinner=False)
def get_agent_executor():
    try:
        return create_agent()
    except Exception as e:
        # bubble up initialization error
        raise RuntimeError(f"Agent initialization failed: {e}")

# Lazy init with user-visible spinner
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False

if not st.session_state.agent_ready:
    with st.spinner("Initializing agent (may take 10-30s)..."):
        try:
            agent_executor = get_agent_executor()
            st.session_state.agent_executor = agent_executor
            st.session_state.agent_ready = True
            st.success("Agent ready ✅")
        except Exception as e:
            st.session_state.agent_ready = False
            st.error("Failed to initialize agent. Check logs and secrets.")
            st.exception(e)

# ========== Session state for chat ==========
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": "..."}

# Top-level UI
st.markdown("Chat with your LangChain Agent. Tools available: datetime, weather, tavily search.")
st.divider()

# Render existing messages
for m in st.session_state.messages:
    role = m.get("role", "user")
    content = m.get("content", "")
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)

# Input
prompt = st.chat_input("Type your message here...")

if prompt:
    # append and show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the agent (synchronously) and show typing indicator
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        try:
            # Ensure agent is initialized
            if not st.session_state.get("agent_ready") or "agent_executor" not in st.session_state:
                # Try to initialize again
                st.warning("Agent not initialized, attempting to initialize...")
                agent_executor = get_agent_executor()
                st.session_state.agent_executor = agent_executor
                st.session_state.agent_ready = True
            else:
                agent_executor = st.session_state.agent_executor

            # Call your agent's chat function (from agent.py)
            # chat(user_input: str, agent_executor) -> str
            start = time.time()
            response_text = agent_chat(prompt, agent_executor)
            elapsed = time.time() - start

            # Update history and UI
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            placeholder.markdown(response_text)
            # optional small footer
            st.caption(f"Response time: {elapsed:.2f}s")

        except Exception as e:
            # show useful debug info (but avoid leaking secrets)
            placeholder.markdown("Error: the agent failed to respond. See details in sidebar logs.")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
            # Print traceback in sidebar for debugging
            with st.sidebar.expander("Error details (traceback)"):
                st.text(traceback.format_exc())

# Sidebar controls


# Instructions
st.sidebar.markdown(
    """HI
    """
)
