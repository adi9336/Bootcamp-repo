import streamlit as st
import requests
import uuid
from typing import List, Dict, Any, Optional
import json

# Set page config
st.set_page_config(
    page_title="AI Agent Chat",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        padding: 12px;
        font-size: 16px;
    }
    .stButton>button {
        width: 100%;
        padding: 10px;
        font-weight: bold;
    }
    .message {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    </style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "http://localhost:8000"

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
st.title("🤖 AI Agent Chat")
st.caption("Chat with the AI agent powered by LangChain and FastAPI")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show typing indicator
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Call the FastAPI endpoint
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "message": prompt,
                    "session_id": st.session_state.session_id
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_response = data["response"]
                
                # Update chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
                # Display assistant response
                message_placeholder.markdown(assistant_response)
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                
        except Exception as e:
            error_msg = f"Connection error: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# Add a clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())  # Generate a new session ID
    st.experimental_rerun()

# Add API status indicator
st.sidebar.markdown("### API Status")
try:
    response = requests.get(f"{API_URL}", timeout=2)
    if response.status_code == 200:
        st.sidebar.success("API is running")
    else:
        st.sidebar.error(f"API error: {response.status_code}")
except:
    st.sidebar.error("API is not running")

# Add instructions
st.sidebar.markdown("### How to use")
st.sidebar.markdown("""
1. Type your message in the chat input
2. Press Enter or click Send
3. The AI agent will respond using its tools
4. Use the sidebar to clear chat history
""")
