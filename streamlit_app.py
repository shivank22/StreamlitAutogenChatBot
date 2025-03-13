import streamlit as st
import asyncio
from autogen_core import CancellationToken
from AgenticModeIndependent import create_team
from streamlit_console import StreamlitConsole  # Import your async Streamlit console

st.set_page_config(page_title="CloudServe Chatbot", layout="wide")

AgenticTeam = create_team()

st.title("🧠 CloudServe AI Chatbot")
st.write("Type a request to generate data and plots.")

# Add custom CSS for avatars and chat layout
st.markdown("""
    <style>
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
    }
    .chat-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        max-width: 70%;
    }
    .chat-container.user {
        justify-content: flex-end;
        margin-left: auto;
    }
    .chat-container.bot {
        justify-content: flex-start;
        margin-right: auto;
    }
    .chat-text {
        padding: 10px;
        border-radius: 10px;
        max-width: fit-content;
        word-wrap: break-word;
    }
    .user .chat-text {
        background-color: #333333;
        color: white;
    }
    .bot .chat-text {
        background-color: #444444;
        color: white;
    }
    .bot-avatar {
        margin-right: 10px;
    }
    .user-avatar {
        margin-left: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history using chat containers
for msg in st.session_state.messages:
    avatar_url = "user_avatar.png" if msg["role"] == "user" else "bot_avatar.png"
    avatar_class = "user-avatar" if msg["role"] == "user" else "bot-avatar"
    
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="chat-container {msg["role"]}">'
                    f'<img src="{avatar_url}" class="avatar {avatar_class}"> '
                    f'<div class="chat-text">{msg["content"]}</div>'
                    f'</div>', unsafe_allow_html=True)

# User input
user_input = st.chat_input("Ask something...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-container user">'
                    f'<div class="chat-text">{user_input}</div>'
                    f'<img src="user_avatar.png" class="avatar user-avatar"> '
                    f'</div>', unsafe_allow_html=True)
    
    # Generate bot response
    with st.chat_message("assistant"):
        cancellation_token = CancellationToken()
        response = asyncio.run(StreamlitConsole(AgenticTeam.run_stream(task=user_input)))
        print(response.__dict__)
        output = response
        if isinstance(output, list):
            for item in output:
                if isinstance(item, str):
                    st.session_state.messages.append({"role": "assistant", "content": item})
                    st.markdown(f'<div class="chat-container bot">'
                                f'<img src="bot_avatar.png" class="avatar bot-avatar"> '
                                f'<div class="chat-text">{item}</div>'
                                f'</div>', unsafe_allow_html=True)
                else:
                    st.image(item)
                    st.session_state.messages.append({"role": "assistant", "content": item})
        else:
            st.session_state.messages.append({"role": "assistant", "content": output})
            st.markdown(f'<div class="chat-container bot">'
                        f'<img src="bot_avatar.png" class="avatar bot-avatar"> '
                        f'<div class="chat-text">{output}</div>'
                        f'</div>', unsafe_allow_html=True)
