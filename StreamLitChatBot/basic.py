import streamlit as st
from autogen_core import CancellationToken
from AgenticModeIndependentURL import create_team
from streamlit_console import StreamlitConsole
import asyncio

if "messages" not in st.session_state:
    st.session_state.messages = []

AgenticTeam = create_team()

st.title("🤖 Streamlit Chatbot")
user_input = st.chat_input("Say something...")

with st.sidebar:
    st.title("Chat Options")
    st.write("Customize your chat experience here.")

chat_container = st.container()

with chat_container:
    st.divider()
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        avatar = "🧑‍💻" if role == "user" else "🤖"
        
        if role == "user":
            st.markdown(f'<div class="user-message">{content} {avatar}</div>', unsafe_allow_html=True)
        else:
            content_data=msg["content"]
            st.markdown(f'<div class="bot-message">{avatar}</div>', unsafe_allow_html=True)
            if isinstance(content_data, dict):
                        if "uuid" in content_data:
                            st.subheader(f"Directory: {content_data['uuid']}")

                        if "code" in content_data:
                            st.write("### Code File:")
                            try:
                                with open(content_data["code"], "r") as code_file:
                                    code_content = code_file.read()
                                st.code(code_content, language="python")
                            except Exception as e:
                                st.error(f"Could not load code file: {e}")

                        if "image_urls" in content_data and isinstance(content_data["image_urls"], list):
                            st.write("### Images:")
                            for url in content_data["image_urls"]:
                                try:
                                    image_name = url.split("/")[-1]
                                    st.write(f"**Image Name:** {image_name}")
                                    st.image(url)
                                except Exception as e:
                                    st.error(f"Could not load image: {e}")
                        if "result" in content_data:
                            if content_data["result"]["exit_code"] == 0:
                                st.success("Execution succeeded")
                            else:
                                st.error(f"Execution failed with exit code: {content_data['result']['exit_code']}")

            elif isinstance(content_data, list):
                st.write("### List Content:")
                for idx, item in enumerate(content_data):
                    st.write(f"{idx + 1}. {item}")

            else:
                st.markdown(f'<div class="bot-message">{content}</div>', unsafe_allow_html=True)

st.markdown("""
    <style>
    .user-message {
        text-align: right;
        background-color: less-black;
        border-radius: 10px;
        margin: 5px;
    }
    .bot-message {
        text-align: left;
        background-color: less-black;
        border-radius: 10px;
        margin: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

if user_input:
    response = asyncio.run(StreamlitConsole(AgenticTeam.run_stream(task=user_input)))

# st.rerun()
