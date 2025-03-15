import asyncio
import streamlit as st
import json
import ast
from typing import AsyncGenerator, List, Optional, TypeVar, Union
from autogen_core import Image
from autogen_core.models import RequestUsage
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    UserInputRequestedEvent,
)

T = TypeVar("T", bound=Union[TaskResult, Response])

async def StreamlitConsole(
    stream: AsyncGenerator[Union[AgentEvent, ChatMessage, T], None],
    *,
    output_stats: bool = False,
    user_input_manager: Optional["UserInputManager"] = None,
) -> T:
    """
    Streamlit-based console to display chatbot messages, including text and images.
    """
    start_time = asyncio.get_event_loop().time()
    total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    last_processed: Optional[T] = None
    streaming_placeholder = st.empty()  # Placeholder for dynamic updates
    chat_container = st.container()
    streaming_chunks: List[str] = []

    async for message in stream:
        if isinstance(message, TaskResult):
            duration = asyncio.get_event_loop().time() - start_time
            if output_stats:
                with chat_container:
                    st.markdown(
                        f"### Summary\n"
                        f"- **Messages:** {len(message.messages)}\n"
                        f"- **Finish reason:** {message.stop_reason}\n"
                        f"- **Prompt tokens:** {total_usage.prompt_tokens}\n"
                        f"- **Completion tokens:** {total_usage.completion_tokens}\n"
                        f"- **Duration:** {duration:.2f} seconds"
                    )
            last_processed = message  # type: ignore

        elif isinstance(message, Response):
            duration = asyncio.get_event_loop().time() - start_time
            with chat_container:
                st.subheader(f"Response from {message.chat_message.source}")
                _display_message(message.chat_message)

            if output_stats and message.chat_message.models_usage:
                total_usage.completion_tokens += message.chat_message.models_usage.completion_tokens
                total_usage.prompt_tokens += message.chat_message.models_usage.prompt_tokens

            last_processed = message  # type: ignore

        elif isinstance(message, ModelClientStreamingChunkEvent):
            streaming_chunks.append(message.content)
            streaming_placeholder.write("".join(streaming_chunks))  # Update dynamically

        elif isinstance(message, UserInputRequestedEvent):
            if user_input_manager is not None:
                user_input_manager.notify_event_received(message.request_id)

        else:
            if streaming_chunks:
                streaming_chunks.clear()
                streaming_placeholder.write("")  # Clear after stream ends
            _display_message(message)

    if last_processed is None:
        raise ValueError("No TaskResult or Response was processed.")

    return last_processed

    
def parse_json_recursively(content):
    """
    Recursively parses a string into a valid JSON object.
    """
    if isinstance(content, dict):
        return {key: parse_json_recursively(value) for key, value in content.items()}
    elif isinstance(content, list):
        return [parse_json_recursively(item) for item in content]
    elif isinstance(content, str):
        try:
            return parse_json_recursively(ast.literal_eval(content))
        except (ValueError, SyntaxError):
            return content  # Return as is if parsing fails 
    else:
        return content

def _display_message(message: Union[AgentEvent, ChatMessage]) -> None:
    """
    Displays messages in Streamlit, handling both plain text and JSON content.
    """
    try:
        st.json(message)
        avatar = "🧑‍💻" if message.source == "user" else "🤖"
        avatarcls = "user" if message.source == "user" else "bot"
        content_str = str(message.content).strip()
        content_data = parse_json_recursively(content_str) if content_str else None
        bot_message = {"role": "user" if message.source == "user" else "bot", "content": content_data, "code": None, "images": []}
        
        if isinstance(content_data, dict):
            if "uuid" in content_data:
                st.subheader(f"Directory: {content_data['uuid']}")

            if "code" in content_data:
                st.write("### Code File:")
                try:
                    with open(content_data["code"], "r") as code_file:
                        code_content = code_file.read()
                    st.code(code_content, language="python")
                    bot_message["code"] = code_content
                except Exception as e:
                    st.error(f"Could not load code file: {e}")

            if "image_urls" in content_data and isinstance(content_data["image_urls"], list):
                st.write("### Images:")
                for url in content_data["image_urls"]:
                    try:
                        image_name = url.split("/")[-1]
                        st.write(f"**Image Name:** {image_name}")
                        st.image(url)
                        bot_message["images"].append(url)  # Store images in session state
                    except Exception as e:
                        st.error(f"Could not load image: {e}")

        elif isinstance(content_data, list):
            st.write("### List Content:")
            for idx, item in enumerate(content_data):
                st.write(f"{idx + 1}. {item}")

        else:
            if avatarcls == "user":
                st.markdown(f'<div class="{avatarcls}-message">{content_data}  {avatar}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="{avatarcls}-message">{avatar}  {content_data}</div>', unsafe_allow_html=True)
            
        st.session_state.messages.append(bot_message)

    except (json.JSONDecodeError, TypeError, SyntaxError, ValueError) as e:
        st.error(f"Error parsing JSON: {e}")
        st.session_state["metadata"]["errors"]["json_parsing"] = str(e)
        st.write(message.content)