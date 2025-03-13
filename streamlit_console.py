import asyncio
import streamlit as st
from typing import AsyncGenerator, List, Optional, TypeVar, Union, cast
from autogen_core import CancellationToken, Image
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

T = TypeVar("T", bound=TaskResult | Response)


async def StreamlitConsole(
    stream: AsyncGenerator[AgentEvent | ChatMessage | T, None],
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
    streaming_chunks: List[str] = []
    chat_container = st.container()

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
            with chat_container:
                st.write(message.content, end="")

        elif isinstance(message, UserInputRequestedEvent):
            if user_input_manager is not None:
                user_input_manager.notify_event_received(message.request_id)

        else:
            if streaming_chunks:
                streaming_chunks.clear()
                with chat_container:
                    st.write("")
            _display_message(message)

    if last_processed is None:
        raise ValueError("No TaskResult or Response was processed.")

    return last_processed


def _display_message(message: Union[AgentEvent, ChatMessage]) -> None:
    """
    Displays messages in Streamlit, including text and images.
    """
    if isinstance(message, MultiModalMessage):
        for content in message.content:
            if isinstance(content, str):
                st.write(content)
            elif isinstance(content, Image):
                st.write(f"Message content type: {type(content)}")
                st.image(content.image)
            else:
                    st.write('Unsupported image type')
    else:
        st.write(message.content)
