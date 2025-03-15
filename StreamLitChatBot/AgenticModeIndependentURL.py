import pandas as pd
import os
import re
import time
import uuid
from dotenv import load_dotenv
from pathlib import Path
from typing import AsyncGenerator, Sequence
from PIL import Image
import asyncio

from autogen_agentchat.agents import BaseChatAgent, UserProxyAgent, AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage,TextMessage
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import SystemMessage, UserMessage
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination

load_dotenv()

class CloudServeAgent(BaseChatAgent):
    def __init__(
        self,
        name: str,
        description: str = "An agent that generates a DataFrame and a matplotlib plot based on user inputs.",
        work_dir: Path = Path("cloudserve")
    ):
        super().__init__(name=name, description=description)
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        self._model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        self._system_message = [SystemMessage(content="""Write Only Python code what user asks for, don't write anything else than code. Save outputs in current working directory. Always use PNG images to save images.""")]
        self.code='print(''Hello, World!'')'
        self.language='python'
        
    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        content = ''
        self._model_context = UnboundedChatCompletionContext()  # Clear context on each new request
        request_uuid = str(uuid.uuid4())  # Generate a UUID for this request
        
        for msg in messages:
            await self._model_context.add_message(UserMessage(content=msg.content, source=msg.source))
        
        conversation_history = self._system_message[:]
        for msg in messages:
            conversation_history.append(UserMessage(content=msg.content, source="user"))
        
        response = await self._model_client.create(conversation_history, cancellation_token=cancellation_token)
        initial_response_content = response.content
        
        pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
        matches = pattern.findall(initial_response_content)
        
        if matches:
            self.language, self.code = matches[-1]  # Take last code block
            if not self.language:
                self.language = "python"
        else:
            self.language = "python"
            self.code = initial_response_content.strip()
        
        while True:
            try:
                code_block = CodeBlock(language=self.language, code=self.code)
                result = await LocalCommandLineCodeExecutor(work_dir=self.work_dir/request_uuid).execute_code_blocks(
                    code_blocks=[code_block],
                    cancellation_token=cancellation_token,
                )
                df_output = {
                    "exit_code": result.exit_code if hasattr(result, 'exit_code') else "N/A",
                    "error_message": result.stderr[:50] if hasattr(result, 'stderr') else "No error message",
                    "stdout": result.stdout if hasattr(result, 'stdout') else str(result)
                }
                break
            except Exception as e:
                error_details = str(e)
                conversation_history.append(UserMessage(content=f"Error: {error_details}", source="user"))
                fix_response = await self._model_client.create(conversation_history, cancellation_token=cancellation_token)
                fixed_response_content = fix_response.content
                
                matches = pattern.findall(fixed_response_content)
                if matches:
                    self.language, self.code = matches[-1]
                    if not self.language:
                        self.language = "python"
                else:
                    self.language = "python"
                    self.code = fixed_response_content.strip()
        image_urls = []
        other_files = []
        code_file=""
        uuid_dir = self.work_dir / request_uuid
        uuid_dir.mkdir(exist_ok=True)
        
        for file in uuid_dir.iterdir():
            if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                image_urls.append(str(file))
            elif file.suffix.lower() in [".py"]:
                code_file = file
            else:
                other_files.append(file)
                
        result_output = df_output if df_output else "No output"
        
        content = f'''{{"uuid": "{request_uuid}",
            "image_urls": {image_urls},
            "code": "{code_file}",
            "result": "{df_output}",
            "other_files": "{other_files}"}}'''
        
        return Response(chat_message=TextMessage(content=content, source=self.name), inner_messages=[])
    
    async def on_messages_stream(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> AsyncGenerator[Response, None]:
        response = await self.on_messages(messages, cancellation_token)
        yield response
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

def create_team() -> SelectorGroupChat:
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    cloudServeAgent = CloudServeAgent("CloudServeAgent")
    termination = TextMentionTermination("APPROVE")
    user = UserProxyAgent("UserAgent", input_func=input)

    selector_prompt = """Select an agent to perform task.
    {roles}
    Current conversation context:
    {history}
    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
    """
    
    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        Decide wether to just engage with User or Delegate task to CloudServeAgent
        You are a planning agent. Your responsibility is to give tasks to CloudServeAgent and evaluate if the task is completed.
        After all tasks are complete, summarize the findings and end with "APPROVE".
        """,
    )

    team = SelectorGroupChat(
        [planning_agent, cloudServeAgent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        max_turns=2,
        allow_repeated_speaker=False,
    )
    return team
