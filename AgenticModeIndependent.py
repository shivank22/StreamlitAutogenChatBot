import pandas as pd
import os
import re
from dotenv import load_dotenv
from pathlib import Path
from typing import AsyncGenerator, Sequence
from PIL import Image
import asyncio

from autogen_agentchat.agents import BaseChatAgent, UserProxyAgent, AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, MultiModalMessage
from autogen_core import CancellationToken, Image as AGImage
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
        self._model_context = UnboundedChatCompletionContext()
        self.local_executor = LocalCommandLineCodeExecutor(work_dir=self.work_dir)
        self._model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        self._system_message = [SystemMessage(content="""Write Only Python code what user asks for dont write anything else than code, Save outputs in current working directory, Always use png images to save images""")]

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (MultiModalMessage,)

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        content = []
        
        for file in self.work_dir.iterdir():
            if file.is_file():
                os.remove(file)
        
        for msg in messages:
            await self._model_context.add_message(UserMessage(content=msg.content, source=msg.source))

        history = [
            (msg.source if hasattr(msg, "source") else "system")
            + ": "
            + (msg.content if isinstance(msg.content, str) else "")
            + "\n"
            for msg in await self._model_context.get_messages()
        ]
        
        print('this is History', history)
        
        error_details = None
        conversation_history = self._system_message[:]
        
        if messages:
            print(messages)
            conversation_history.append(UserMessage(content=messages[0].content, source="user"))
        response = await self._model_client.create(conversation_history, cancellation_token=cancellation_token)
        initial_response_content = response.content
        
        pattern = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)
        match = pattern.search(initial_response_content)
        if match:
            language = match.group(1)
            code = match.group(2)
        else:
            language = "python"
            code = initial_response_content

        while True:
            try:
                code_block = CodeBlock(language=language, code=code)
                result = await self.local_executor.execute_code_blocks(
                    code_blocks=[code_block],
                    cancellation_token=cancellation_token,
                )
                df_output = result.stdout if hasattr(result, 'stdout') else str(result)
                print(code_block.code)
                break
            except Exception as e:
                error_details = str(e)
                conversation_history.append(UserMessage(content=f"Error: {error_details}", source="user"))
                fix_response = await self._model_client.create(conversation_history, cancellation_token=cancellation_token)
                fixed_response_content = fix_response.content
                match = pattern.search(fixed_response_content)
                if match:
                    language = match.group(1)
                    code = match.group(2)
                else:
                    language = "python"
                    code = fixed_response_content

        print("Final Output:", df_output)
        content.append(df_output)
        
        for file in self.work_dir.iterdir():
            if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                GenImage = Image.open(file)
                content.append(AGImage(GenImage))
            
        multimodal_msg = MultiModalMessage(content=content, source=self.name)
        
        return Response(chat_message=multimodal_msg, inner_messages=[])
    
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
        You are a planning agent your responsibility is to give task to CloudServeAgent and evaluate if task is completed.
        After all tasks are complete, summarize the findings and end with "APPROVE".
        """,
    )

    team = SelectorGroupChat(
        [planning_agent, cloudServeAgent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        max_turns=2,
        allow_repeated_speaker=False,  # Allow an agent to speak multiple turns in a row.
    )
    return team

#Plot a graph for country and their population generated data yourself