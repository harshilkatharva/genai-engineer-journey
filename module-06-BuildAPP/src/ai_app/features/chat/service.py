from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks
from jinja2 import Template

from ai_app.models.message import Message

from ai_app.core.AIConfig import AiConfig
from ai_app.core.conversation_manager import ConversationManager
from ai_app.core.cost_tracker import CostTracker
from ai_app.models.llm_response_model import LLMResponseModel
from ai_app.services.llm_service import LLMClient


class ChatService:
    def __init__(self):
        self.llm_client = LLMClient()
        self.ai_config = AiConfig()
        self.conversation_manager = ConversationManager()
        self.cost_tracker = CostTracker()

    async def get_answer(
        self,
        conversation_id: UUID,
        user_id: str,
        request_id: UUID,
        user_message: str,
        background_tasks: BackgroundTasks,
    ) -> LLMResponseModel:
        conversations = await self._get_conversation(conversation_id=conversation_id)
        prompt = self._build_prompt(conversations=conversations, user_message=user_message)
        response = await self.llm_client.complete(
            provider=self.ai_config.default_provider, prompt=prompt
        )

        background_tasks.add_task(
            self._add_conversations,
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
            user_message=user_message,
            response=response,
        )

        return response

    async def _get_conversation(self, conversation_id: UUID):
        return await self.conversation_manager.get_conversations(conversation_id=conversation_id)

    def _build_prompt(self, conversations: list[Message], user_message: str):
        prompt_template = Template(Path("src/ai_app/features/chat/prompts/prompt.md").read_text())
        history = [{"role": con["role"], "content": con["content"]} for con in conversations]

        return prompt_template.render(conversation_history=history, user_message=user_message)

    async def _add_conversations(
        self,
        conversation_id: UUID,
        user_id: str,
        request_id: UUID,
        user_message: str,
        response: LLMResponseModel,
    ) -> None:
        # Save user message
        input_cost = self.cost_tracker.get_cost(
            input_token=response.input_tokens, output_token=0, model=response.model
        )
        output_cost = self.cost_tracker.get_cost(
            input_token=0, output_token=response.output_tokens, model=response.model
        )
        await self.conversation_manager.add_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
            role="user",
            content=user_message,
            feature="chat",
            llm_model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=0,
            estimated_cost=input_cost,
            duration_ms=response.latency_ms,
        )

        # Save assistant response
        await self.conversation_manager.add_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
            role="assistant",
            content=response.text,
            feature="chat",
            llm_model=response.model,
            input_tokens=0,
            output_tokens=response.output_tokens,
            estimated_cost=output_cost,
            duration_ms=response.latency_ms,
        )
