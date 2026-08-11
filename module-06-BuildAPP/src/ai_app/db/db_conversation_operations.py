import uuid
from uuid import UUID

import psycopg

from ai_app.models.message import Message


class DBOperator:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    async def create_conversation(self, user_id: str):
        conversation_id = uuid.uuid4()

        async with await psycopg.AsyncConnection.connect(self.connection_string) as conn:
            await conn.execute(
                """
                INSERT INTO conversations (conversation_id, user_id)
                VALUES (%s, %s)
                """,
                (
                    conversation_id,
                    user_id,
                ),
            )

        return conversation_id

    async def add_history(
        self,
        conversation_id: UUID,
        user_id,
        request_id: UUID,
        role: str,
        content: str,
        feature: str,
        llm_model: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float,
        duration_ms: float,
    ):
        async with await psycopg.AsyncConnection.connect(self.connection_string) as conn:
            await conn.execute(
                """
                INSERT INTO history 
                    (conversation_id,user_id,request_id, role, content, feature, llm_model, input_tokens, output_tokens, estimated_cost, duration_ms)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    conversation_id,
                    user_id,
                    request_id,
                    role,
                    content,
                    feature,
                    llm_model,
                    input_tokens,
                    output_tokens,
                    estimated_cost,
                    duration_ms,
                ),
            )

    async def get_history(self, conversation_id):
        async with (
            await psycopg.AsyncConnection.connect(self.connection_string) as conn,
            conn.cursor() as cursor,
        ):
            await cursor.execute(
                """
                    SELECT
                        role,
                        content,
                        input_tokens,
                        output_tokens
                    FROM history
                    WHERE conversation_id = %s
                    ORDER BY created_at ASC, id ASC
                    """,
                (conversation_id,),
            )

            rows = await cursor.fetchall()

        return [
            Message(
                role=role,
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            for role, content, input_tokens, output_tokens in rows
        ]
