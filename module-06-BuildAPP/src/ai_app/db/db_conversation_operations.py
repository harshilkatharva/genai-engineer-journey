import uuid

import psycopg

from ai_app.core.conversation_manager import Message


class DBOperator:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def create_conversation(self, user_id: str):
        conversation_id = uuid.uuid4()

        with psycopg.connect(self.connection_string) as conn:
            conn.execute(
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

    def add_history(
        self,
        conversation_id,
        user_id,
        request_id,
        role: str,
        content: str,
        feature: str,
        llm_model: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float,
        duration_ms: float,
    ):
        with psycopg.connect(self.connection_string) as conn:
            conn.execute(
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

    def get_history(self, conversation_id):
        with psycopg.connect(self.connection_string) as conn:
            rows = conn.execute(
                """
                SELECT role,content,input_tokens, output_tokens
                FROM history
                WHERE conversation_id = %s
                ORDER BY created at ASC, id ASC
                """(
                    conversation_id,
                ),
            ).fetchall()

        return [
            Message(
                role=role,
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            for role, content, input_tokens, output_tokens in rows
        ]
