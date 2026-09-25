from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg
from psycopg.rows import dict_row

from customer_support_agent.core import get_settings
from customer_support_agent.models import (
    CancellationPolicy,
    Order,
    Product,
    SupportDocument,
)


class CustomerSupportDB:
    """Async read/write boundary for the existing e-commerce PostgreSQL data."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or get_settings().database_url

    async def _fetch_all(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        async with (
            await psycopg.AsyncConnection.connect(
                self.database_url, row_factory=dict_row
            ) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(query, params)
            return await cursor.fetchall()

    async def get_order(self, order_id: str) -> Order | None:
        rows = await self._fetch_all(
            """
            SELECT order_id, customer_id, status, ordered_at, total_amount, items
            FROM orders
            WHERE order_id = %s
            """,
            (order_id,),
        )
        return Order.model_validate(rows[0]) if rows else None

    async def search_products(
        self, query: str, max_price: float | None = None, limit: int = 10
    ) -> list[Product]:
        filters = ["(name ILIKE %s OR description ILIKE %s)"]
        params: list[Any] = [f"%{query}%", f"%{query}%"]
        if max_price is not None:
            filters.append("price <= %s")
            params.append(max_price)
        params.append(limit)
        rows = await self._fetch_all(
            f"""
            SELECT product_id, name, description, price, available, metadata
            FROM products
            WHERE {" AND ".join(filters)}
            ORDER BY available DESC, name
            LIMIT %s
            """,
            params,
        )
        return [Product.model_validate(row) for row in rows]

    async def get_product(self, product_id: str) -> Product | None:
        rows = await self._fetch_all(
            """
            SELECT product_id, name, description, price, available, metadata
            FROM products
            WHERE product_id = %s
            """,
            (product_id,),
        )
        return Product.model_validate(rows[0]) if rows else None

    async def search_support_documents(
        self, query: str, document_type: str | None = None, limit: int = 5
    ) -> list[SupportDocument]:
        filters = ["chunk_text ILIKE %s"]
        params: list[Any] = [f"%{query}%"]
        if document_type is not None:
            filters.append("document_type = %s")
            params.append(document_type)
        params.append(limit)
        rows = await self._fetch_all(
            f"""
            SELECT chunk_id, chunk_text AS content, document_type, metadata
            FROM document_chunks
            WHERE {" AND ".join(filters)}
            ORDER BY chunk_id
            LIMIT %s
            """,
            params,
        )
        return [SupportDocument.model_validate(row) for row in rows]

    async def get_cancellation_policy(self) -> CancellationPolicy | None:
        documents = await self.search_support_documents(
            "cancellation", document_type="cancellation_policy", limit=3
        )
        if not documents:
            return None
        document = documents[0]
        return CancellationPolicy(
            policy_name=document.document_type,
            content=document.content,
            metadata=document.metadata,
        )

    async def create_escalation(
        self, customer_id: str | None, reason: str, summary: str, priority: str = "normal"
    ) -> str:
        async with await psycopg.AsyncConnection.connect(self.database_url) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO support_escalations
                        (customer_id, reason, summary, priority, status)
                    VALUES (%s, %s, %s, %s, 'open')
                    RETURNING ticket_id
                    """,
                    (customer_id, reason, summary, priority),
                )
                row = await cursor.fetchone()
            await connection.commit()
        if row is None:
            raise RuntimeError("Database did not return an escalation ticket ID")
        return str(row[0])
