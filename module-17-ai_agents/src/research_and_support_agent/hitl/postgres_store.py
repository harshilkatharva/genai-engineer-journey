"""PostgreSQL-backed durable approval store implementing genai-core ApprovalStore protocol."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from genai_core.observability import ObservabilityEvent, get_logger, safe_log
from genai_core.tools.guards import ApprovalStore
from genai_core.tools.models import ApprovalDecision, ApprovalRequest, RiskLevel

_logger = get_logger("research_and_support_agent.hitl.postgres_store")


class PostgresApprovalStore(ApprovalStore):
    """PostgreSQL implementation of ApprovalStore with fallback to memory for tests."""

    def __init__(
        self,
        connection_string: str | None = None,
        connection_pool: Any = None,
        table_prefix: str = "hitl_",
    ) -> None:
        self.connection_string = connection_string
        self.connection_pool = connection_pool
        self.requests_table = f"{table_prefix}approval_requests"
        self.decisions_table = f"{table_prefix}approval_decisions"
        # In-memory storage cache / fallback for zero-dependency test runs
        self._mem_requests: dict[str, ApprovalRequest] = {}
        self._mem_decisions: dict[str, ApprovalDecision] = {}
        self._db_initialized: bool = False

    @asynccontextmanager
    async def _connection_context(self) -> AsyncIterator[Any]:
        """Obtain an async database connection context if available."""
        if self.connection_pool is not None:
            async with self.connection_pool.connection() as conn:
                yield conn
        elif self.connection_string:
            import psycopg

            async with await psycopg.AsyncConnection.connect(self.connection_string) as conn:
                yield conn
        else:
            yield None

    async def initialize(self) -> None:
        """Create PostgreSQL approval tables if database is configured and accessible."""
        if not (self.connection_pool or self.connection_string):
            return

        try:
            async with self._connection_context() as conn:
                if conn is None:
                    return
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self.requests_table} (
                            request_id TEXT PRIMARY KEY,
                            call_id TEXT NOT NULL,
                            tool_name TEXT NOT NULL,
                            arguments JSONB NOT NULL,
                            risk_level TEXT NOT NULL,
                            reason TEXT NOT NULL,
                            created_at DOUBLE PRECISION NOT NULL
                        );
                        CREATE TABLE IF NOT EXISTS {self.decisions_table} (
                            request_id TEXT PRIMARY KEY REFERENCES {self.requests_table}(request_id) ON DELETE CASCADE,
                            approved BOOLEAN NOT NULL,
                            reason TEXT,
                            modified_arguments JSONB,
                            decided_at DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW())
                        );
                        """
                    )
                if hasattr(conn, "commit"):
                    await conn.commit()
            self._db_initialized = True
        except Exception as exc:  # noqa: BLE001
            safe_log(
                _logger,
                logging.WARNING,
                f"PostgreSQL approval store table initialization failed: {exc}. Using memory cache.",
                event=ObservabilityEvent.TOOL_EXECUTION_FAILED,
            )
            self._db_initialized = False

    async def save_request(self, request: ApprovalRequest) -> None:
        """Persist a new pending approval request."""
        self._mem_requests[request.request_id] = request

        if not self._db_initialized and (self.connection_string or self.connection_pool):
            await self.initialize()

        if self._db_initialized:
            try:
                async with self._connection_context() as conn:
                    if conn is not None:
                        async with conn.cursor() as cur:
                            await cur.execute(
                                f"""
                                INSERT INTO {self.requests_table}
                                (request_id, call_id, tool_name, arguments, risk_level, reason, created_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (request_id) DO UPDATE SET
                                    call_id = EXCLUDED.call_id,
                                    arguments = EXCLUDED.arguments,
                                    risk_level = EXCLUDED.risk_level,
                                    reason = EXCLUDED.reason;
                                """,
                                (
                                    request.request_id,
                                    request.call_id,
                                    request.tool_name,
                                    json.dumps(request.arguments),
                                    request.risk_level.value,
                                    request.reason,
                                    request.created_at,
                                ),
                            )
                        if hasattr(conn, "commit"):
                            await conn.commit()
            except Exception as exc:  # noqa: BLE001
                safe_log(
                    _logger,
                    logging.WARNING,
                    f"Failed to persist approval request to Postgres: {exc}",
                    event=ObservabilityEvent.TOOL_EXECUTION_FAILED,
                )

    async def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Retrieve an approval request by ID."""
        if request_id in self._mem_requests:
            return self._mem_requests[request_id]

        if self._db_initialized:
            try:
                async with self._connection_context() as conn:
                    if conn is not None:
                        async with conn.cursor() as cur:
                            await cur.execute(
                                f"""
                                SELECT request_id, call_id, tool_name, arguments, risk_level, reason, created_at
                                FROM {self.requests_table}
                                WHERE request_id = %s
                                """,
                                (request_id,),
                            )
                            row = await cur.fetchone()
                            if row:
                                args = row[3] if isinstance(row[3], dict) else json.loads(row[3])
                                req = ApprovalRequest(
                                    request_id=row[0],
                                    call_id=row[1],
                                    tool_name=row[2],
                                    arguments=args,
                                    risk_level=RiskLevel(row[4]),
                                    reason=row[5],
                                    created_at=row[6],
                                )
                                self._mem_requests[request_id] = req
                                return req
            except Exception as exc:  # noqa: BLE001
                safe_log(
                    _logger,
                    logging.WARNING,
                    f"Failed to query approval request from Postgres: {exc}",
                    event=ObservabilityEvent.TOOL_EXECUTION_FAILED,
                )

        return None

    async def record_decision(self, request_id: str, decision: ApprovalDecision) -> None:
        """Record an approval or rejection decision."""
        self._mem_decisions[request_id] = decision

        if self._db_initialized:
            try:
                async with self._connection_context() as conn:
                    if conn is not None:
                        async with conn.cursor() as cur:
                            mod_args = (
                                json.dumps(decision.modified_arguments)
                                if decision.modified_arguments is not None
                                else None
                            )
                            await cur.execute(
                                f"""
                                INSERT INTO {self.decisions_table}
                                (request_id, approved, reason, modified_arguments)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (request_id) DO UPDATE SET
                                    approved = EXCLUDED.approved,
                                    reason = EXCLUDED.reason,
                                    modified_arguments = EXCLUDED.modified_arguments;
                                """,
                                (request_id, decision.approved, decision.reason, mod_args),
                            )
                        if hasattr(conn, "commit"):
                            await conn.commit()
            except Exception as exc:  # noqa: BLE001
                safe_log(
                    _logger,
                    logging.WARNING,
                    f"Failed to record approval decision to Postgres: {exc}",
                    event=ObservabilityEvent.TOOL_EXECUTION_FAILED,
                )

    async def get_decision(self, request_id: str) -> ApprovalDecision | None:
        """Retrieve recorded decision for an approval request."""
        if request_id in self._mem_decisions:
            return self._mem_decisions[request_id]

        if self._db_initialized:
            try:
                async with self._connection_context() as conn:
                    if conn is not None:
                        async with conn.cursor() as cur:
                            await cur.execute(
                                f"""
                                SELECT approved, reason, modified_arguments
                                FROM {self.decisions_table}
                                WHERE request_id = %s
                                """,
                                (request_id,),
                            )
                            row = await cur.fetchone()
                            if row:
                                mod_args = (
                                    row[2]
                                    if (row[2] is None or isinstance(row[2], dict))
                                    else json.loads(row[2])
                                )
                                dec = ApprovalDecision(
                                    approved=row[0],
                                    reason=row[1],
                                    modified_arguments=mod_args,
                                )
                                self._mem_decisions[request_id] = dec
                                return dec
            except Exception as exc:  # noqa: BLE001
                safe_log(
                    _logger,
                    logging.WARNING,
                    f"Failed to query approval decision from Postgres: {exc}",
                    event=ObservabilityEvent.TOOL_EXECUTION_FAILED,
                )

        return None

    async def list_pending_requests(self) -> list[ApprovalRequest]:
        """List all requests awaiting decision."""
        if self._db_initialized:
            try:
                async with self._connection_context() as conn:
                    if conn is not None:
                        async with conn.cursor() as cur:
                            await cur.execute(
                                f"""
                                SELECT r.request_id, r.call_id, r.tool_name, r.arguments, r.risk_level, r.reason, r.created_at
                                FROM {self.requests_table} r
                                LEFT JOIN {self.decisions_table} d ON r.request_id = d.request_id
                                WHERE d.request_id IS NULL
                                ORDER BY r.created_at ASC
                                """
                            )
                            rows = await cur.fetchall()
                            pending: list[ApprovalRequest] = []
                            for row in rows:
                                args = row[3] if isinstance(row[3], dict) else json.loads(row[3])
                                req = ApprovalRequest(
                                    request_id=row[0],
                                    call_id=row[1],
                                    tool_name=row[2],
                                    arguments=args,
                                    risk_level=RiskLevel(row[4]),
                                    reason=row[5],
                                    created_at=row[6],
                                )
                                self._mem_requests[req.request_id] = req
                                pending.append(req)
                            return pending
            except Exception as exc:  # noqa: BLE001
                safe_log(
                    _logger,
                    logging.WARNING,
                    f"Failed to list pending requests from Postgres: {exc}",
                    event=ObservabilityEvent.TOOL_EXECUTION_FAILED,
                )

        return [
            req for req_id, req in self._mem_requests.items() if req_id not in self._mem_decisions
        ]
