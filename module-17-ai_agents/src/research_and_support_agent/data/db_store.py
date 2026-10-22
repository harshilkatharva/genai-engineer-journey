"""Database storage helpers for customer accounts and support tickets in PostgreSQL."""

from __future__ import annotations

import json

import psycopg

from ..models import CustomerAccount, SupportTicket, TicketStatus


async def init_support_tables(connection_string: str) -> None:
    """Ensure support_customers and support_tickets tables exist in PostgreSQL."""
    async with await psycopg.AsyncConnection.connect(connection_string) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS support_customers (
                    customer_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    tier TEXT NOT NULL DEFAULT 'standard',
                    balance_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    active_tickets JSONB NOT NULL DEFAULT '[]'::jsonb,
                    risk_flag BOOLEAN NOT NULL DEFAULT false,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );

                CREATE TABLE IF NOT EXISTS support_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    assigned_team TEXT NOT NULL DEFAULT 'tier1_support',
                    escalation_reason TEXT,
                    priority TEXT NOT NULL DEFAULT 'medium',
                    history JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                );
                """
            )
        await conn.commit()


async def load_customer_accounts(connection_string: str) -> dict[str, CustomerAccount]:
    """Load all customer account records from PostgreSQL into domain models."""
    await init_support_tables(connection_string)
    accounts: dict[str, CustomerAccount] = {}

    async with (
        await psycopg.AsyncConnection.connect(connection_string) as conn,
        conn.cursor() as cur,
    ):
        await cur.execute(
            """
                SELECT customer_id, name, email, tier, balance_usd, active_tickets, risk_flag, metadata
                FROM support_customers
                """
        )
        rows = await cur.fetchall()
        for r in rows:
            active_t = r[5] if isinstance(r[5], list) else (json.loads(r[5]) if r[5] else [])
            meta = r[7] if isinstance(r[7], dict) else (json.loads(r[7]) if r[7] else {})
            acc = CustomerAccount(
                customer_id=r[0],
                name=r[1],
                email=r[2],
                tier=r[3],
                balance_usd=float(r[4]),
                active_tickets=active_t,
                risk_flag=bool(r[6]),
                metadata=meta,
            )
            accounts[acc.customer_id] = acc

    return accounts


async def load_support_tickets(connection_string: str) -> dict[str, SupportTicket]:
    """Load all support tickets from PostgreSQL into domain models."""
    await init_support_tables(connection_string)
    tickets: dict[str, SupportTicket] = {}

    async with (
        await psycopg.AsyncConnection.connect(connection_string) as conn,
        conn.cursor() as cur,
    ):
        await cur.execute(
            """
                SELECT ticket_id, customer_id, title, description, status, assigned_team,
                       escalation_reason, priority, history, created_at, updated_at
                FROM support_tickets
                """
        )
        rows = await cur.fetchall()
        for r in rows:
            hist = r[8] if isinstance(r[8], list) else (json.loads(r[8]) if r[8] else [])
            t = SupportTicket(
                ticket_id=r[0],
                customer_id=r[1],
                title=r[2],
                description=r[3],
                status=TicketStatus(r[4]),
                assigned_team=r[5],
                escalation_reason=r[6],
                priority=r[7],
                history=hist,
                created_at=float(r[9]),
                updated_at=float(r[10]),
            )
            tickets[t.ticket_id] = t

    return tickets


async def upsert_customer_account(connection_string: str, account: CustomerAccount) -> None:
    """Insert or update a customer account in PostgreSQL."""
    await init_support_tables(connection_string)
    async with await psycopg.AsyncConnection.connect(connection_string) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO support_customers
                (customer_id, name, email, tier, balance_usd, active_tickets, risk_flag, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (customer_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    email = EXCLUDED.email,
                    tier = EXCLUDED.tier,
                    balance_usd = EXCLUDED.balance_usd,
                    active_tickets = EXCLUDED.active_tickets,
                    risk_flag = EXCLUDED.risk_flag,
                    metadata = EXCLUDED.metadata;
                """,
                (
                    account.customer_id,
                    account.name,
                    account.email,
                    account.tier,
                    account.balance_usd,
                    json.dumps(account.active_tickets),
                    account.risk_flag,
                    json.dumps(account.metadata),
                ),
            )
        await conn.commit()


async def upsert_support_ticket(connection_string: str, ticket: SupportTicket) -> None:
    """Insert or update a support ticket in PostgreSQL."""
    await init_support_tables(connection_string)
    async with await psycopg.AsyncConnection.connect(connection_string) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO support_tickets
                (ticket_id, customer_id, title, description, status, assigned_team,
                 escalation_reason, priority, history, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticket_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    status = EXCLUDED.status,
                    assigned_team = EXCLUDED.assigned_team,
                    escalation_reason = EXCLUDED.escalation_reason,
                    priority = EXCLUDED.priority,
                    history = EXCLUDED.history,
                    updated_at = EXCLUDED.updated_at;
                """,
                (
                    ticket.ticket_id,
                    ticket.customer_id,
                    ticket.title,
                    ticket.description,
                    ticket.status.value,
                    ticket.assigned_team,
                    ticket.escalation_reason,
                    ticket.priority,
                    json.dumps(ticket.history),
                    ticket.created_at,
                    ticket.updated_at,
                ),
            )
        await conn.commit()
