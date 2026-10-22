#!/usr/bin/env python3
"""Module 17: Database Seed Script.

Initializes the PostgreSQL database test environment and seeds sample customers,
support tickets, knowledge-base documents with vector embeddings, long-term memories,
and episodic records.

Safe to rerun without duplicating or corrupting existing records.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

# Ensure local src directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from research_and_support_agent.data.seed import seed_test_environment


async def main() -> None:
    load_dotenv()

    default_db = os.getenv(
        "DATABASE_CONNECTION_CONVERSATION_URL",
        "postgresql://postgres:postgres@127.0.0.1:5432/ai_search",
    )

    parser = argparse.ArgumentParser(
        description="Seed the Module 17 PostgreSQL database with idempotent sample data."
    )
    parser.add_argument(
        "--db-url",
        default=default_db,
        help=f"PostgreSQL connection URL (default: {default_db})",
    )
    args = parser.parse_args()

    print("\n================================================================================")
    print("MODULE 17: SEEDING POSTGRESQL TEST ENVIRONMENT")
    print("================================================================================")
    print(f"Target Database: {args.db_url}")
    print("Initializing schemas, pgvector extension, and seeding sample records...\n")

    try:
        stats = await seed_test_environment(connection_string=args.db_url)
    except Exception as exc:  # noqa: BLE001
        print(f"Error during seeding: {exc}", file=sys.stderr)
        sys.exit(1)

    print("--------------------------------------------------------------------------------")
    print("SEEDING SUMMARY:")
    print(f"  • Customers Seeded (support_customers):          {stats['customers_seeded']}")
    print(f"  • Support Tickets Seeded (support_tickets):       {stats['tickets_seeded']}")
    print(f"  • Knowledge-Base Chunks Indexed (document_chunks): {stats['kb_documents_indexed']}")
    print(
        f"  • Long-Term Memories Seeded (memory_lt_records):   {stats['longterm_memories_seeded']}"
    )
    print(
        f"  • Episodic Episodes Seeded (memory_ep_episodes):  {stats['episodic_episodes_seeded']}"
    )
    print("================================================================================")
    print("✓ PostgreSQL test environment successfully initialized and seeded!\n")


if __name__ == "__main__":
    asyncio.run(main())
