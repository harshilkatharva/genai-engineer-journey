import pytest

from customer_support_agent.db.customer_support_db import CustomerSupportDB


@pytest.mark.asyncio
async def test_db_lookup_and_search_methods_map_rows(monkeypatch):
    db = CustomerSupportDB("postgresql://fake")
    rows = [{"order_id": "o1", "status": "shipped"}]

    async def fetch(*args):
        return rows

    monkeypatch.setattr(db, "_fetch_all", fetch)
    assert (await db.get_order("o1")).status == "shipped"
    rows[:] = [{"product_id": "p1", "name": "Widget", "price": 10}]
    assert (await db.get_product("p1")) is not None

    async def no_rows(*args):
        return []

    monkeypatch.setattr(db, "_fetch_all", no_rows)
    assert await db.search_products("widget") == []
    assert await db.search_support_documents("policy") == []


@pytest.mark.asyncio
async def test_db_policy_and_escalation_use_dependencies(monkeypatch):
    db = CustomerSupportDB("postgresql://fake")

    async def no_documents(*args, **kwargs):
        return []

    monkeypatch.setattr(db, "search_support_documents", no_documents)
    assert await db.get_cancellation_policy() is None

    class Cursor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def execute(self, query, params):
            self.params = params

        async def fetchone(self):
            return ("ticket-1",)

    class Connection:
        def cursor(self):
            return Cursor()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def commit(self):
            pass

    class Psycopg:
        @staticmethod
        async def connect(*args, **kwargs):
            return Connection()

    monkeypatch.setattr(
        "customer_support_agent.db.customer_support_db.psycopg.AsyncConnection", Psycopg
    )
    assert await db.create_escalation("c1", "reason", "summary") == "ticket-1"
