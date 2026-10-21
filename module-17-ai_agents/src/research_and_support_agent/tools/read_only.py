"""Read-only operational tools for Module 17."""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote

import httpx
from bs4 import BeautifulSoup
from genai_core.observability import get_logger
from genai_core.services import RetrievalService

from ..memory.manager import AgentMemoryManager
from ..models import CustomerAccount, SupportTicket
from .sanitizer import sanitize_tool_output

_logger = get_logger("research_and_support_agent.tools.read_only")


async def search_knowledge_base(
    query: str,
    top_k: int = 3,
    retrieval_service: RetrievalService | None = None,
) -> str:
    """Search internal documentation and policy manuals for company guidelines and technical rules."""
    if not retrieval_service:
        return "Error: Knowledge base retrieval service is not configured."

    try:
        response = await retrieval_service.search(query=query, top_k=top_k)
        if not response.results:
            return f"No relevant documentation found for: '{query}'."

        snippets: list[str] = []
        for idx, res in enumerate(response.results, 1):
            source_tag = res.metadata.get("source", res.source_id or "internal_kb")
            snippets.append(f"[{idx}] Source: {source_tag} (Score: {res.score:.2f})\n{res.content}")

        raw_text = "\n\n".join(snippets)
        return sanitize_tool_output(raw_text, source="knowledge_base")
    except Exception as exc:  # noqa: BLE001
        return f"Error querying knowledge base: {exc}"


async def search_web(
    query: str,
    max_results: int = 3,
    timeout_s: float = 8.0,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Search public web pages for documentation, public updates, and technical guides."""
    if not query or not query.strip():
        return "Error: Empty search query."

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    url = "https://html.duckduckgo.com/html/"
    data = {"q": query.strip()}

    try:
        if client is not None:
            resp = await client.post(url, data=data, headers=headers, timeout=timeout_s)
        else:
            async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as local_client:
                resp = await local_client.post(url, data=data, headers=headers)

        if resp.status_code != 200:
            return f"Web search service returned status {resp.status_code}."

        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[dict[str, str]] = []

        for result_elem in soup.find_all("div", class_="result", limit=max_results * 2):
            title_elem = result_elem.find("a", class_="result__a")
            snippet_elem = result_elem.find("a", class_="result__snippet")

            if title_elem and snippet_elem:
                title = title_elem.get_text(strip=True)
                href_val = title_elem.get("href", "")
                raw_href = href_val if isinstance(href_val, str) else str(href_val or "")
                snippet = snippet_elem.get_text(strip=True)

                # Clean DuckDuckGo redirect link
                clean_url = raw_href
                if "uddg=" in raw_href:
                    part = raw_href.split("uddg=")[-1].split("&")[0]
                    clean_url = unquote(part)

                results.append({"title": title, "url": clean_url, "snippet": snippet})
                if len(results) >= max_results:
                    break

        if not results:
            return f"No web search results found for query: '{query}'."

        formatted: list[str] = []
        for i, item in enumerate(results, 1):
            formatted.append(f"[{i}] {item['title']}\nURL: {item['url']}\n{item['snippet']}")

        combined = "\n\n".join(formatted)
        return sanitize_tool_output(combined, source="web_search")

    except httpx.RequestError as exc:
        _logger.warning("Web search network error: %s", exc)
        return f"Web search could not be completed (network unreachable): {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Error executing web search: {exc}"


async def get_longterm_memory(
    user_id: str,
    query: str,
    top_k: int = 5,
    memory_manager: AgentMemoryManager | None = None,
) -> list[dict[str, Any]]:
    """Retrieve relevant long-term memory records (preferences, profile facts) for an authorized user."""
    if not memory_manager:
        return [{"error": "Memory manager is not configured."}]
    return await memory_manager.get_longterm_memory(user_id=user_id, query=query, top_k=top_k)


async def get_episodic_memory(
    user_id: str,
    action_type: str | None = None,
    limit: int = 15,
    memory_manager: AgentMemoryManager | None = None,
) -> list[dict[str, Any]]:
    """Retrieve past action history, parameters, and execution outcomes for an authorized user."""
    if not memory_manager:
        return [{"error": "Memory manager is not configured."}]
    return await memory_manager.get_episodic_memory(
        user_id=user_id, action_type=action_type, limit=limit
    )


async def ticket_status(
    ticket_id: str,
    ticket_store: dict[str, SupportTicket] | None = None,
) -> dict[str, Any]:
    """Retrieve the current lifecycle status, details, and assignment for a support ticket."""
    if ticket_store is None:
        return {"error": "Ticket store is not configured."}

    ticket = ticket_store.get(ticket_id)
    if not ticket:
        return {"error": f"Ticket '{ticket_id}' not found."}

    return {
        "ticket_id": ticket.ticket_id,
        "customer_id": ticket.customer_id,
        "title": ticket.title,
        "description": ticket.description,
        "status": str(ticket.status),
        "assigned_team": ticket.assigned_team,
        "priority": ticket.priority,
        "updated_at": ticket.updated_at,
    }


async def lookup_customer_account(
    customer_id: str,
    account_store: dict[str, CustomerAccount] | None = None,
) -> dict[str, Any]:
    """Retrieve customer account details, tier, balance, and open ticket count."""
    if account_store is None:
        return {"error": "Account store is not configured."}

    account = account_store.get(customer_id)
    if not account:
        return {"error": f"Customer '{customer_id}' not found."}

    return {
        "customer_id": account.customer_id,
        "name": account.name,
        "tier": account.tier,
        "balance_usd": account.balance_usd,
        "active_tickets": account.active_tickets,
        "risk_flag": account.risk_flag,
    }
