# Hospitality Booking Assistant: Agent-Only vs. Hybrid Architecture

## 1. Project Overview

Build a hospitality assistant that helps users explore hotels, compare amenities and reviews, investigate existing reservations, and complete booking-related tasks.

The system uses an **LLM-based classifier as the central routing controller**. Based on the user's intent and current conversation, it routes each request to a direct response, a RAG-based hotel discovery workflow, a task-oriented agent, or human assistance.

The primary recommendation is a **hybrid architecture**: use RAG for hotel discovery and conversational exploration, and use the agent for specific transactional tasks.

## 2. Initial Query Classification and Routing

Every incoming user query first passes through the LLM-based classifier.

The classifier determines the appropriate route:

- **Simple questions or greetings:** Respond directly when possible, or ask a clarifying question.
- **Hotel discovery and follow-up questions:** Route to the RAG workflow.
- **Booking, payment, refunds, and existing reservation operations:** Route to the task-oriented agent.
- **Requests requiring human intervention:** Escalate to the appropriate person or support team.

The classifier also considers the current conversation. For example, a user already comparing hotels can continue that conversation in the RAG workflow until they request a transactional operation.

## 3. Option A: Agent-Only Architecture

In this approach, the classifier routes hotel-related requests to the agent, which performs both discovery and transactional tasks.

### Example conversation

**Step 1 — Discover hotels**

User: "Find hotels in Goa near the beach, with a swimming pool and a budget of ₹1,000–₹20,000 per night."

The classifier routes the request to the agent. The agent decides which tools to use, retrieves matching hotels and their details, and asks the LLM to generate an overview for the user.

**Step 2 — Ask about a specific hotel**

User: "Does Hotel A offer free breakfast and late checkout?"

The classifier routes the request to the agent. The agent selects the appropriate tool, retrieves hotel details, and generates an answer.

**Step 3 — Compare hotels**

User: "Compare Hotel A, Hotel B, and Hotel C based on their amenities, reviews, and beach access."

The agent selects the necessary tools, retrieves and organizes the details, and generates the comparison.

**Step 4 — Continue exploring**

The user asks approximately ten follow-up questions about amenities, reviews, services, and hotel policies. The agent repeatedly determines which tools to call and generates responses.

**Step 5 — Book a hotel**

Once the user selects a hotel, the agent checks live availability and date-specific pricing, confirms the details, and coordinates the reservation and payment process.

### Estimated resource usage for ten discovery questions

Assume each question needs one classifier call and approximately two further LLM calls in the agent loop.

- **LLM calls:** 10 classifier calls + 20 agent calls = approximately 30.
- **Tool usage:** Potentially 10–20 or more calls, depending on the questions and how much information the agent reuses.
- **Latency:** Higher when the agent repeatedly performs tool selection, retrieval, and answer generation.
- **Token and cost usage:** Potentially higher because of repeated reasoning calls, tool descriptions, and conversation context.

These are illustrative estimates, not guaranteed measurements.

## 4. Option B: Hybrid Architecture — Recommended

In this approach, the classifier remains the first component, but hotel discovery is delegated to a dedicated RAG workflow. The agent is reserved for transactional operations.

### Phase A — Hotel discovery

**Step 1: Classify and route**

User: "Find hotels in Goa near the beach, with a swimming pool and a budget of ₹1,000–₹20,000 per night."

The classifier identifies the hotel-discovery intent and routes the request to the RAG workflow.

**Step 2: Retrieve and temporarily retain details**

The workflow retrieves relevant hotels and collects available descriptions, amenities, review information, images, and indicative price ranges.

The results are retained in session context or a temporary cache so that follow-up questions can reuse them.

**Step 3: Present the hotel overview**

The LLM generates a concise overview of approximately 10–15 hotels, including their key features and image links where available.

### Phase B — Continue the discovery conversation

**Step 4: Answer specific questions**

User: "Does Hotel A offer free breakfast?"

The classifier routes the query back to the active RAG workflow. The system answers using the retained hotel information if it is available and sufficiently fresh.

**Step 5: Compare hotels**

User: "Compare Hotel A and Hotel B based on breakfast, swimming pools, reviews, and beach access."

The workflow compares the retained information and generates a structured response without unnecessarily repeating the original hotel-detail retrieval.

**Step 6: Continue exploration**

The user asks ten follow-up questions about amenities, reviews, services, and comparisons. The classifier continues routing those questions to the RAG workflow unless the user's intent changes.

### Phase C — Transactional operations

**Step 7: Transfer to the agent**

User: "Book Hotel A for two people from November 10 to November 13."

The classifier recognizes a booking request and routes it to the agent.

The agent checks live room availability and the current price, confirms the booking details, and coordinates booking and payment. Refund requests are handled through the agent, with human approval required before an actual refund is executed.

## 5. Illustrative LLM Calls, Latency, and Resource Comparison

Consider ten hotel-exploration questions after the user's initial discovery request. Assume that the hotel information is already available in the discovery context.

| Measurement | Agent-only | Hybrid |
|---|---:|---:|
| Classifier calls | 10 | 10 |
| Further LLM calls for responses and reasoning | ~20 | ~10 |
| **Total LLM calls** | **~30** | **~20** |
| Relative downstream LLM calls | ~20 | ~10 |
| Hotel information retrieval | Potentially repeated tool calls | Reuses retained results where appropriate |
| Expected latency | Higher due to extra agent reasoning and tool execution | Lower for cached discovery and comparison |
| Token and API cost | Potentially higher | Potentially lower |
| Best suited for | Flexible investigations and transactional tasks | Repeated discovery, questions, and comparisons |

For a simple demonstration, suppose the classifier takes 0.2–0.5 seconds per call, a response-generation call takes 0.6–1.2 seconds, and cached retrieval takes 0.1–0.3 seconds.

Under these illustrative assumptions, ten sequential discovery questions might take roughly **9–20 seconds with the hybrid workflow**, compared with approximately **18–35 seconds with the agent-only approach**, depending on the agent's actual execution path and tool latency.

The hybrid approach is expected to use fewer LLM calls for this example. Actual latency and cost depend on the model, prompt length, provider, retrieval backend, caching, network conditions, and number of agent iterations.

## 6. Why Hybrid Is the Recommended Architecture

The LLM-based classifier provides the initial routing decision in both designs. The difference is what happens after routing.

In the agent-only approach, the agent handles hotel exploration as well as transactional tasks. Its flexibility can be useful, but repeated tool selection and retrieval may introduce unnecessary reasoning overhead.

In the hybrid approach, the RAG workflow handles hotel discovery, follow-up questions, and comparisons using retained information. The agent handles operations that need live data, external APIs, multiple execution steps, or controlled state changes.

Indicative hotel price ranges can be displayed during discovery. Exact date-specific prices and room availability must be verified through a live booking service before a reservation is made.

**Final recommendation:** Use an LLM-based classifier to control routing, a RAG workflow to support hotel exploration, and a task-oriented agent for booking, payment, refunds, and other transactional operations. This architecture aims to reduce unnecessary agent iterations while keeping flexible execution available when the task requires it.