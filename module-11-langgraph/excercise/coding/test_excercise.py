import pytest
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

# Task 4


class CountState(TypedDict):
    count: int


def increment(state: CountState):
    return {"count": state["count"] + 1}


def build_graph(checkpointer):
    builder = StateGraph(CountState)

    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)

    return builder.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_resume_from_checkpoint():
    checkpointer = InMemorySaver()

    # Process 1
    graph1 = build_graph(checkpointer)

    config = {"configurable": {"thread_id": "test-thread"}}

    result1 = await graph1.ainvoke(
        {"count": 0},
        config=config,
    )

    assert result1["count"] == 1

    # Simulate process restart:
    # create a completely new graph instance
    graph2 = build_graph(checkpointer)

    # Resume using the same thread_id
    state = await graph2.aget_state(config)

    assert state.values["count"] == 1


# Task 5


class SafeguardState(TypedDict, total=False):
    max_iteration: int
    iteration_no: int
    answer: str


def increase_interation(state: SafeguardState) -> dict:
    return {"iteration_no": state.get("iteration_no", 0) + 1}


def genrate_answer(state: SafeguardState) -> dict:
    return {"answer": f"Final answer Genrated after {state['iteration_no']} iteration"}


def route_after_iteration(state: SafeguardState) -> str:
    return (
        "increase_interation"
        if state["iteration_no"] < state.get("max_iteration", 3)
        else "genrate_answer"
    )


def build_safeguard_graph():
    builder = StateGraph(SafeguardState)

    builder.add_node("increase_interation", increase_interation)
    builder.add_node("genrate_answer", genrate_answer)

    builder.add_edge(START, "increase_interation")
    builder.add_conditional_edges(
        "increase_interation", route_after_iteration, ["increase_interation", "genrate_answer"]
    )
    builder.add_edge("genrate_answer", END)

    graph = builder.compile()

    return graph


def test_safeguard_graph():
    graph = build_safeguard_graph()

    result = graph.invoke({})

    assert "Genrated after 3 iteration" in result["answer"]

    result = graph.invoke({"max_iteration": 2})

    assert "Genrated after 2 iteration" in result["answer"]


# Task 8


class Ticket(TypedDict, total=False):
    ticket_id: str
    customer: str
    amount: float
    reason: str
    approval: str
    result: str


def prepare_refund(state: Ticket) -> dict:
    print("Prepare refund")

    return {"customer": "Test 1", "amount": 258.5, "reason": "for testing purpose"}


def human_approval(state: Ticket) -> dict:
    print("Human approval")

    decision = interrupt(
        {
            "type": "refund_approval",
            "ticked_id": state["ticket_id"],
            "customer": state["customer"],
            "amount": state["amount"],
            "reason": state["reason"],
            "message": "Approve this refund ?",
        }
    )

    return {
        "approval": decision,
    }


def route_result(state: Ticket) -> str:
    if state["approval"] == "approved":
        return "execute_refund"

    return "reject_refund"


def execute_refund(state: Ticket) -> dict:
    print("Execute refund")
    return {"result": f"Refund {state['amount']} to {state['customer']}"}


def reject_refund(state: Ticket) -> dict:
    print("Reject refund")

    return {"result": f"Refund rejected by human reviewer of {state['customer']}"}


def build_human_loop_graph():
    graph = StateGraph(Ticket)

    graph.add_node("prepare_refund", prepare_refund)
    graph.add_node("human_approval", human_approval)
    graph.add_node("execute_refund", execute_refund)
    graph.add_node("reject_refund", reject_refund)

    graph.add_edge(START, "prepare_refund")
    graph.add_edge("prepare_refund", "human_approval")
    graph.add_conditional_edges("human_approval", route_result, ["execute_refund", "reject_refund"])
    graph.add_edge("execute_refund", END)
    graph.add_edge("reject_refund", END)

    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)


def test_human_in_loop_approved():
    config = {"configurable": {"thread_id": "refund-5001"}}

    app = build_human_loop_graph()
    first = app.invoke({"ticket_id": "test-001"}, config=config, version="v2")

    assert first.interrupts

    resume = app.invoke(Command(resume="approved"), config=config, version="v2")

    assert "Refund 258.5" in resume.value["result"]


def test_human_in_loop_rejected():
    config = {"configurable": {"thread_id": "refund-5001"}}

    app = build_human_loop_graph()
    first = app.invoke({"ticket_id": "test-001"}, config=config, version="v2")

    assert first.interrupts

    resume = app.invoke(Command(resume="rejected"), config=config, version="v2")

    assert "Refund rejected" in resume.value["result"]
