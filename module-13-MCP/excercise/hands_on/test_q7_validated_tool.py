import re
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ValidationServer")


class OrderService:
    def get(self, order_id):
        print("Underlying system called!")
        return {"status": "shipped"}


order_service = OrderService()


@mcp.tool()
def get_order_status(order_id: str) -> dict:
    """Get status of an order."""
    if not re.fullmatch(r"ORD-\d{8}", order_id):
        raise ValueError("Invalid order ID format")

    return order_service.get(order_id)


def test_invalid_order_id():
    try:
        get_order_status("INVALID-123")
        assert False, "Expected validation error"
    except ValueError as e:
        assert str(e) == "Invalid order ID format"
