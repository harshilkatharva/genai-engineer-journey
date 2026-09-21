from mcp.server.mcpserver import MCPServer

mcp = MCPServer("LeastPrivilegeServer")


@mcp.tool()
def get_employee(employee_id: str) -> dict:
    """Read employee information."""
    if employee_id != "EMP-001":
        raise PermissionError("Access denied")

    return {"employee_id": "EMP-001", "name": "Alice"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
