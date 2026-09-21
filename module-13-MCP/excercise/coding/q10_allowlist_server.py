from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Capability Allowlist Server")


@mcp.tool()
def get_employee(employee_id: str) -> dict:
    """Get employee information."""
    employees = {
        "EMP-001": {
            "id": "EMP-001",
            "name": "Harshil",
            "department": "Engineering",
        },
        "EMP-002": {
            "id": "EMP-002",
            "name": "Rahul",
            "department": "HR",
        },
    }

    employee = employees.get(employee_id)

    if employee is None:
        raise ValueError(f"Employee '{employee_id}' not found")

    return employee


@mcp.tool()
def search_documents(query: str) -> dict:
    """Search internal documents."""
    return {
        "query": query,
        "results": [
            "Employee Handbook",
            "Company Security Policy",
        ],
    }


@mcp.tool()
def delete_employee(employee_id: str) -> dict:
    """Delete an employee."""
    return {"message": f"Employee {employee_id} deleted"}


@mcp.tool()
def update_employee(employee_id: str, department: str) -> dict:
    """Update an employee."""
    return {
        "message": f"Employee {employee_id} updated",
        "department": department,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
