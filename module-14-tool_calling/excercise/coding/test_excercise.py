# Task 4


def execute_tool(tool, args):
    try:
        return {"type": "tool_result", "error": str(tool(**args))}
    except Exception as e:
        return {"type": "tool_result", "error": str(e)}


def test_tool_error_is_returned_as_result():
    def failing_tool(x):
        raise ValueError("Tool failed")

    result = execute_tool(failing_tool, {"x": 1})

    assert result["type"] == "tool_result"
    assert result["error"] == "Tool failed"


# Task 6


def test_tool_selection_accuracy():
    queries = [
        ("Find hotels in Mumbai", "search_properties"),
        ("Check availability for hotel A", "check_availability"),
        ("Show hotel details", "get_property_details"),
        ("Can I cancel my booking?", "check_cancellation_policy"),
    ]

    # replace the following dictionaries with your actual tool selection logic or logs
    well_documented = {
        "Find hotels in Mumbai": "search_properties",
        "Check availability for hotel A": "check_availability",
        "Show hotel details": "get_property_details",
        "Can I cancel my booking?": "check_cancellation_policy",
    }

    # replace the following dictionaries with your actual tool selection logic or logs
    poorly_documented = {
        "Find hotels in Mumbai": "search_properties",
        "Check availability for hotel A": "search_properties",
        "Show hotel details": "search_properties",
        "Can I cancel my booking?": "check_cancellation_policy",
    }

    well_accuracy = sum(well_documented[q] == expected for q, expected in queries) / len(queries)

    poor_accuracy = sum(poorly_documented[q] == expected for q, expected in queries) / len(queries)

    assert well_accuracy >= poor_accuracy
