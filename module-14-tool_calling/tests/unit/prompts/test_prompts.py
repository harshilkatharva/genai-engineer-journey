from customer_support_agent.prompts.prompt_manager import SupportPromptManager


def test_prompt_manager_builds_system_and_user_messages():
    messages = SupportPromptManager().build_messages("Where is my order?")
    assert [message.role for message in messages] == ["system", "user"]
    assert messages[1].content == "Where is my order?"
    assert "Never invent" in messages[0].content
