from rag_app.models.response_model import ResponseModel


def test_response_model_defaults() -> None:
    response = ResponseModel()

    assert response.success is True
    assert response.message is None


def test_response_model_with_message() -> None:
    response = ResponseModel(
        success=False,
        message="Something went wrong.",
    )

    assert response.success is False
    assert response.message == "Something went wrong."
