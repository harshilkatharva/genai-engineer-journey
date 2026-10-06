from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Literal

from anthropic.types import ToolChoiceAnyParam, ToolChoiceAutoParam, ToolChoiceToolParam, ToolParam
from openai.types.responses import FunctionToolParam, ToolChoiceFunctionParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ToolArgumentError(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class ToolChoice(BaseModel):
    mode: Literal["auto", "any", "specific"] = "auto"
    tool_name: str | None = None

    # Ensures a specific tool name is supplied only for specific-tool selection.
    @model_validator(mode="after")
    def validate_tool_name(self) -> ToolChoice:
        if (self.mode == "specific") != (self.tool_name is not None):
            raise ValueError("tool_name is required only when mode is 'specific'")
        return self

    # Converts the tool-selection mode to the OpenAI API format.
    def to_openai(self) -> Literal["auto", "required"] | ToolChoiceFunctionParam:
        if self.mode == "auto":
            return "auto"
        if self.mode == "any":
            return "required"
        return {"type": "function", "name": self.tool_name or ""}

    # Converts the tool-selection mode to the Anthropic API format.
    def to_anthropic(self) -> ToolChoiceAutoParam | ToolChoiceAnyParam | ToolChoiceToolParam:
        if self.mode == "auto":
            return {"type": "auto"}
        if self.mode == "any":
            return {"type": "any"}
        return {"type": "tool", "name": self.tool_name or ""}

    # Converts the tool-selection mode to the Google API format.
    def to_google(self) -> dict[str, Any]:
        config: dict[str, Any] = {"mode": "AUTO" if self.mode == "auto" else "ANY"}
        if self.mode == "specific" and self.tool_name:
            config["allowed_function_names"] = [self.tool_name]
        return config


class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    argument_model: type[BaseModel]
    handler: Callable[..., Any] | None = Field(default=None, exclude=True)

    # Returns the JSON schema for this tool's argument model.
    @property
    def parameters(self) -> dict[str, Any]:
        return self.argument_model.model_json_schema()

    # Serializes the tool definition for OpenAI function calling.
    def to_openai_tool(self) -> FunctionToolParam:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": False,
        }

    # Serializes the tool definition for Anthropic tool use.
    def to_anthropic_tool(self) -> ToolParam:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    # Serializes the tool definition for Google function calling.
    def to_google_function_declaration(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    # Parses and validates raw arguments against this tool's argument model.
    def parse_arguments(
        self, raw_arguments: str | Mapping[str, Any] | BaseModel
    ) -> tuple[BaseModel | None, ToolArgumentError | None]:
        if isinstance(raw_arguments, str):
            try:
                parsed: Any = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                return None, ToolArgumentError(
                    code="invalid_json",
                    message="Tool arguments are not valid JSON.",
                    details=[{"line": error.lineno, "column": error.colno}],
                )
        elif isinstance(raw_arguments, self.argument_model):
            return raw_arguments, None
        elif isinstance(raw_arguments, Mapping):
            parsed = dict(raw_arguments)
        else:
            return None, ToolArgumentError(
                code="invalid_argument_type",
                message="Tool arguments must be a JSON object or JSON object string.",
            )

        if not isinstance(parsed, dict):
            return None, ToolArgumentError(
                code="arguments_not_object",
                message="Tool arguments must decode to a JSON object.",
            )

        try:
            return self.argument_model.model_validate(parsed), None
        except (ValidationError, TypeError, ValueError) as error:
            return None, ToolArgumentError(
                code="validation_error",
                message="Tool arguments did not match the tool's argument model.",
                details=(
                    [
                        {
                            "location": list(item["loc"]),
                            "message": item["msg"],
                            "type": item["type"],
                        }
                        for item in error.errors()
                    ]
                    if isinstance(error, ValidationError)
                    else [{"message": str(error), "type": type(error).__name__}]
                ),
            )
