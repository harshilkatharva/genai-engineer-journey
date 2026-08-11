from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str
    input_tokens: int
    output_tokens: int
