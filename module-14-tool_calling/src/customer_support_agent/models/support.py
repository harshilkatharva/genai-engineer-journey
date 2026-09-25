from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class SupportDocument(BaseModel):
    chunk_id: str
    content: str
    document_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Order(BaseModel):
    order_id: str
    customer_id: str | None = None
    status: str
    ordered_at: date | None = None
    total_amount: float | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)


class Product(BaseModel):
    product_id: str
    name: str
    description: str | None = None
    price: float | None = None
    available: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class CancellationPolicy(BaseModel):
    policy_name: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
