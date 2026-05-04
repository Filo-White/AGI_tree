from __future__ import annotations
from pydantic import BaseModel
from typing import Optional


class TreeNodeConfig(BaseModel):
    id: str
    name: str
    model: str
    role: str
    system_prompt: str
    children: list[TreeNodeConfig] = []


class QueryRequest(BaseModel):
    query: str
    document_context: Optional[str] = None


class ScoreResult(BaseModel):
    node_id: str
    node_name: str
    score: float


class LeafResponse(BaseModel):
    node_id: str
    node_name: str
    response: str
    score: float
