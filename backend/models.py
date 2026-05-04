from __future__ import annotations
from pydantic import BaseModel
from typing import Optional


class TreeNodeConfig(BaseModel):
    id: str
    name: str
    model: str
    role: str  # "root" | "node" | "leaf"
    system_prompt: str
    context: str = ""  # document excerpts relevant to this node
    children: list[TreeNodeConfig] = []


class TopicInfo(BaseModel):
    name: str
    subtopics: list[str] = []
    excerpts: str = ""  # relevant text from the document


class DocumentAnalysis(BaseModel):
    topics: list[TopicInfo] = []


class QueryRequest(BaseModel):
    query: str


class ScoreResult(BaseModel):
    node_id: str
    node_name: str
    score: float


class LeafResponse(BaseModel):
    node_id: str
    node_name: str
    response: str
    score: float
