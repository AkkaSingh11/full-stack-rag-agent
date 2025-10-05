from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


class RouteDecision(BaseModel):
    intent: str = Field(
        description="The classified intent: 'conversational' for greetings/chitchat, 'rag' for document/KB queries, or 'research' for web search questions."
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen."
    )


class RagJudge(BaseModel):
    sufficient: bool = Field(
        description="Whether the retrieved documents are sufficient to answer the user's question."
    )
    reasoning: str = Field(
        description="Brief explanation of why the retrieved information is or isn't sufficient."
    )
