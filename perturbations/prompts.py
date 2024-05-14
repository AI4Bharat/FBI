from langchain_core.pydantic_v1 import BaseModel, Field


class Facts(BaseModel):
    facts: str = Field(description="bullet points of facts from the passage")