from langchain_core.pydantic_v1 import BaseModel, Field


class Facts(BaseModel):
    facts: list = Field(description="list of bullet points of the facts from the passage")
