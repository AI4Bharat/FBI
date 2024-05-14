from langchain_core.pydantic_v1 import BaseModel, Field


class Facts(BaseModel):
    facts: list = Field(description="list of bullet points of the facts from the passage")

class Errors(BaseModel):
    errors: list = Field(description="list of facts with introduced error")
    explanation: str = Field(description="explanation of the introduced error")

class Stitch(BaseModel):
    answer: str = Field(description="gold answer with the introduced errors")