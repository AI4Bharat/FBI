from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Score(BaseModel):
    justification: str = Field(description="Justification for the score")
    score: int = Field(description="Score for the metric")
    