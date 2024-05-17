from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class CompareVanillaScore(BaseModel):
    justification: str = Field(description="Justification for the verdict")
    verdict: int = Field(description="final verdict for the comparison")
    
    
class SingleVanillaScore(BaseModel):
    justification: str = Field(description="Justification for the score")
    score: int = Field(description="Score for the metric")
    
class SingleAxesRubricsScore(BaseModel):
    justification: str = Field(description="Justification for the score")
    score: int = Field(description="Score for the metric")