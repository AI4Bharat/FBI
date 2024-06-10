from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class CompareVanillaCOTScore(BaseModel):
    justification: str = Field(description="Justification for the verdict")
    verdict: int = Field(description="final verdict for the comparison")
    
class CompareVanillaScore(BaseModel):
    verdict: int = Field(description="final verdict for the comparison")
    
class CompareRulesScore(BaseModel):
    justification: str = Field(description="Justification for the verdict")
    verdict: int = Field(description="final verdict for the comparison")
    
class SingleVanillaScore(BaseModel):
    score: int = Field(description="Score for the Answer")
    
class SingleVanillaCOTScore(BaseModel):
    justification: str = Field(description="Justification for the score")
    score: int = Field(description="Score for the Answer")
    
class SingleRubricsScore(BaseModel):
    justification: str = Field(description="Justification for the score")
    score: int = Field(description="Score for the Answer")

class SingleAxesScore(BaseModel):
    justification: str = Field(description="Justification for the score")
    score: int = Field(description="Score for the metric")

class SingleAxesRubricsScore(BaseModel):
    justification: str = Field(description="Justification for the score")
    score: int = Field(description="Score for the metric")