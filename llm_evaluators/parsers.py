from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class CompareVanillaCOTScore(BaseModel):
    justification: str = Field(description="Justification for the verdict")
    verdict: str = Field(description="final verdict for the comparison. Give 'A' if assistant A is better, 'B' if assistant B is better, 'C' for a tie, and 'D' if both the answers are unacceptable.")
    
class CompareVanillaScore(BaseModel):
    verdict: str = Field(description="final verdict for the comparison. Give 'A' if assistant A is better, 'B' if assistant B is better, 'C' for a tie, and 'D' if both the answers are unacceptable.")
    
class CompareRulesScore(BaseModel):
    justification: str = Field(description="Justification for the verdict")
    verdict: str = Field(description="final verdict for the comparison. Give 'A' if assistant A is better, 'B' if assistant B is better, 'C' for a tie, and 'D' if both the answers are unacceptable.")
    
class CompareAxesRulesScore(BaseModel):
    justification: str = Field(description="Justification for the verdict")
    verdict: str = Field(description="final verdict for the comparison. Give 'A' if assistant A is better, 'B' if assistant B is better, 'C' for a tie, and 'D' if both the answers are unacceptable.")
    
class CompareAxesScore(BaseModel):
    justification: str = Field(description="Justification for the verdict")
    verdict: str = Field(description="final verdict for the comparison. Give 'A' if assistant A is better, 'B' if assistant B is better, 'C' for a tie, and 'D' if both the answers are unacceptable.")
    
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
    
class ReferenceScore(BaseModel):
    justification: str = Field(description="Justification for the score")
    score: int = Field(description="Score for the metric")