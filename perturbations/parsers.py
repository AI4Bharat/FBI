import re
import json
import ast
import warnings
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, ValidationError

# Suppress only SyntaxWarnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

class Facts(BaseModel):
    facts: list = Field(description="list of bullet points of the facts from the passage")

class Errors(BaseModel):
    errors: list = Field(description="list of facts with introduced error")
    explanation: str = Field(description="explanation of the introduced error")

class Stitch(BaseModel):
    answer: str = Field(description="gold answer with the introduced errors")

class DirectError(BaseModel):
    err_answer: str = Field(description="entire answer with the introduced error")
    explanation: str = Field(description="explanation of the introduced error")

class MultipleDirectErrors(BaseModel):
    err_answers: list = Field(description="list of answers with the introduced errors")
    explanations: list = Field(description="list of explanations of the introduced errors")
    

class CustomJsonOutputParser:
    """
    A custom JSON output parser that extracts the outermost JSON object from a string,
    attempts to parse it, and validates it against a specified Pydantic model.

    Attributes:
        pydantic_model (BaseModel): The Pydantic model class used for validating the parsed JSON.

    Methods:
        find_outermost_json(data): Extracts the outermost JSON object from a string.
        invoke(data): Parses and validates the JSON object against the Pydantic model.
    """

    def __init__(self, pydantic_model):
        """
        Initializes the parser with a Pydantic model for JSON validation.

        Args:
            pydantic_model (BaseModel): A subclass of pydantic.BaseModel that defines the expected JSON schema.

        Raises:
            ValueError: If the provided model is not a subclass of pydantic.BaseModel.
        """
        if not issubclass(pydantic_model, BaseModel):
            raise ValueError("pydantic_model must be a subclass of pydantic.BaseModel")
        self.pydantic_model = pydantic_model

    def find_outermost_json(self, data):
        """
        Extracts the outermost JSON string from a potentially nested or malformed JSON input.

        Args:
            data (str): The string containing JSON data.

        Returns:
            str: The outermost JSON string if found.

        Raises:
            Exception: If no valid JSON object could be extracted.
        """
        stack = []
        in_string = False
        escape = False
        start = -1

        for i, char in enumerate(data):
            if char == '"' and not escape:
                in_string = not in_string
            elif char == '\\' and in_string:
                escape = not escape
            elif char == '{' and not in_string:
                if not stack:
                    start = i
                stack.append(char)
            elif char == '}' and not in_string:
                if stack:
                    stack.pop()
                if not stack and start != -1:
                    return data[start:i+1]
            elif escape:
                escape = False
        
        raise Exception("No valid JSON could be extracted")

    def invoke(self, data):
        """
        Parses the extracted JSON string and validates it against the Pydantic model.

        Args:
            data (str): The string from which to extract and validate JSON.

        Returns:
            BaseModel: An instance of the Pydantic model populated with validated data.

        Raises:
            Exception: If the JSON string is invalid or if validation against the Pydantic model fails.
        """
        json_string = self.find_outermost_json(data)
        if json_string:
            try:
                json_data = json.loads(json_string)
            except json.JSONDecodeError:
                try:
                    json_data = ast.literal_eval(json_string)
                except Exception as e:
                    raise Exception("JSON parsing failed") from e

            try:
                self.pydantic_model(**json_data)
                return json_data
            except ValidationError as e:
                raise Exception("JSON validation failed") from e

        raise Exception("No valid JSON could be extracted")