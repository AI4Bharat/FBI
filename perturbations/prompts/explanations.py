import argparse
import pandas as pd
from ast import literal_eval
from typing import List

from langchain_core.output_parsers import JsonOutputParser

from perturbations.utils import create_jsonl
from perturbations.parsers import JustificationEvalParser


def analyze_explanations(args: argparse.Namespace, testset: pd.DataFrame) -> List[dict]:
    """Given the evaluations of 2 model outputs, compare the explanations to check if an error has been identified in the explanation.
    
    Args:
        args (argparse.Namespace): Arguments from the command line
        testset (pd.DataFrame): DataFrame of testset data
        
    Returns:
        List of JSON objects    
    """
    jsons = []
    parser = JsonOutputParser(pydantic_object=JustificationEvalParser)
    for _, row in testset.iterrows():
        orig_explanation = row['orig_explanation']
        pert_explanation = row['pert_explanation']
        orig_score = row['orig_score']
        pert_score = row['pert_score']

        PROMPT = (
            "Given the evaluation of a model output, please check if an error has been identified in the explanation.\n"
            "It is possible that the score is quite high, but the explanation may still identify an error.\n"
            "Only identify an error if the explanation clearly identifies an error in the model output.\n\n"
            "Generate an explanation of the identified error in the 'Score Justification' then return 'true' if the 'Score Justification' has identified an error, else return 'false'.\n\n"
            f"{parser.get_format_instructions()}\n\n"
            "Score Justtification:\n"
            f"{pert_explanation}\n\n"
            "Score:\n"
            f"{pert_score}\n\n"
        )
        
        dict_ = create_jsonl(row['qid'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    return jsons