import argparse
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser

from perturbations.utils import create_jsonl, dump_jsonl
from perturbations.parsers import DirectError


def calculation_errors(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a reasoning question and it's corresponding answer, generate calculation errors.
    
    Args:
        args (argparse.Namespace): Command line arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Given a reasoning question and it's corresponding answer, generate calculation errors.\n"
            "Make sure the final answer is unchanged, only introduce erros in the intermediate calculation steps.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/calculation-errors-temp{args.temperature}.jsonl')
    return


def final_answer_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a reasoning question and it's corresponding answer, generate final answer errors.
    
    Args:
        args (argparse.Namespace): Command line arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        # TODO: condesing the answer and just returning the final answer -> fix this
        PROMPT = (
            "Given a reasoning question and it's corresponding answer, generate final answer errors.\n"
            "Only change the final calculation step abd introduce an error.\n"
            "Do not change anything except the final answer.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/final-answer-errors-temp{args.temperature}.jsonl')
    return


def final_answer_perturbations_v2(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a reasoning question and it's corresponding answer, generate final answer errors.
    
    Args:
        args (argparse.Namespace): Command line arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        # TODO: condesing the answer and just returning the final answer -> fix this
        PROMPT = (
            "Given a reasoning question and it's corresponding solution, generate final answer errors.\n"
            "Follow these instructions carefully to introduce the error:\n"
            "1. Ensure all intermediate steps in the solution are correct and unchanged.\n"
            "2. Only change the final calculation step and introduce an error in the final answer.\n"
            "3. The final answer should be incorrect, but the logical flow and reasoning steps leading up to it must remain accurate and consistent.\n"
            "4. Maintain the structure and clarity of the original solution, making sure the error is subtle and isolated to the final answer.\n"
            "5. The error can be due to a simple arithmetic mistake, a wrong conclusion drawn from the correct steps, or a minor oversight in the last calculation.\n\n"
            "Ensure that you give the entire gold answer with the introduced error along with the explanation for the error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/final-answer-errors-temp{args.temperature}.jsonl')
    return

def final_answer_perturbations_v3(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a reasoning question and it's corresponding answer, generate final answer errors.
    
    Args:
        args (argparse.Namespace): Command line arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        # TODO: condesing the answer and just returning the final answer -> fix this
        PROMPT = (
            "I want to create a few questions to test how good my evaluations system is. For this, I have collected some reasoning questions along with their correct solutions i.e., gold answers."
            "For testing my evaluators, I want to make some errors in the final answer while the intermediate steps in the solution remain same to see if it is being detected by the system or not.\n"
            "Follow these instructions carefully to introduce the error:\n"
            "1. Ensure all intermediate steps in the solution are correct and unchanged.\n"
            "2. Only change the final calculation step and introduce an error in the final answer.\n"
            "3. The final answer should be incorrect, but the logical flow and reasoning steps leading up to it must remain accurate and consistent.\n"
            "4. Maintain the structure and clarity of the original solution, making sure the error is subtle and isolated to the final answer.\n"
            "5. The error can be due to a simple arithmetic mistake, a wrong conclusion drawn from the correct steps, or a minor oversight in the last calculation.\n\n"
            "Given the question and the gold answer (i.e., solution), return the 'entire' solution with the introduced error in the final answer.\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error and also give an explanation about the error introduced.\n"
            f"{parser.get_format_instructions()}\n\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/final-answer-errors-temp{args.temperature}.jsonl')
    return

def incorrect_units(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given the following answer for a reasoning question, change the units of measurement.
    
    Args:
        args (argparse.Namespace): Command line arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Given the following answer for a reasoning question, change the units of measurement.\n"
            "Make sure the final answer is unchanged, only introduce erros in the units of measurement.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/incorrect-units-temp{args.temperature}.jsonl')
    return


def operation_order(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given the following answer for a reasoning question, change the order of operations.
    
    Args:
        args (argparse.Namespace): Command line arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        # TODO: this is also changing the formulas, I think this is good only
        PROMPT = (
            "Given the following answer for a reasoning question, change the order of operations by NOT following PEMDAS or BODMAS.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/operation-order-temp{args.temperature}.jsonl')
    return


def wrong_formula(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given the following answer for a reasoning question, change the formula.
    
    Args:
        args (argparse.Namespace): Command line arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Based on the given question and its corresponding answer, apply an incorrect formula instead of the expected one to solve the reasoning question.\n"
            "Make sure the final answer is unchanged, only introduce erros in the formula.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/wrong-formula-temp{args.temperature}.jsonl')
    return

def wrong_formula_v2(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given the following answer for a reasoning question, change the formula.
    
    Args:
        args (argparse.Namespace): Command line arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "I want to create a few questions to test how good my evaluations system is. For this, I have collected some reasoning questions along with their correct solutions i.e., gold answers."
            "For testing my evaluators, I want to make some subtle errors in the formulae used in the solution to see if it is being detected by the system or not.\n"
            "Given a question and a solution (gold answer), I want you to help me in creating such errored solutions. You will only make subtle changes in some formula used in the solution that will render the solution incorrect while not changing any other step including the final answer.\n\n"    
            "Make sure the final answer is unchanged, only introduce errors in the formulae of the solution.\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/wrong-formula-temp{args.temperature}.jsonl')
    return


def copying_numbers_errors(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given the following reasoning question and it's answer, introduce errors in copying the numbers from the question to the answer.
    
    Args:
        args (argparse.Namespace): Command line arguments
        testset (pd.DataFrame): Testset
        
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Given the following reasoning question and it's answer, introduce errors in copying the numbers from the question to the answer.\n"
            "Some of the following errors can be introduced are:\n"
            "1. Transposing Digits\n"
            "2. Omitting Digits or Symbols\n"
            "3. Adding Extra Digits or Symbols\n"
            "4. Writing Incorrect Exponents\n"
            "5. Writing Incorrect Decimals\n"
            "6. Writing Incorrect Fractions\n"
            "Pick one or more of the above errors and introduce them in the answer.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/copying-numbers-errors-temp{args.temperature}.jsonl')
    return