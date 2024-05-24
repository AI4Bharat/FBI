import argparse
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser

from perturbations.utils import create_jsonl, dump_jsonl
from perturbations.parsers import DirectError


def incorrect_sequence(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    # TODO: can be used in long form perturbations too
    """Given a instruction-following question and it's corresponding answer, change the answer to follow the wrong sequence of steps in the instruction.
    
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
            "Given a instruction-following question and it's corresponding answer, change the answer to follow the given instructions in the wrong sequence.\n"
            "Only change the sequence of the steps and introduce an error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/incorrect-sequence-temp{args.temperature}.jsonl')
    return

def omit_step(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    # TODO: it is omiting parts of the answer, not the question
    """Given a instruction-following question and it's corresponding answer, omit a step in the answer.
    
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
            "Given a instruction-following question and it's corresponding answer, omit a step in the answer.\n"
            "Only omit one step and introduce an error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/omit-step-temp{args.temperature}.jsonl')
    return

def incomplete_execution(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    # TODO: remove the answer and regenerate it
    """Given an instruction-following question and its corresponding answer, ensure that the instructions are only partially executed.
    
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
            "Given an instruction-following question and its corresponding answer, ensure that the instructions are only partially executed.\n"
            "Only execute a part of the instructions and introduce an error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/incomplete-execution-temp{args.temperature}.jsonl')
    return

def misread_instructions(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    # TODO: should probably remove the answer and ask the model to generate the answer
    """Given an instruction-following question and its corresponding answer, misread the instructions such as misinterpreting words or numbers.
    
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
            "Given an instruction-following question and its corresponding answer, misread the instructions such as misinterpreting words or numbers giving in the instruction.\n"
            "Only misread the instructions and introduce an error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/misread-instructions-temp{args.temperature}.jsonl')
    return

def assumptions(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    # TODO: answers are compressed
    """Given an instruction-following question, make an assumption about the instructions and generate a new answer.
    
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
            "Given an instruction-following question, make an assumption about the instructions which is not provided in the question and generate an answer.\n"
            "Only make an assumption about the instructions and introduce an error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/assumptions-temp{args.temperature}.jsonl')
    return


def do_more(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    # TODO: not doing more than what is asked
    """Given an instruction-following question, generate a new answer that does more than what is asked in the question.
    
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
            "Given an instruction-following question, generate a new answer that does more than what is asked in the question.\n"
            "Add references, definitions and other augmenting material which is not essential solve the given question.\n"
            "But make sure that the additional material is relevant to the question.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/do-more-temp{args.temperature}.jsonl')
    return