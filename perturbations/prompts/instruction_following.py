import argparse
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser

from perturbations.utils import create_jsonl, dump_jsonl
from perturbations.parsers import DirectError


def incorrect_sequence(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    # TODO: this is still changing the steps in the answer, not the question. similar to coherence_perturbations
    """Given a instruction-following question and it's corresponding answer, rewrite the answer in the wrong sequence of execution.
    
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
            "Given a question and its corresponding answer, alter the sequence of steps in the question to introduce errors in the answer.\n"
            "Here are some things to keep in mind when generating the answer:\n"
            "1. Identify the logical steps in the given question.\n"
            "2. Change the order of these steps to create a new sequence.\n"
            "3. Rewrite the provided answer, reflecting the errors that result from this new sequence.\n"
            "4. Ensure that the introduced errors are a direct consequence of the changed order of execution.\n"
            "5. Maintain the original context and terminology used in the question and answer.\n\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with errors introduced by changing the order of execution of the question.\n"
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
            "Given an instruction-following question and its corresponding answer, omit one step in the answer to introduce an error.\n"
            "Here are the instructions to follow:\n"
            "1. Review the provided question and answer carefully.\n"
            "2. Identify a single step in the answer.\n"
            "3. Omit this step to introduce an error.\n"
            "4. Ensure that the error directly affects the accuracy or completeness of the answer.\n"
            "5. Preserve the original context and terminology used in the question and answer.\n\n"
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
            "Given an instruction-following question and its corresponding answer, ensure that the instructions are only partially executed to introduce an error.\n"
            "Here are the instructions to follow:\n"
            "1. Carefully review the provided question and answer.\n"
            "2. Identify the steps in the answer that correspond to the instructions in the question.\n"
            "3. Partially execute one or more instructions.\n"
            "4. Ensure that the omission introduces a clear error.\n"
            "5. Maintain the original context and terminology of the question and answer.\n\n"
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

def incomplete_execution_v2(args: argparse.Namespace, testset: pd.DataFrame) -> None:
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
            "I want to create a few questions to test how good my evaluations system is. For this, I have collected some questions along with their gold answers."
            "For testing my evaluators, I want to make some subtle errors in the responses that are not easily detectable by the evaluators.\n"
            "Given a question and a solution (gold answer), I want you to help me in creating such errored answers based on the 'Omission of Requested Details' perturbation. Your task is to perturb the gold answer such that deliberately omits specific details or steps that were explicitly requested in the original question. Ensure that the perturbed response, while being same as gold answer, lacks the completeness requested, making it subtly incorrect in the context of the question's requirements.\n"
            "For example:\n"
            "1. If the question asks for 3 steps, you can give only 2 steps in the answer.\n"
            "2. If the question asks for a specific detail, you can omit that detail in the answer.\n"
            "3. If the question asks for list of 5 items, you can give only 4 items in the answer.\n"
            "4. If the question asks for a specific order, you can give the items in a different order in the answer.\n"
            "The perturbed answer must be same as gold answer but with the added perturbation (similar to the ones mentioned above) to make the answer incorrect with respect to the question.\n\n"
            "You must only remove something that is specifically asked in the question. Do not add any new information or remove some information from the answer which is not explicitly asked in the question.\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the Gold Answer with the introduced error and give an explanation for the error made.\n"
            f"{parser.get_format_instructions()}\n\n"
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

def do_more_v2(args: argparse.Namespace, testset: pd.DataFrame) -> None:
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
            "I am developing a system to evaluate the quality of answers and need to test its ability to detect subtle errors in responses. "
            "For this, I have prepared a list of questions along with their ideal answers (gold answers). "
            "To challenge my evaluators, I want to introduce errors in the responses by adding unrequested information that is not directly relevant or requested by the question.\n"
            "Given a question and a gold answer, your task is to modify the gold answer by intentionally inserting additional, unnecessary details or information. "
            "The added information should make the response less concise and potentially misleading without being overtly incorrect on factual grounds.\n"
            "For example:\n"
            "1. If the question asks for the causes of the French Revolution, you can add irrelevant details about the architecture of that period.\n"
            "2. If the question asks to describe the process of photosynthesis, you can include an unnecessary explanation of plant anatomy unrelated to the process.\n"
            "3. If the question asks for a brief history of a company, include extraneous details about the industry that are not directly relevant.\n"
            "4. If the question asks for a solution to a math problem, add a discussion on the importance of learning mathematics.\n"
            "The modified answer should still contain the correct information from the gold answer but with the addition of unrelated or unrequested information.\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error and provide an explanation for the error made.\n"
            f"{parser.get_format_instructions()}\n\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/do-more-temp{args.temperature}.jsonl')
    return

