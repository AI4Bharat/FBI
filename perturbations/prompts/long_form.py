import argparse
import pandas as pd
from ast import literal_eval

from langchain_core.output_parsers import JsonOutputParser

from perturbations.utils import create_jsonl, dump_jsonl
from perturbations.parsers import Facts, Errors, Stitch, DirectError, MultipleDirectErrors


def grammar_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce grammatical errors.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
        
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT =  (
            "Given the following Gold Answer, introduce a grammatical error in the answer.\n"
            "Here are a list of possible errors you can introduce:\n"
            "1. Subject-Verb Agreement Errors\n"
            "2. Pronoun-Antecedent Agreement Errors\n"
            "3. Misplaced Modifiers\n"
            "4. Dangling Modifiers\n"
            "5. Run-On Sentences\n"
            "6. Comma Splices\n"
            "7. Incorrect Use of Apostrophes\n"
            "8. Confusion Between 'Its' and 'It's'\n"
            "9. Incorrect Verb Tense\n"
            "10. Fragmented Sentences\n"
            "11. Double Negatives\n"
            "12. Incorrect Word Usage\n"
            "13. Confusion Between 'Your' and 'You're'\n"
            "14. Confusion Between 'Then' and 'Than'\n"
            "15. Preposition at the End of a Sentence\n"
            "16. Incorrect Pluralization\n\n"
            "Pick one of the errors from the above list and introduce it in the answer.\n"
            "And make this error consistent throughout the answer.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/grammar-errors-temp{args.temperature}.jsonl')
    return


def spelling_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce spelling errors.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset

    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Given the following Gold Answer, introduce a spelling error in the answer.\n"
            "Make sure that the introduced error is not immediately obvious and requires some level of knowledge to identify.\n"
            "Please make sure that the introduced error is consistent throughout the answer.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/spelling-errors-temp{args.temperature}.jsonl')
    return


def chronological_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce chronological errors.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
        
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Given the following Gold Answer, introduce chronological errors in the answer.\n"
            "Mix up the sequence of events, use incorrect time references, switch between past, present, and future tenses inappropriately, and introduce conflicting timelines to create inconsistencies in the order of actions.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/chronological-errors-temp{args.temperature}.jsonl')
    return


def consistency_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce inconsistencies in the answer.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        # removed: conflicting information, contradictory facts, and
        PROMPT = (
            "Given the following Gold Answer, introduce inconsistencies in the answer.\n"
            "Introduce logical inconsistencies to create errors in the answer.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/consistency-errors-temp{args.temperature}.jsonl')
    return


def coherence_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce coherence errors.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        # todo: blend the errors into the answer
        PROMPT = (
            "Given the following Gold Answer, introduce coherence errors in the answer.\n"
            "Introduce disjointed ideas, disconnected thoughts, and fragmented information to create errors in the answer.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/coherence-errors-temp{args.temperature}.jsonl')
    return


def formatting_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, remove the formatting in the answers. This is a score invariant perturbation.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Given the following Gold Answer, remove the formatting in the answer.\n"
            "Except newline characters remove any bold, italic, underline, or any other formatting in the answer.\n"
            "Do not change any logical or factual information in the answer. Only change the formatting.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/formatting-errors-temp{args.temperature}.jsonl')
    return


def comprehensiveness_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a Gold Answer, introduce errors that make the answer less comprehensive.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        # TODO: errors are too short answers
        PROMPT = (
            "Given the following Question and Gold Answer, introduce one of the below errors.\n"
            "1. Lack of Detail\n"
            "2. Vagueness\n"
            "3. Irrelevance\n"
            "4. Incomplete Information\n"
            "5. Poor Structure\n"
            "6. Lack of Examples\n"
            "7. Inaccuracies\n"
            "8. Lack of Context\n"
            "9. Unclear Language\n"
            "10. Absence of Concluding Remarks\n"
            "11. Failure to Address All Parts of the Question\n\n"
            "Pick one of the errors from the above list and introduce it in the answer.\n"
            "Do not drastically change the answer, only introduce an error that makes the answer less comprehensive with respect to the given question.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Question:\n"
            f"{row['question']}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/comprehensiveness-errors-temp{args.temperature}.jsonl')
    return


def superficial_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a Gold Answer, introduce superficial cues to make it look like LLM generated answers.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
    
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=DirectError)
        PROMPT = (
            "Given the following Gold Answer, introduce a factual error in the answers and also introduce superficial cues to make it look like a Language Model generated answer.\n"
            "These could include any of the below errors that do not add any value to the answer.\n"
            "1. Redundancy\n"
            "2. Filler Words and Phrases\n"
            "3. Overly Long Sentences\n"
            "4. Unrelated Information\n"
            "5. Excessive Qualifiers\n"
            "6. Repetitive Phrasing\n"
            "7. Verbose Introductions and Conclusions\n"
            "8. Unnecessary Examples\n"
            "9. Generalizations\n"
            "10. Abstract Language\n"
            "11. Parenthetical Statements\n"
            "12. Unnecessary Citations or References\n\n"
            "Pick one of the errors from the above list and introduce it in the answer.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/superficial-errors-temp{args.temperature}.jsonl')
    return