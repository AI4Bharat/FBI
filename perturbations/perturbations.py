import os
import json
import random
import argparse
import pandas as pd
from ast import literal_eval

from langchain_core.output_parsers import JsonOutputParser

from perturbations.prompts import Facts, Errors, Stitch, DirectError, MultipleDirectErrors


def parse_args():
    parser = argparse.ArgumentParser(description='Batch processing')
    parser.add_argument('--data_dir', type=str, default='/Users/sumanth/code/fbi/data', help='Path to data directory')
    parser.add_argument('--subset', type=str, choices=['all', 'reasoning', 'factual', 'instruction-following', 'long-form'], help='Subset of testset')
    parser.add_argument('--file_name', type=str, default='testset-v0-answers.tsv', help='name of the input file')
    # model arguments
    parser.add_argument('--model', type=str, default='gpt-4-turbo', help='Model to use')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=1, help='Top p for sampling')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Max tokens for sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0, help='Frequency penalty for sampling')
    parser.add_argument('--presence_penalty', type=float, default=0, help='Presence penalty for sampling')
    # perturbation arguments
    parser.add_argument('--facts', action='store_true', help='Generate factual statements')
    parser.add_argument('--errors', action='store_true', help='Generate factual statements with errors')
    parser.add_argument('--stitch', action='store_true', help='Stitch factual statements with errors')
    # debug
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    return args


def create_jsonl(cdx: str, model_name: str, prompt: str, max_tokens: int, temperature: float, top_p: float, frequency_penalty: float, presence_penalty: float) -> dict:
    return {
        'custom_id': cdx,
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': {
            'model': f'{model_name}',
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.'
                },
                {
                    'role': 'user',
                    'content': f'{prompt}'
                }
            ],
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty,
        }
    }


def dump_jsonl(args: argparse.Namespace, jsons: list, file_name: str) -> None:
    if args.debug:
        jsons = random.sample(jsons, 20)
    
    with open(file_name, 'w') as f:
        for json_ in jsons:
            f.write(json.dumps(json_) + '\n')


def factual_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Generate factual statements, introduce errors in factual statements, and stitch factual statements with errors.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset

    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():     

        if args.facts:
            parser = JsonOutputParser(pydantic_object=Facts)
            PROMPT = (
                "Given the following passage, generate factual statements. Provide a bullet-point list of the key factual statements in the answer.\n\n"
                f"{parser.get_format_instructions()}\n\n"
                "Passage:\n"
                f"{row['answer']}\n"
            )
            save_name = 'get-facts'
        elif args.errors:
            parser = JsonOutputParser(pydantic_object=Errors)
            facts = '\n'.join(literal_eval(row['json_parsed'])['facts'])
            PROMPT = (
                "Given the following list of factual statements, introduce an error in one of the statements and provide an explanation of the introduced error.\n"
                "Make sure the error is subtle and not immediately obvious.\n"
                "If you are introducing errors in numerical data, ensure that the error is within a reasonable range.\n\n"
                f"{parser.get_format_instructions()}\n\n"
                "Factual Statements:\n"
                f"{facts}"
            )
            save_name = 'add-errors'
        elif args.stitch:
            parser = JsonOutputParser(pydantic_object=Stitch)
            errors = '\n'.join(literal_eval(row['error'])['errors'])
            PROMPT = (
                "Given the following Question, Gold Answer and a list of factual statements with errors, rewrite the gold answer with the introduced errors.\n"
                "Only change the sentence in the Gold Answer that corresponds to the error in the factual statements.\n"
                "Keep the rest of the Gold Answer the same.\n"
                "DO NOT add any highlighted text or any other formatting to the introduced errors.\n\n"
                f"{parser.get_format_instructions()}\n\n"
                "Question:\n"
                f"{row['question']}\n\n"
                "Gold Answer:\n"
                f"{row['answer']}\n\n"
                "Factual Statements with Errors:\n"
                f"{errors}"
            )
            save_name = 'stitch-facts'
        else:
            raise ValueError('Invalid perturbation')

        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/{save_name}.jsonl')
    return
        

def factual_perturbations_v2(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given the gold answers, directly add errors to the answers.
    
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
            "Given the following Gold Answer, introduce an error in the answer and provide an explanation of the introduced error.\n"
            "Make sure the error is not immediately obvious and requires some level of knowledge to identify.\n"
            "If you are introducing errors in numerical data, ensure that the error is within a reasonable range.\n"
            "Do not add any highlighted text or any other formatting to the introduced errors.\n"
            "Only change the sentence in the Gold Answer that corresponds to the error and keep the rest of the Gold Answer the same.\n\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/direct-errors-temp{args.temperature}.jsonl')
    return


def factual_perturbations_v3(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given the gold answer, add multiple errors into the answer.
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset

    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=MultipleDirectErrors)
        PROMPT = (
            "Given the following Gold Answer, introduce errors in the answer and provide an explanation of the introduced errors.\n"
            "Make sure the errors are not immediately obvious and require some level of topical knowledge to identify.\n"
            "If you are introducing errors in numerical data, ensure that the errors are within a reasonable range.\n"
            "Do not add any highlighted text or any other formatting to the introduced errors.\n"
            "Only change the sentences in the Gold Answer that corresponds to the errors and keep the rest of the Gold Answer the same.\n\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced errors.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/multiple-direct-errors-temp{args.temperature}.jsonl')
    return


def contextual_fact_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, we want to replace contextual facts with errors.
    
    Examples would include replacing photon with quanta, electric with magnetic, oxygen with nitrogen, blue with green, etc.

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
            "Given the following Question and the Gold Answer, introduce an error in the answer by replacing a contextual fact with an error.\n"
            "Examples of these would include replacing scientific terms like photon with quanta, electric with magnetic, kinetic with potential, oxygen with nitrogen, or even replacing colours etc.\n"
            "These could also be replacing anything that is contextually relevant fact to the answer.\n"
            "Please make sure that the introduced error is consistent throughout the answer.\n"
            "Make sure that the error is not immediately obvious and requires some level of knowledge to identify.\n"
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
    
    dump_jsonl(args, jsons, f'{args.data_dir}/contextual-errors-temp{args.temperature}.jsonl')
    return


def number_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce errors in numerical data.
    
    Examples would include changing numerical values, changing units, changing percentages, changing dates etc.

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
            "Given the following Gold Answer, introduce an error in the answer by changing numerical data.\n"
            "Examples of these would include changing numerical values, changing units, changing percentages, changing dates etc.\n"
            "Make sure that the introduced error is within a reasonable range and is not immediately obvious.\n"
            "Please make sure that the introduced error is consistent throughout the answer.\n"
            "Provide an explanation of the introduced error.\n"
            f"{parser.get_format_instructions()}\n\n"
            "Gold Answer:\n"
            f"{row['answer']}\n\n"
            "Rewrite the full Gold Answer with the introduced error.\n"
        )
        dict_ = create_jsonl(row['cdx'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)

    dump_jsonl(args, jsons, f'{args.data_dir}/number-errors-temp{args.temperature}.jsonl')
    return


def entity_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce errors in named entities.
    
    Examples would include changing names of people, places, organizations, etc.
    
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
            "Given the following Gold Answer, introduce an error in the answer by changing named entities.\n"
            "Examples of these would include changing names of people, places, organizations, etc.\n"
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

    dump_jsonl(args, jsons, f'{args.data_dir}/entity-errors-temp{args.temperature}.jsonl')
    return


def add_incorrect_fact(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce an incorrect fact.
    
    Examples could include introducing a fact that is incorrect or misleading that is relevant to the context of the answer.

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
            "Given the following Gold Answer, introduce an incorrect fact in the answer.\n"
            "Examples of these would include introducing a fact that is incorrect or misleading that is relevant to the context of the answer.\n"
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
    
    dump_jsonl(args, jsons, f'{args.data_dir}/incorrect-fact-temp{args.temperature}.jsonl')
    return


def opposite_perturbations(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, introduce a negation or an opposite fact.
    
    Examples could include introducing a negation or an opposite fact that is relevant to the context of the answer.
    
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
            "Given the following Gold Answer, introduce a negation or an opposite fact in the answer.\n"
            "Examples of these would include introducing a negation or an opposite fact that is relevant to the context of the answer.\n"
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
    
    dump_jsonl(args, jsons, f'{args.data_dir}/opposite-fact-temp{args.temperature}.jsonl')
    return


def remove_fact(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a gold answer, remove a fact from the answer.
    
    Examples could include removing a fact that is relevant and important to the context of the answer.
    
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
            "Given the following Gold Answer, remove a fact from the answer.\n"
            "Examples of these would include removing a fact that is relevant and important to the context of the answer.\n"
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
    
    dump_jsonl(args, jsons, f'{args.data_dir}/remove-fact-temp{args.temperature}.jsonl')
    return


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


def main(args):
    testset = pd.read_csv(f'{args.data_dir}/{args.file_name}', sep='\t')
    if args.subset != 'all':
        testset = testset[testset['ability'] == args.subset]

    if args.subset == 'factual':
        # factual_perturbations(args, testset)
        # factual_perturbations_v2(args, testset)
        # factual_perturbations_v3(args, testset)
        contextual_fact_perturbations(args, testset)
        number_perturbations(args, testset)
        entity_perturbations(args, testset)
        add_incorrect_fact(args, testset)
        opposite_perturbations(args, testset)
        remove_fact(args, testset)
    elif args.subset == 'reasoning':
        pass
    elif args.subset == 'instruction-following':
        pass
    elif args.subset == 'long-form':
        # grammar_perturbations(args, testset)
        # spelling_perturbations(args, testset)
        # chronological_perturbations(args, testset)
        # consistency_perturbations(args, testset)
        # coherence_perturbations(args, testset)
        # formatting_perturbations(args, testset)
        # comprehensiveness_perturbations(args, testset)
        superficial_perturbations(args, testset)
    else:
        raise ValueError('Invalid subset')


if __name__ == '__main__':
    args = parse_args()
    main(args)