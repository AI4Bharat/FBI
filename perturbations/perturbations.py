import os
import json
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
    args = parser.parse_args()
    return args


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

        dict_ = {
            'custom_id': row['cdx'],
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': f'{args.model}',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': f'{PROMPT}'
                    }
                ],
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'frequency_penalty': args.frequency_penalty,
                'presence_penalty': args.presence_penalty,
            }
        }
        jsons.append(dict_)

    with open(f'{args.data_dir}/{save_name}.jsonl', 'w') as f:
        for json_ in jsons:
            f.write(json.dumps(json_) + '\n')
        

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
        dict_ = {
            'custom_id': row['cdx'],
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': f'{args.model}',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': f'{PROMPT}'
                    }
                ],
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'frequency_penalty': args.frequency_penalty,
                'presence_penalty': args.presence_penalty,
            }
        }
        jsons.append(dict_)
    
    with open(f'{args.data_dir}/direct-errors-temp{args.temperature}.jsonl', 'w') as f:
        for json_ in jsons:
            f.write(json.dumps(json_) + '\n')


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
        dict_ = {
            'custom_id': row['cdx'],
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': f'{args.model}',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': f'{PROMPT}'
                    }
                ],
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'frequency_penalty': args.frequency_penalty,
                'presence_penalty': args.presence_penalty,
            }
        }
        jsons.append(dict_)
    
    with open(f'{args.data_dir}/multiple-direct-errors-temp{args.temperature}.jsonl', 'w') as f:
        for json_ in jsons:
            f.write(json.dumps(json_) + '\n')


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
        dict_ = {
            'custom_id': row['cdx'],
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': f'{args.model}',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': f'{PROMPT}'
                    }
                ],
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'frequency_penalty': args.frequency_penalty,
                'presence_penalty': args.presence_penalty,
            }
        }
        jsons.append(dict_)
    
    with open(f'{args.data_dir}/contextual-errors-temp{args.temperature}.jsonl', 'w') as f:
        for json_ in jsons:
            f.write(json.dumps(json_) + '\n')
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
        dict_ = {
            'custom_id': row['cdx'],
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': f'{args.model}',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': f'{PROMPT}'
                    }
                ],
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'frequency_penalty': args.frequency_penalty,
                'presence_penalty': args.presence_penalty,
            }
        }
        jsons.append(dict_)

    with open(f'{args.data_dir}/number-errors-temp{args.temperature}.jsonl', 'w') as f:
        for json_ in jsons:
            f.write(json.dumps(json_) + '\n')
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
        dict_ = {
            'custom_id': row['cdx'],
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': f'{args.model}',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': f'{PROMPT}'
                    }
                ],
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'frequency_penalty': args.frequency_penalty,
                'presence_penalty': args.presence_penalty,
            }
        }
        jsons.append(dict_)

    with open(f'{args.data_dir}/entity-errors-temp{args.temperature}.jsonl', 'w') as f:
        for json_ in jsons:
            f.write(json.dumps(json_) + '\n')
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
        dict_ = {
            'custom_id': row['cdx'],
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': f'{args.model}',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': f'{PROMPT}'
                    }
                ],
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'frequency_penalty': args.frequency_penalty,
                'presence_penalty': args.presence_penalty,
            }
        }
        jsons.append(dict_)
    
    with open(f'{args.data_dir}/incorrect-fact-temp{args.temperature}.jsonl', 'w') as f:
        for json_ in jsons:
            f.write(json.dumps(json_) + '\n')
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
        dict_ = {
            'custom_id': row['cdx'],
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': f'{args.model}',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': f'{PROMPT}'
                    }
                ],
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'frequency_penalty': args.frequency_penalty,
                'presence_penalty': args.presence_penalty,
            }
        }
        jsons.append(dict_)
    
    with open(f'{args.data_dir}/opposite-fact-temp{args.temperature}.jsonl', 'w') as f:
        for json_ in jsons:
            f.write(json.dumps(json_) + '\n')
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
        dict_ = {
            'custom_id': row['cdx'],
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': f'{args.model}',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': f'{PROMPT}'
                    }
                ],
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'frequency_penalty': args.frequency_penalty,
                'presence_penalty': args.presence_penalty,
            }
        }
        jsons.append(dict_)
    
    with open(f'{args.data_dir}/remove-fact-temp{args.temperature}.jsonl', 'w') as f:
        for json_ in jsons:
            f.write(json.dumps(json_) + '\n')
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
        pass
    else:
        raise ValueError('Invalid subset')


if __name__ == '__main__':
    args = parse_args()
    main(args)