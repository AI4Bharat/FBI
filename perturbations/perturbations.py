import os
import json
import argparse
import pandas as pd
from ast import literal_eval
from openai import OpenAI

from langchain_core.output_parsers import JsonOutputParser

from perturbations.prompts import Facts, Errors, Stitch

API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=API_KEY)


def parse_args():
    parser = argparse.ArgumentParser(description='Batch processing')
    parser.add_argument('--data_dir', type=str, default='/Users/sumanth/code/fbi/data', help='Path to data directory')
    parser.add_argument('--subset', type=str, choices=['all', 'reasoning', 'factual', 'instruction-following', 'long-form'], help='Subset of testset')
    parser.add_argument('--file_name', type=str, default='v0', help='Version of testset')
    # model arguments
    parser.add_argument('--model', type=str, default='gpt-4-turbo', help='Model to use')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0, help='Top p for sampling')
    parser.add_argument('--top_k', type=int, default=0, help='Top k for sampling')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Max tokens for sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0, help='Frequency penalty for sampling')
    parser.add_argument('--presence_penalty', type=float, default=0, help='Presence penalty for sampling')
    # perturbation arguments
    parser.add_argument('--facts', action='store_true', help='Generate factual statements')
    parser.add_argument('--errors', action='store_true', help='Generate factual statements with errors')
    parser.add_argument('--stitch', action='store_true', help='Stitch factual statements with errors')
    args = parser.parse_args()
    return args


def factual_perturbations(args: argparse.Namespace, testset: pd.DataFrame):
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
        


def main(args):
    testset = pd.read_csv(f'{args.data_dir}/{args.file_name}', sep='\t')
    if args.subset != 'all':
        testset = testset[testset['ability'] == args.subset]

    if args.subset == 'factual':
        factual_perturbations(args, testset)
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