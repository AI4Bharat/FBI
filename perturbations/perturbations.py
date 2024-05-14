import os
import json
import argparse
import pandas as pd
from openai import OpenAI

from langchain_core.output_parsers import JsonOutputParser

from perturbations.prompts import Facts

API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=API_KEY)


def parse_args():
    parser = argparse.ArgumentParser(description='Batch processing')
    parser.add_argument('--data_dir', type=str, default='/Users/sumanth/code/fbi/data', help='Path to data directory')
    parser.add_argument('--subset', type=str, choices=['all', 'reasoning', 'factual', 'instruction-following', 'long-form'], help='Subset of testset')
    parser.add_argument('--testset_version', type=str, default='v0', help='Version of testset')
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
        answer = row["answer"]
        
        # Set up a parser + inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=Facts)

        if args.facts:
            PROMPT = (
                "Given the following passage, generate factual statements.\n\n"
                f"{parser.get_format_instructions()}\n\n"
                "Passage:\n\n"
                f"{answer}\n"
            )
            save_name = 'get-facts'
        elif args.errors:
            save_name = 'add-errors'
            pass
        elif args.stitch:
            save_name = 'stitch-facts'
            pass
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
    testset = pd.read_csv(f'{args.data_dir}/testset-{args.testset_version}-answers.tsv', sep='\t')
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