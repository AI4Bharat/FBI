import os
import json
import argparse
import pandas as pd
from glob import glob

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from perturbations.utils import dump_jsonl
from perturbations.parsers import DirectError, CustomJsonOutputParser, CompareVanillaCOTScore, SingleVanillaCOTScore
from perturbations.prompts.explanations import analyze_explanations


import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze explanations from predictions')
    parser.add_argument('--file_name', type=str, help='File name of jsonl')
    parser.add_argument('--data_dir', type=str, default='/Users/sumanth/code/fbi/outputs', help='Path to model outputs directory')
    parser.add_argument('--eval_type', type=str, help='eval_strategy')
    parser.add_argument('--job', type=str, default='create_jsonl', help='Job type')
    # model arguments
    parser.add_argument('--model', type=str, default='gpt-4-turbo', help='Model to use')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=1, help='Top p for sampling')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Max tokens for sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0, help='Frequency penalty for sampling')
    parser.add_argument('--presence_penalty', type=float, default=0, help='Presence penalty for sampling')
    # debug
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    return args


def parse_and_get_score(data):
    parser = JsonOutputParser(pydantic_object=SingleVanillaCOTScore)
    #try by default using the langchain parser
    try:
        return parser.invoke(data)['justification'], parser.invoke(data)['score']
    except OutputParserException:
        #if fails, try custom parser
        custom_json_parser = CustomJsonOutputParser(pydantic_model=SingleVanillaCOTScore)
        return custom_json_parser.invoke(data)['justification'], custom_json_parser.invoke(data)['score']
    except Exception as e:
        raise e


def main(args):
    categories = ['factual', 'if', 'lf', 'reasoning']
    if args.job == 'create_jsonl':
        
        if 'single' in args.eval_type:
            default_parser = JsonOutputParser(pydantic_object=SingleVanillaCOTScore)
            json_parser = CustomJsonOutputParser(pydantic_model=SingleVanillaCOTScore)
            print("Single")
        elif 'compare' in args.eval_type:
            default_parser = JsonOutputParser(pydantic_object=CompareVanillaCOTScore)
            json_parser = CustomJsonOutputParser(pydantic_model=CompareVanillaCOTScore)
        
        for category in categories:
            jsonls = glob(f'{args.data_dir}/{args.eval_type}/{category}/*.jsonl')
            for jsonl in jsonls:
                with open(jsonl, 'r') as f:
                    data = [json.loads(line) for line in f]

                df = pd.DataFrame(data)
                df['category'] = df['custom_id'].apply(lambda x: x.split('~')[-1])
                df['qid'] = df['custom_id'].apply(lambda x: x.split('~')[0])

                evals = []
                groups = df.groupby('qid')
                count = 0
                for name, group in groups:
                    original = group[group['category'] == 'orig'].iloc[0]['response']['body']['choices'][0]['message']['content']
                    perturbed = group[group['category'] == 'pert'].iloc[0]['response']['body']['choices'][0]['message']['content']

                    try:
                        orig_explanation, orig_score = parse_and_get_score(original)
                        pert_explanation, pert_score = parse_and_get_score(perturbed)

                        dict_ = {
                            'qid': name,
                            'orig_explanation': orig_explanation,
                            'orig_score': orig_score,
                            'pert_explanation': pert_explanation,
                            'pert_score': pert_score
                        }
                        if orig_score == pert_score:
                            evals.append(dict_)
                            count += 1

                    except Exception as e:
                        print("Error parsing explanation")
                        print(e)

                eval_df = pd.DataFrame(evals)
                jsons = analyze_explanations(args, eval_df)

                new_path = jsonl.replace('output_objects', 'batch_call')
                dir = os.path.dirname(new_path)
                if not os.path.exists(dir):
                    os.makedirs(dir)

                dump_jsonl(args, jsons, new_path)

    elif args.job == 'check_results':
        json_parser = JsonOutputParser(pydantic_object=DirectError)
        for category in categories:
            print(f"-------------------{category}-------------------")
            jsonls = glob(f'{args.data_dir}/{args.eval_type}/{category}/*.jsonl')
            for jsonl in jsonls:
                with open(jsonl, 'r') as f:
                    data = [json.loads(line) for line in f]

                df = pd.DataFrame(data)
                df['predictions'] = df['response'].apply(lambda x: json_parser.parse(x['body']['choices'][0]['message']['content']))

                df['pred_score'] = df['predictions'].apply(lambda x: x['score'])
                df['pred_explanation'] = df['predictions'].apply(lambda x: x['explanation'])

                print(f"Number of time explanation identified error in {jsonl.split('/')[-1]}: {len(df[df['pred_score'] == True])}/{len(df)}")


if __name__ == '__main__':
    args = parse_args()
    main(args)