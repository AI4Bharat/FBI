import json
import argparse
from langchain_core.output_parsers import JsonOutputParser
from llm_evaluators.parsers import *



def read_jsonl(file_name):
    results_data = []
    with open(file_name) as f:
        for line in f:
            results_data.append(json.loads(line))
    return results_data


def analyze_single_vanilla_batch_result(data):
    
    #processing the batch results
    parser = JsonOutputParser(pydantic_object=SingleVanillaScore)
    results = dict()
    errors = 0
    for item in data:
        id = item['custom_id']
        item_id = id.split("~")[0]
        res_type = id.split("~")[-1]
        try:
            if item_id in results:
                results[item_id][res_type] = parser.invoke(item['response']['body']['choices'][0]['message']['content'])['score']
                
            else:
                results[item_id] = {res_type: parser.invoke(item['response']['body']['choices'][0]['message']['content'])['score']}
        except Exception as e:
            print(e)
            errors += 1
            
    changed = 0
    changed_ids = []
    unprocessed_ids = []
    for id, res in results.items():
        try:
            if res['pert'] < res['orig']:
                changed += 1
                changed_ids.append(id)
        except Exception as e:
            print(e)
            unprocessed_ids.append(id)
            
    return results, changed, changed_ids, unprocessed_ids, errors
            
        
def analyze_single_axes_rubrics_batch_result(data):
    pass


def analyze_compare_vanilla_batch_result(data):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze LLM results')
    parser.add_argument("--file_name", type=str, help="File name of the data")
    parser.add_argument("--type", type=str, choices=['single_vanilla', 'single_axes_rubrics', 'compare_vanilla'], help="Type of the data")
    return parser.parse_args()

def main(args):
    if args.type == 'single_vanilla':
        data = read_jsonl(args.file_name)
        results, changed, changed_ids, unprocessed_ids, errors = analyze_single_vanilla_batch_result(data)
        print(f"Total number of results: {len(data)/2}")
        print(f"Total number of changed results: {changed}")
        print(f"Total number of unchanged results: {len(data)/2 - changed}")
        print(f"Changed ids: {changed_ids}")
        print(f"Unprocessed ids: {unprocessed_ids}")
        print(f"Errors: {errors}")
    else:
        print("Still pending")
        
        
if __name__ == "__main__":
    args = parse_args()
    main(args)