import json
import argparse
import re
import ast
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from llm_evaluators.parsers import *



def read_jsonl(file_name):
    results_data = []
    with open(file_name) as f:
        for line in f:
            results_data.append(json.loads(line))
    return results_data

def custom_json_parser(data):
    
    #first check for triple ticks pattern
    triple_tick_pattern = r'```(.*?)```'
    matches = re.findall(triple_tick_pattern, data, re.DOTALL)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError as e:
                continue
            
    #if no valid match, attempt to find and parse json in whole string
    json_like_pattern = r'\{.*?\}'
    matches = re.findall(json_like_pattern, data, re.DOTALL)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError as e:
                #final attempt using ast.literal_eval as fallback
                try:
                    return ast.literal_eval(match)
                except Exception as e:
                    continue
                
    raise Exception("No valid json could be extracted")
    

def parse_and_get_score(data):
    parser = JsonOutputParser(pydantic_object=SingleVanillaScore)
    #try by default using the langchain parser
    try:
        return parser.invoke(data)['score']
    except OutputParserException as e:
        #if fails, try custom parser
        return custom_json_parser(data)['score']
    except Exception as e:
        raise e
        

def analyze_single_vanilla_batch_result(data):
    
    #processing the batch results
    results = dict()
    errors = 0
    for item in data:
        id = item['custom_id']
        item_id = id.split("~")[0]
        res_type = id.split("~")[-1]
        
        try:
            score = parse_and_get_score(item['response']['body']['choices'][0]['message']['content'])
            if item_id in results:
                results[item_id][res_type] = score
            else:
                results[item_id] = {res_type: score}
        except Exception as e:
            print(e)
            score = None
            errors += 1
            
    changed = 0
    changed_ids = []
    unprocessed_ids = []
    for id, res in results.items():
        try:
            if int(res['pert']) < int(res['orig']):
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
        print(f"Total number of unchanged results: {len(data)/2 - errors/2 - changed}")
        print(f"Changed ids: {changed_ids}")
        print(f"Unprocessed ids: {unprocessed_ids}")
        print(f"Errors: {errors}")
    else:
        print("Still pending")
        
        
if __name__ == "__main__":
    args = parse_args()
    main(args)