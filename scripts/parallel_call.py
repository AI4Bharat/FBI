import os
import json
import argparse
from turtle import back
import backoff
import logging
from openai import OpenAI, RateLimitError
from joblib import Parallel, delayed
import openai
from config import *

API_KEY = OPENAI_API_KEY


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def read_jsonl(file_name):
    results_data = []
    with open(file_name) as f:
        for line in f:
            results_data.append(json.loads(line))
    return results_data

def write_jsonl(file_name, responses):
    with open(file_name, 'w') as f:
        for response in responses:
            f.write(json.dumps(response) + '\n')

def format_result_dict(res, custom_id):
    result_dict = {
            'id' : f"parallel_req_{res['id']}",
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": None,
                "body":{
                    "id": res['id'],
                    "object": "chat.completion",
                    "created": res['created'],
                    "model": res['model'],
                    "choices": res['choices'],
                    "usage": res['usage'],
                    "system_fingerprint": res['system_fingerprint']
                }
            },
            "error": None
        }
    return result_dict


def call_openai(data_dict):
    openai_client = OpenAI(api_key = API_KEY)
    
    model = data_dict['body']['model']
    max_tokens = data_dict['body']['max_tokens']
    temperature = data_dict['body']['temperature']
    custom_id = data_dict['custom_id']
    messages = data_dict['body']['messages']
    
    try:
        res = openai_client.chat.completions.create(
            model = model, 
            messages = messages,
            max_tokens = max_tokens, 
            temperature = temperature
        )
        return_res = format_result_dict(res.model_dump(), custom_id)
        return return_res
    except RateLimitError as e:
        raise
    except Exception as e:
        print(e)
        logger.error(f"Error processing request for custom_id={custom_id}: {str(e)}")
        return {'error': str(e), 'custom_id': custom_id}
        
        
def call_llama3(data_dict):
    openai_client = OpenAI(
        base_url = LLAMA3_BASE_URL,
        api_key = LLAMA3_API_KEY
    )
    
    model = data_dict['body']['model']
    if model == 'llama3-70b':
        model = "meta-llama/Meta-Llama-3-70B-Instruct"
    max_tokens = data_dict['body']['max_tokens']
    temperature = data_dict['body']['temperature']
    custom_id = data_dict['custom_id']
    messages = data_dict['body']['messages']
    
    try:
        res = openai_client.chat.completions.create(
            model = model, 
            messages = messages,
            max_tokens = max_tokens, 
            temperature = temperature
        )
        return_res = format_result_dict(res.model_dump(), custom_id)
        return return_res
    except RateLimitError as e:
        raise
    except Exception as e:
        print(e)
        logger.error(f"Error processing request for custom_id={custom_id}: {str(e)}")
        return {'error': str(e), 'custom_id': custom_id}
    
    

@backoff.on_exception(backoff.expo, RateLimitError)
def backoff_openai_call(data_dict):
    return call_openai(data_dict)
    
@backoff.on_exception(backoff.expo, RateLimitError)
def backoff_llama3_call(data_dict):
    return call_llama3(data_dict)
        
    
    

def parse_args():
    parser = argparse.ArgumentParser(description = 'parallel processing')
    parser.add_argument("--input_file_name", type=str, help = "Input file name")
    parser.add_argument("--output_file_name", type=str, help = "Output file name")
    parser.add_argument("--n_jobs", type=int, help = "Number of parallel jobs to run")
    args = parser.parse_args()
    return args

def main(args):
    data = read_jsonl(args.input_file_name)
    
    #check the model type
    if data[0]['body']['model'] in ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo-0125', 'gpt-4o']:
        results = Parallel(n_jobs = args.n_jobs)(delayed(backoff_openai_call)(data_dict) for data_dict in data)
        write_jsonl(args.output_file_name, results)
    elif data[0]['body']['model'] in ['llama3-70b']:
        results = Parallel(n_jobs = args.n_jobs)(delayed(backoff_llama3_call)(data_dict) for data_dict in data)
        write_jsonl(args.output_file_name, results)
    else:
        print("Still pending")

if __name__ == '__main__':
    args = parse_args()
    main(args)