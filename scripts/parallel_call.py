import os
import json
import argparse
import backoff
import logging
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from joblib import Parallel, delayed
from anthropic import Anthropic
from anthropic import RateLimitError as AnthropicRateLimitError
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
            
def format_anthropic_messages(messages):
    formatted_messages = []
    for message in messages:
        if message['role'] == 'system':
            system_prompt = message['content']
            continue
        else:
            formatted_message = {
                'role': message['role'],
                'content': [
                    {
                        "type": "text",
                        "text": message['content']
                    }
                ]
            }
            formatted_messages.append(formatted_message)
    return system_prompt, formatted_messages

def format_openai_result_dict(res, custom_id):
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

def format_anthropic_result_dict(res, custom_id):
    result_dict = {
            'id' : f"parallel_req_{res['id']}",
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": None,
                "body":{
                    "id": res['id'],
                    "object": "chat.completion",
                    "created": None,
                    "model": res['model'],
                    "choices": [{
                        'finish_reason': res['stop_reason'],
                        'index': 0,
                        'logprobs': None,
                        'message': {
                            'content': res['content'][0]['text'],
                            'role': res['role'],
                            'function_call': None,
                            'tools_call': None,
                            'name': None
                        }
                    }],
                    "usage": res['usage'],
                    "system_fingerprint": None
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
        return_res = format_openai_result_dict(res.model_dump(), custom_id)
        return return_res
    except OpenAIRateLimitError as e:
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
        return_res = format_openai_result_dict(res.model_dump(), custom_id)
        return return_res
    except OpenAIRateLimitError as e:
        raise
    except Exception as e:
        print(e)
        logger.error(f"Error processing request for custom_id={custom_id}: {str(e)}")
        return {'error': str(e), 'custom_id': custom_id}
    

def call_claude(data_dict):
    anthropic_client = Anthropic(
        api_key = CLAUDE_API_KEY
    )
    
    model = data_dict['body']['model']
    if model == 'claude3-opus':
        model = 'claude-3-opus-20240229'
    max_tokens = data_dict['body']['max_tokens']
    temperature = data_dict['body']['temperature']
    custom_id = data_dict['custom_id']
    system_prompt, messages = format_anthropic_messages(data_dict['body']['messages'])
    
    try:
        res = anthropic_client.messages.create(
            model = model,
            max_tokens = max_tokens,
            temperature = temperature,
            system = system_prompt,
            messages = messages
        )
        return_res = format_anthropic_result_dict(res.to_dict(), custom_id)
        return return_res
    except AnthropicRateLimitError as e:
        raise
    except Exception as e:
        print(e)
        logger.error(f"Error processing request for custom_id={custom_id}: {str(e)}")
        return {'error': str(e), 'custom_id': custom_id}


@backoff.on_exception(backoff.expo, OpenAIRateLimitError)
def backoff_openai_call(data_dict):
    return call_openai(data_dict)
    
@backoff.on_exception(backoff.expo, OpenAIRateLimitError)
def backoff_llama3_call(data_dict):
    return call_llama3(data_dict)

@backoff.on_exception(backoff.expo, AnthropicRateLimitError)
def backoff_claude_call(data_dict):
    return call_claude(data_dict)

        
    
    

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
    elif data[0]['body']['model'] in ['claude3-opus']:
        results = Parallel(n_jobs = args.n_jobs)(delayed(backoff_claude_call)(data_dict) for data_dict in data)
        write_jsonl(args.output_file_name, results)
    else:
        print("Still pending")

if __name__ == '__main__':
    args = parse_args()
    main(args)