import os
import json
import argparse
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from openai import OpenAI
from prompts.compare_vanilla import *
from parsers import CompareVanillaScore as Score



prompt = PromptTemplate(
        template=general_schema,
        input_variables=["question", "answer_a", "answer_b", "format_instruction"],
    )

parser = JsonOutputParser(pydantic_object=Score)
model = "gpt-4"


def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
            
def create_dict(id, question, answer_a, answer_b):
    
    #processing the original answer first
    orig_prompt = prompt.invoke(
        {
            "question": question, 
            "answer_a": answer_a, 
            "answer_b": answer_b,
            "format_instruction": parser.get_format_instructions()
        }
    )
    orig_dict = {
        "custom_id": id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model, 
            "messages": [
                {
                    "role": "system", 
                    "content": system_prompt},
                {
                    "role": "user", 
                    "content": orig_prompt.text}
                ],
            "max_tokens": 1000,
            "temperature": 0}
    } 
    return orig_dict 

def process_instance(data_row, p_mode):
    if p_mode:
        id = data_row['cdx']
        prompt = data_row['question']
        original_answer = data_row['perturbed_gpt4']
        perturbed_answer = data_row['og']
        orig_id = f"{id}~compare_vanilla~perturb_orig"
    else:        
        id = data_row['cdx']
        prompt = data_row['question']
        original_answer = data_row['og']
        perturbed_answer = data_row['perturbed_gpt4']
        orig_id = f"{id}~compare_vanilla~orig_perturb"
    
    #processing the original answer
    
    orig_dict = create_dict(orig_id, prompt, original_answer, perturbed_answer)
    
    return [orig_dict] 
    

    

def parse_args():
    parser = argparse.ArgumentParser(description='Axes with Rubrics')
    parser.add_argument("--file_name", type=str, help="File name of the data")
    parser.add_argument("--out_file_name", type=str, help="Output jsonl File name")
    parser.add_argument("--batch_mode", action="store_true", default=True, help="Run evaluation in batch mode")
    parser.add_argument("--model", type=str, choices=['gpt-4o', "gpt-4-turbo", "gpt-3.5-turbo-0125", "llama3-70b", "claude3-opus", 'gemini-1.5-flash', 'gemini-1.5-pro'], help="Model name")
    parser.add_argument("--p_mode", action="store_true", default=False, help="Run in perturbed first mode")
    args = parser.parse_args()
    return args

def main(args):
    global model
    
    df = pd.read_csv(args.file_name, sep='\t')
    df_dict = df.to_dict('records')
    model = args.model

    if model in ('gpt-4o', "gpt-4-turbo", "gpt-3.5-turbo-0125", "llama3-70b", "claude3-opus", 'gemini-1.5-flash', 'gemini-1.5-pro'):
        if args.batch_mode:
            final_jsonl = []
            for row in df_dict:
                row_dicts = process_instance(row, args.p_mode)
                final_jsonl.extend(row_dicts)
                if args.p_mode:
                    out_file_name = args.out_file_name.split(".json")[0] + "_perturb.jsonl"
                    write_jsonl(final_jsonl, out_file_name)
                else:
                    write_jsonl(final_jsonl, args.out_file_name)
        else:
            print("2Still pending")
    else:
        print("3Still pending")
        
        
if __name__ == '__main__':
    args = parse_args()
    main(args)