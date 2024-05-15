import os
import json
import argparse
from openai import OpenAI
from dotenv import dotenv_values
from config import *

# env_vars = dotenv_values('.env')
# API_KEY = env_vars['OPENAI_API_KEY']
API_KEY = OPENAI_API_KEY
client = OpenAI(api_key=API_KEY)


def parse_args():
    parser = argparse.ArgumentParser(description='Batch processing')
    parser.add_argument('--create_batch', action="store_true", help='Create a batch job')
    parser.add_argument('--get_results', action="store_true", help='Get results from batch job')
    parser.add_argument('--check_status', action="store_true", help='Check status of batch job')
    parser.add_argument('--job_name', type=str, help='File name of batch job')
    parser.add_argument('--input_file_name', type=str, help='File name of batch job')
    parser.add_argument('--output_file_name', type=str, help='File name of batch job')
    parser.add_argument('--job_desc', type=str, help='Description of batch job')
    args = parser.parse_args()
    return args

def main(args):
    if args.create_batch:
        batch_input_file = client.files.create(
        file=open(f"{args.input_file_name}", "rb"),
        purpose="batch"
        )
        batch_input_file_id = batch_input_file.id

        batch_obj = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": args.job_desc
            }
        )
        print(batch_obj)
        print(f"Here is the generated file name: {batch_input_file_id}")
    elif args.get_results:
        content = client.files.content(args.job_name)
        with open(f"{args.output_file_name}", "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

    elif args.check_status:
        print(client.batches.retrieve(args.job_name))


if __name__ == '__main__':
    args = parse_args()
    main(args)