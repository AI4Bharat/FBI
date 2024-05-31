import os
import json
import argparse
from openai import OpenAI

API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=API_KEY)

def parse_args():
    parser = argparse.ArgumentParser(description='Batch processing')
    parser.add_argument('--file_name', type=str, help='File name of batch job')
    parser.add_argument('--data_path', type=str, default='/Users/sumanth/code/fbi/data', help='Path to data directory')
    args = parser.parse_args()
    return args

def main(args):
    #read the batch list from the file
    with open(f"{args.file_name}", "r") as f:
        batch_list = json.load(f)
        
    #for each batch, fetch the status object
    for batch in batch_list:
        print(f"Collecting {batch['input_file']}_answers.jsonl")
        status = client.batches.retrieve(batch['batch_id']).to_dict()
        output_file_id = status['output_file_id']
        content = client.files.content(output_file_id)
        jsonl_lines = content.content.decode("utf-8").splitlines()

        with open(f"{args.data_path}/{batch['input_file']}_answers.jsonl", "w") as f:
            for line in jsonl_lines:
                json_obj = json.loads(line)
                f.write(json.dumps(json_obj) + "\n")
        
if __name__ == '__main__':
    args = parse_args()
    main(args)