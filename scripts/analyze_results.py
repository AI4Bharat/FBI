import json
import argparse
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field



def read_jsonl(file_name):
    results_data = []
    with open(file_name) as f:
        for line in f:
            results_data.append(json.loads(line))
    return results_data

def analyze_vanilla_batch_result(data):
    results = dict()
    errors = 0
    for item in data:
        id = item['custom_id']
        item_id = id.split("~")[0]
        res_type = id.split("~")[-1]
        try:
            if item_id in results:
                results[item_id][res_type] = 
        