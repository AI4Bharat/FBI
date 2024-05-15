import json
import pandas as pd
import argparse
import pathlib

def parse_args():
    parser = argparse.ArgumentParser(description='Convert jsonl to tsv')
    parser.add_argument('--file_name', type=str, help='File name of jsonl')
    parser.add_argument('--data_dir', type=str, default='/Users/sumanth/code/fbi/outputs', help='Path to model outputs directory')
    args = parser.parse_args()
    return args


def main(args):
    testset = pd.read_csv(f"/Users/sumanth/code/fbi/data/testset-v0-answers.tsv", sep="\t")

    with open(f"{args.data_dir}/{args.file_name}", "r") as f:
        data = [json.loads(line) for line in f]

    jsons = []
    for d in data:
        cdx = d['custom_id']
        answer = d['response']['body']['choices'][0]['message']['content']
        logprobs = d['response']['body']['choices'][0]['logprobs']
        finish_reason = d['response']['body']['choices'][0]['finish_reason']
        created = d['response']['body']['created']
        model = d['response']['body']['model']
        object_ = d['response']['body']['object']
        total_tokens = d['response']['body']['usage']['total_tokens']
        prompt_tokens = d['response']['body']['usage']['prompt_tokens']
        completion_tokens = d['response']['body']['usage']['completion_tokens']
        error = d['error']

        dict_ = {
            'cdx': d['custom_id'],
            'ability': testset[testset['cdx'] == cdx]['ability'].values[0],
            'source': testset[testset['cdx'] == cdx]['source'].values[0],
            'question': testset[testset['cdx'] == cdx]['question'].values[0],
            'answer': answer,
            'logprobs': logprobs,
            'finish_reason': finish_reason,
            'created': created,
            'model': model,
            'object': object_,
            'total_tokens': total_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'error': error
        }
        jsons.append(dict_)


    df = pd.DataFrame(jsons)
    df.to_csv(f"{args.data_dir}/{pathlib.Path(args.file_name).stem}.tsv", sep="\t", index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)