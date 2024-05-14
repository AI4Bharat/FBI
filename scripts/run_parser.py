import  argparse
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser

from perturbations.prompts import Facts, Errors, Stitch


def parse_args():
    parser = argparse.ArgumentParser(description='Run JSON parser on model outputs')
    parser.add_argument('--file_name', type=str, help='File name of tsv')
    parser.add_argument('--data_dir', type=str, default='/Users/sumanth/code/fbi/outputs', help='Path to model outputs directory')
    parser.add_argument('--parser', type=str, default='facts', help='Parser to use')
    args = parser.parse_args()
    return args


def main(args):
    if args.parser == 'facts':
        json_parser = JsonOutputParser(pydantic_object=Facts)
    elif args.parser == 'errors':
        json_parser = JsonOutputParser(pydantic_object=Errors)
    elif args.parser == 'stitch':
        json_parser = JsonOutputParser(pydantic_object=Stitch)
    else:
        raise ValueError('Invalid parser')
    
    df = pd.read_csv(f"{args.data_dir}/{args.file_name}", sep="\t")
    df['json_parsed'] = df['answer'].apply(lambda x: json_parser.invoke(x))

    df.to_csv(f"{args.data_dir}/{args.file_name.split('.')[0]}-parsed.tsv", sep="\t", index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)