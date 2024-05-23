import argparse
import pandas as pd

from perturbations.prompts.factual import (
    factual_perturbations,
    factual_perturbations_v2,
    factual_perturbations_v3,
    contextual_fact_perturbations,
    number_perturbations,
    entity_perturbations,
    add_incorrect_fact,
    opposite_perturbations,
    remove_fact
)
from perturbations.prompts.long_form import (
    grammar_perturbations,
    spelling_perturbations,
    chronological_perturbations,
    consistency_perturbations,
    coherence_perturbations,
    formatting_perturbations,
    comprehensiveness_perturbations,
    superficial_perturbations
)
from perturbations.prompts.reasoning import (
    calculation_errors,
    final_answer_perturbations,
    incorrect_units,
    operation_order,
    wrong_formula,
    copying_numbers_errors
)


def parse_args():
    parser = argparse.ArgumentParser(description='Batch processing')
    parser.add_argument('--data_dir', type=str, default='/Users/sumanth/code/fbi/data', help='Path to data directory')
    parser.add_argument('--subset', type=str, choices=['all', 'reasoning', 'factual', 'instruction-following', 'long-form'], help='Subset of testset')
    parser.add_argument('--file_name', type=str, default='testset-v0-answers.tsv', help='name of the input file')
    # model arguments
    parser.add_argument('--model', type=str, default='gpt-4-turbo', help='Model to use')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=1, help='Top p for sampling')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Max tokens for sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0, help='Frequency penalty for sampling')
    parser.add_argument('--presence_penalty', type=float, default=0, help='Presence penalty for sampling')
    # perturbation arguments
    parser.add_argument('--facts', action='store_true', help='Generate factual statements')
    parser.add_argument('--errors', action='store_true', help='Generate factual statements with errors')
    parser.add_argument('--stitch', action='store_true', help='Stitch factual statements with errors')
    # debug
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    return args


def main(args):
    testset = pd.read_csv(f'{args.data_dir}/{args.file_name}', sep='\t')
    if args.subset != 'all':
        testset = testset[testset['ability'] == args.subset]

    if args.subset == 'factual':
        factual_perturbations(args, testset)
        factual_perturbations_v2(args, testset)
        factual_perturbations_v3(args, testset)
        contextual_fact_perturbations(args, testset)
        number_perturbations(args, testset)
        entity_perturbations(args, testset)
        add_incorrect_fact(args, testset)
        opposite_perturbations(args, testset)
        remove_fact(args, testset)
    elif args.subset == 'reasoning':
        calculation_errors(args, testset)
        final_answer_perturbations(args, testset)
        incorrect_units(args, testset)
        operation_order(args, testset)
        wrong_formula(args, testset)
        copying_numbers_errors(args, testset)
    elif args.subset == 'instruction-following':
        pass
    elif args.subset == 'long-form':
        grammar_perturbations(args, testset)
        spelling_perturbations(args, testset)
        chronological_perturbations(args, testset)
        consistency_perturbations(args, testset)
        coherence_perturbations(args, testset)
        formatting_perturbations(args, testset)
        comprehensiveness_perturbations(args, testset)
        superficial_perturbations(args, testset)
    else:
        raise ValueError('Invalid subset')


if __name__ == '__main__':
    args = parse_args()
    main(args)