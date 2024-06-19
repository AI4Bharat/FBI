import json
import argparse
import re
import ast
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, ValidationError
from llm_evaluators.parsers import *



def read_jsonl(file_name):
    results_data = []
    with open(file_name) as f:
        for line in f:
            results_data.append(json.loads(line))
    return results_data

class CustomJsonOutputParser:
    """
    A custom JSON output parser that extracts the outermost JSON object from a string,
    attempts to parse it, and validates it against a specified Pydantic model.

    Attributes:
        pydantic_model (BaseModel): The Pydantic model class used for validating the parsed JSON.

    Methods:
        find_outermost_json(data): Extracts the outermost JSON object from a string.
        invoke(data): Parses and validates the JSON object against the Pydantic model.
    """

    def __init__(self, pydantic_model):
        """
        Initializes the parser with a Pydantic model for JSON validation.

        Args:
            pydantic_model (BaseModel): A subclass of pydantic.BaseModel that defines the expected JSON schema.

        Raises:
            ValueError: If the provided model is not a subclass of pydantic.BaseModel.
        """
        if not issubclass(pydantic_model, BaseModel):
            raise ValueError("pydantic_model must be a subclass of pydantic.BaseModel")
        self.pydantic_model = pydantic_model

    def find_outermost_json(self, data):
        """
        Extracts the outermost JSON string from a potentially nested or malformed JSON input.

        Args:
            data (str): The string containing JSON data.

        Returns:
            str: The outermost JSON string if found.

        Raises:
            Exception: If no valid JSON object could be extracted.
        """
        stack = []
        in_string = False
        escape = False
        start = -1

        for i, char in enumerate(data):
            if char == '"' and not escape:
                in_string = not in_string
            elif char == '\\' and in_string:
                escape = not escape
            elif char == '{' and not in_string:
                if not stack:
                    start = i
                stack.append(char)
            elif char == '}' and not in_string:
                if stack:
                    stack.pop()
                if not stack and start != -1:
                    return data[start:i+1]
            elif escape:
                escape = False
        
        raise Exception("No valid JSON could be extracted")

    def invoke(self, data):
        """
        Parses the extracted JSON string and validates it against the Pydantic model.

        Args:
            data (str): The string from which to extract and validate JSON.

        Returns:
            BaseModel: An instance of the Pydantic model populated with validated data.

        Raises:
            Exception: If the JSON string is invalid or if validation against the Pydantic model fails.
        """
        json_string = self.find_outermost_json(data)
        if json_string:
            try:
                json_data = json.loads(json_string)
            except json.JSONDecodeError:
                try:
                    json_data = ast.literal_eval(json_string)
                except Exception as e:
                    raise Exception("JSON parsing failed") from e

            try:
                self.pydantic_model(**json_data)
                return json_data
            except ValidationError as e:
                raise Exception("JSON validation failed") from e

        raise Exception("No valid JSON could be extracted")
    

def analyze_single_vanilla_batch_result(data):
    
    def parse_and_get_score(data):
        parser = JsonOutputParser(pydantic_object=SingleVanillaScore)
        #try by default using the langchain parser
        try:
            return parser.invoke(data)['score']
        except OutputParserException:
            #if fails, try custom parser
            custom_json_parser = CustomJsonOutputParser(pydantic_model=SingleVanillaScore)
            return custom_json_parser.invoke(data)['score']
        except Exception as e:
            raise e
    
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
        

def analyze_single_vanilla_cot_batch_result(data):
    
    def parse_and_get_score(data):
        parser = JsonOutputParser(pydantic_object=SingleVanillaCOTScore)
        #try by default using the langchain parser
        try:
            return parser.invoke(data)['score']
        except OutputParserException:
            #if fails, try custom parser
            custom_json_parser = CustomJsonOutputParser(pydantic_model=SingleVanillaCOTScore)
            return custom_json_parser.invoke(data)['score']
        except Exception as e:
            raise e
    
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

def analyze_single_rubrics_batch_result(data):
    
    def parse_and_get_score(data):
        parser = JsonOutputParser(pydantic_object=SingleRubricsScore)
        #try by default using the langchain parser
        try:
            return parser.invoke(data)['score']
        except OutputParserException:
            #if fails, try custom parser
            custom_json_parser = CustomJsonOutputParser(pydantic_model=SingleRubricsScore)
            return custom_json_parser.invoke(data)['score']
        except Exception as e:
            raise e
    
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
    
    def parse_and_get_score(data):
        parser = JsonOutputParser(pydantic_object=SingleAxesRubricsScore)
        #try by default using the langchain parser
        try:
            return parser.invoke(data)['score']
        except OutputParserException:
            #if fails, try custom parser
            custom_json_parser = CustomJsonOutputParser(pydantic_model=SingleAxesRubricsScore)
            return custom_json_parser.invoke(data)['score']
        except Exception as e:
            raise e
        
    #processing the batch results
    results = dict()
    errors = 0
    for item in data:
        id = item['custom_id']
        item_id = id.split("~")[0]
        res_type = id.split("~")[-1]
        axes_type = id.split("~")[-2]
        
        try:
            score = parse_and_get_score(item['response']['body']['choices'][0]['message']['content'])
            if axes_type in results:
                if item_id in results[axes_type]:
                    results[axes_type][item_id][res_type] = score
                else:
                    results[axes_type][item_id] = {res_type: score}
            else:
                results[axes_type] = {item_id: {res_type: score}}
        except Exception as e:
            print(e)
            score = None
            errors += 1
            
    axes_list = results.keys()
    changed = dict()
    changed_ids = dict()
    unprocessed_ids = dict()
    for axes in axes_list:
        changed[axes] = 0
        changed_ids[axes] = []
        unprocessed_ids[axes] = []
        
    for axes, inst in results.items():
        for id, res in inst.items():
            try:
                if int(res['pert']) < int(res['orig']):
                    changed[axes] += 1
                    changed_ids[axes].append(id)
            except Exception as e:
                print(e)
                unprocessed_ids[axes].append(id)
    
    return results, changed, changed_ids, unprocessed_ids, errors

def analyze_single_axes_batch_result(data):
    
    def parse_and_get_score(data):
        parser = JsonOutputParser(pydantic_object=SingleAxesScore)
        #try by default using the langchain parser
        try:
            return parser.invoke(data)['score']
        except OutputParserException:
            #if fails, try custom parser
            custom_json_parser = CustomJsonOutputParser(pydantic_model=SingleAxesScore)
            return custom_json_parser.invoke(data)['score']
        except Exception as e:
            raise e
        
    #processing the batch results
    results = dict()
    errors = 0
    for item in data:
        id = item['custom_id']
        item_id = id.split("~")[0]
        res_type = id.split("~")[-1]
        axes_type = id.split("~")[-2]
        
        try:
            score = parse_and_get_score(item['response']['body']['choices'][0]['message']['content'])
            if axes_type in results:
                if item_id in results[axes_type]:
                    results[axes_type][item_id][res_type] = score
                else:
                    results[axes_type][item_id] = {res_type: score}
            else:
                results[axes_type] = {item_id: {res_type: score}}
        except Exception as e:
            print(e)
            score = None
            errors += 1
            
    axes_list = results.keys()
    changed = dict()
    changed_ids = dict()
    unprocessed_ids = dict()
    for axes in axes_list:
        changed[axes] = 0
        changed_ids[axes] = []
        unprocessed_ids[axes] = []
        
    for axes, inst in results.items():
        for id, res in inst.items():
            try:
                if int(res['pert']) < int(res['orig']):
                    changed[axes] += 1
                    changed_ids[axes].append(id)
            except Exception as e:
                print(e)
                unprocessed_ids[axes].append(id)
    
    return results, changed, changed_ids, unprocessed_ids, errors


def analyze_compare_vanilla_cot_batch_result(data):
    def parse_and_get_score(data):
        parser = JsonOutputParser(pydantic_object=CompareVanillaCOTScore)
        #try by default using the langchain parser
        try:
            return parser.invoke(data)['verdict']
        except OutputParserException:
            #if fails, try custom parser
            custom_json_parser = CustomJsonOutputParser(pydantic_model=CompareVanillaCOTScore)
            return custom_json_parser.invoke(data)['verdict']
        except Exception as e:
            raise e
        
    #processing the batch results
    results = dict()
    errors = 0
    error_ids = []
    for item in data:
        id = item['custom_id']
        item_id = id.split("~")[0]
        res_type = id.split("~")[-1]
        
        try:
            verdict = parse_and_get_score(item['response']['body']['choices'][0]['message']['content'])
            results[item_id] = verdict
            
        except Exception as e:
            print(e)
            errors += 1
            error_ids.append(item_id)
            verdict = None
        
    value_set = list(results.values())
    #dictionary of value counts
    value_counts = {i: value_set.count(i) for i in value_set}
    return results, value_counts, errors, error_ids

def analyze_compare_vanilla_batch_result(data):
    def parse_and_get_score(data):
        parser = JsonOutputParser(pydantic_object=CompareVanillaScore)
        #try by default using the langchain parser
        try:
            return parser.invoke(data)['verdict']
        except OutputParserException:
            #if fails, try custom parser
            custom_json_parser = CustomJsonOutputParser(pydantic_model=CompareVanillaScore)
            return custom_json_parser.invoke(data)['verdict']
        except Exception as e:
            raise e
        
    #processing the batch results
    results = dict()
    errors = 0
    error_ids = []
    for item in data:
        id = item['custom_id']
        item_id = id.split("~")[0]
        res_type = id.split("~")[-1]
        
        try:
            verdict = parse_and_get_score(item['response']['body']['choices'][0]['message']['content'])
            results[item_id] = verdict
            
        except Exception as e:
            print(e)
            errors += 1
            error_ids.append(item_id)
            verdict = None
        
    value_set = list(results.values())
    #dictionary of value counts
    value_counts = {i: value_set.count(i) for i in value_set}
    return results, value_counts, errors, error_ids

def analyze_compare_rules_batch_result(data):
    def parse_and_get_score(data):
        parser = JsonOutputParser(pydantic_object=CompareRulesScore)
        #try by default using the langchain parser
        try:
            return parser.invoke(data)['verdict']
        except OutputParserException:
            #if fails, try custom parser
            custom_json_parser = CustomJsonOutputParser(pydantic_model=CompareRulesScore)
            return custom_json_parser.invoke(data)['verdict']
        except Exception as e:
            raise e
        
    #processing the batch results
    results = dict()
    errors = 0
    error_ids = []
    for item in data:
        id = item['custom_id']
        item_id = id.split("~")[0]
        res_type = id.split("~")[-1]
        
        try:
            verdict = parse_and_get_score(item['response']['body']['choices'][0]['message']['content'])
            results[item_id] = verdict
            
        except Exception as e:
            print(e)
            errors += 1
            error_ids.append(item_id)
            verdict = None
        
    value_set = list(results.values())
    #dictionary of value counts
    value_counts = {i: value_set.count(i) for i in value_set}
    return results, value_counts, errors, error_ids  

def analyze_compare_axes_batch_result(data):
    def parse_and_get_score(data):
        parser = JsonOutputParser(pydantic_object=CompareAxesScore)
        #try by default using the langchain parser
        try:
            return parser.invoke(data)['verdict']
        except OutputParserException:
            #if fails, try custom parser
            custom_json_parser = CustomJsonOutputParser(pydantic_model=CompareAxesScore)
            return custom_json_parser.invoke(data)['verdict']
        except Exception as e:
            raise e
        
    #processing the batch results
    results = dict()
    errors = 0
    error_ids = []
    for item in data:
        id = item['custom_id']
        item_id = id.split("~")[0]
        res_type = id.split("~")[-1]
        
        try:
            verdict = parse_and_get_score(item['response']['body']['choices'][0]['message']['content'])
            results[item_id] = verdict
            
        except Exception as e:
            print(e)
            errors += 1
            error_ids.append(item_id)
            verdict = None
        
    value_set = list(results.values())
    # print(value_set)
    if any(isinstance(i, list) for i in value_set):
        value_set = [item for item in value_set if not isinstance(item, list)]
    #dictionary of value counts
    value_counts = {i: value_set.count(i) for i in value_set}
    return results, value_counts, errors, error_ids  
 
def analyze_compare_results(normal_results, perturbed_results):
    answer_a = 0
    answer_b = 0
    answer_c = 0
    answer_d = 0
    other_answers = 0
    
    answer_a_allowed = ('A', '[[A]]', 0, 1, '0', '1')
    answer_b_allowed = ('B', '[[B]]', 2, '2')
    answer_c_allowed = ('C', '[[C]]', 3, '3')
    answer_d_allowed = ('D', '[[D]]', 4, '4')
    
    position_bias = 0
    position_bias_id = []
    error_counts = 0
    error_id = []
    
    for key, normal_value in normal_results.items():
        if key in perturbed_results:
            # print(key, value, perturbed_results[key])
            perturbed_value = perturbed_results[key]
            # if (normal_value == 'A' and perturbed_value == 'B') or (normal_value == '[[A]]' and perturbed_value == '[[B]]'):
            if normal_value in answer_a_allowed and perturbed_value in answer_b_allowed:
                answer_a += 1
            # elif (normal_value == 'B' and perturbed_value == 'A') or (normal_value == '[[B]]' and perturbed_value == '[[A]]'):
            elif normal_value in answer_b_allowed and perturbed_value in answer_a_allowed:
                answer_b += 1
            # elif (normal_value == 'C' and perturbed_value == 'C') or (normal_value == '[[C]]' and perturbed_value == '[[C]]'):
            elif normal_value in answer_c_allowed and perturbed_value in answer_c_allowed:
                answer_c += 1
            # elif (normal_value == 'D' and perturbed_value == 'D') or (normal_value == '[[D]]' and perturbed_value == '[[D]]'):
            elif normal_value in answer_d_allowed and perturbed_value in answer_d_allowed:
                answer_d += 1
            else:
                position_bias += 1
                position_bias_id.append(key)
        else:
            error_counts += 1
            error_id.append(key)
    answer_dict = {
        "a": answer_a,
        "b": answer_b,
        "c": answer_c,
        "d": answer_d,
        'others': other_answers
    }
    return answer_dict, position_bias, position_bias_id, error_counts, error_id
    
    
def analyze_reference_batch_result(data):
    
    def parse_and_get_score(data):
        parser = JsonOutputParser(pydantic_object=ReferenceScore)
        #try by default using the langchain parser
        try:
            return parser.invoke(data)['score']
        except OutputParserException:
            #if fails, try custom parser
            custom_json_parser = CustomJsonOutputParser(pydantic_model=ReferenceScore)
            return custom_json_parser.invoke(data)['score']
        except Exception as e:
            raise e
    
    #processing the batch results
    results = dict()
    errors = 0
    for item in data:
        id = item['custom_id']
        item_id = id.split("~")[0]
        res_type = id.split("~")[-1]
        
        try:
            score = parse_and_get_score(item['response']['body']['choices'][0]['message']['content'])
            results[item_id] = score
        except Exception as e:
            print(e)
            score = None
            errors += 1
            
    value_counts = dict()
    for key, value in results.items():
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1
            
    return results, value_counts, errors


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze LLM results')
    parser.add_argument("--file_name", type=str, help="File name of the data")
    parser.add_argument("--type", type=str, choices=['single_vanilla', 'single_vanilla_cot', 'single_axes_rubrics', 'single_axes', 
                                                     'single_rubrics', 'compare_vanilla', 'compare_vanilla_cot', 'compare_rules', 
                                                     'compare_axes','reference', 'single_axes_rubrics_inc', 'single_rubrics_inc', 'compare_axes_nr'], help="Type of the data")
    parser.add_argument("--num_axes", type=int, help="Number of axes for axes rubrics", default=5, required=False)
    return parser.parse_args()

def main(args):
    if args.type == 'single_vanilla_cot':
        data = read_jsonl(args.file_name)
        results, changed, changed_ids, unprocessed_ids, errors = analyze_single_vanilla_cot_batch_result(data)
        print(f"Total number of results: {len(data)/2}")
        print(f"Total number of changed results: {changed}")
        print(f"Total number of unchanged results: {len(data)/2 - errors/2 - changed}")
        print(f"Changed ids: {changed_ids}")
        print(f"Unprocessed ids: {unprocessed_ids}")
        print(f"Errors: {errors}")
    elif args.type == 'single_axes_rubrics' or args.type == 'single_axes_rubrics_inc':
        data = read_jsonl(args.file_name)
        results, changed, changed_ids, unprocessed_ids, errors = analyze_single_axes_rubrics_batch_result(data)
        for axes, inst in results.items():
            print("*********************")
            print(f"Results for axes: {axes}")
            print(f"Total number of results: {len(data)/args.num_axes/2}")
            print(f"Total number of changed results: {changed[axes]}")
            print(f"Total number of unchanged results: {len(data)/args.num_axes/2 - errors/args.num_axes/2 - changed[axes]}")
            print(f"Changed ids: {changed_ids[axes]}")
            print(f"Unprocessed ids: {unprocessed_ids[axes]}")
            print(f"Errors: {errors}")
    elif args.type == 'single_vanilla':
        data = read_jsonl(args.file_name)
        results, changed, changed_ids, unprocessed_ids, errors = analyze_single_vanilla_batch_result(data)
        print(f"Total number of results: {len(data)/2}")
        print(f"Total number of changed results: {changed}")
        print(f"Total number of unchanged results: {len(data)/2 - errors/2 - changed}")
        print(f"Changed ids: {changed_ids}")
        print(f"Unprocessed ids: {unprocessed_ids}")
        print(f"Errors: {errors}")
    elif args.type == 'single_axes':
        data = read_jsonl(args.file_name)
        results, changed, changed_ids, unprocessed_ids, errors = analyze_single_axes_batch_result(data)
        for axes, inst in results.items():
            print("*********************")
            print(f"Results for axes: {axes}")
            print(f"Total number of results: {len(data)/args.num_axes/2}")
            print(f"Total number of changed results: {changed[axes]}")
            print(f"Total number of unchanged results: {len(data)/args.num_axes/2 - errors/args.num_axes/2 - changed[axes]}")
            # print(f"Changed ids: {changed_ids[axes]}")
            print(f"Unprocessed ids: {unprocessed_ids[axes]}")
            print(f"Errors: {errors}")
    elif args.type == 'single_rubrics' or args.type == 'single_rubrics_inc':
        data = read_jsonl(args.file_name)
        results, changed, changed_ids, unprocessed_ids, errors = analyze_single_rubrics_batch_result(data)
        print(f"Total number of results: {len(data)/2}")
        print(f"Total number of changed results: {changed}")
        print(f"Total number of unchanged results: {len(data)/2 - errors/2 - changed}")
        # print(f"Changed ids: {changed_ids}")
        print(f"Unprocessed ids: {unprocessed_ids}")
        print(f"Errors: {errors}")
    elif args.type == 'compare_vanilla_cot':
        #analyzing normal mode results
        data = read_jsonl(args.file_name)
        normal_results, normal_value_counts, normal_errors, normal_error_ids = analyze_compare_vanilla_cot_batch_result(data)
        
        #analyzing perturbed mode results
        perturbed_data = read_jsonl(args.file_name.split(".jsonl_out")[0] + "_perturbed.jsonl_outputs.jsonl")
        perturbed_results, perturbed_value_counts, perturbed_errors, perturbed_error_ids = analyze_compare_vanilla_cot_batch_result(perturbed_data)
        
        common_error_ids = set(normal_error_ids).intersection(set(perturbed_error_ids))
        answer_dict, position_bias, position_bias_id, error_counts, error_id = analyze_compare_results(normal_results, perturbed_results)
        print("Total number of results: ", len(data))
        print("Answer Dictionary: ", answer_dict)
        print("Potential Position Bias: ", position_bias)
        print("Error Counts: ", error_counts)
        print("Parsing Errors: ", len(common_error_ids))
    elif args.type == 'compare_rules':
        data = read_jsonl(args.file_name)
        normal_results, normal_value_counts, normal_errors, normal_error_ids = analyze_compare_rules_batch_result(data)
        
        #analyzing perturbed mode results
        perturbed_data = read_jsonl(args.file_name.split(".jsonl_out")[0] + "_perturbed.jsonl_outputs.jsonl")
        perturbed_results, perturbed_value_counts, perturbed_errors, perturbed_error_ids = analyze_compare_rules_batch_result(perturbed_data)
        
        common_error_ids = set(normal_error_ids).intersection(set(perturbed_error_ids))
        answer_dict, position_bias, position_bias_id, error_counts, error_id = analyze_compare_results(normal_results, perturbed_results)
        
        print("Total number of results: ", len(data))
        print("Answer Dictionary: ", answer_dict)
        print("Potential Position Bias: ", position_bias)
        print("Error Counts: ", error_counts)
        print("Parsing Errors: ", len(common_error_ids))
    
    elif args.type == 'compare_axes' or args.type == 'compare_axes_nr':
        data = read_jsonl(args.file_name)
        normal_results, normal_value_counts, normal_errors, normal_error_ids = analyze_compare_axes_batch_result(data)

        # print(normal_value_counts)
        #analyzing perturbed mode results
        perturbed_data = read_jsonl(args.file_name.split(".jsonl_out")[0] + "_perturbed.jsonl_outputs.jsonl")
        perturbed_results, perturbed_value_counts, perturbed_errors, perturbed_error_ids = analyze_compare_axes_batch_result(perturbed_data)
        
        # print(perturbed_value_counts)
        common_error_ids = set(normal_error_ids).intersection(set(perturbed_error_ids))
        answer_dict, position_bias, position_bias_id, error_counts, error_id = analyze_compare_results(normal_results, perturbed_results)
        print("Total number of results: ", len(data))
        print("Answer Dictionary: ", answer_dict)
        print("Potential Position Bias: ", position_bias)
        print("Error Counts: ", error_counts)
        print("Parsing Errors: ", len(common_error_ids))
        
    elif args.type == 'compare_vanilla':
        #analyzing normal mode results
        data = read_jsonl(args.file_name)
        normal_results, normal_value_counts, normal_errors, normal_error_ids = analyze_compare_vanilla_batch_result(data)
        
        #analyzing perturbed mode results
        perturbed_data = read_jsonl(args.file_name.split(".jsonl_out")[0] + "_perturbed.jsonl_outputs.jsonl")
        perturbed_results, perturbed_value_counts, perturbed_errors, perturbed_error_ids = analyze_compare_vanilla_batch_result(perturbed_data)
        
        common_error_ids = set(normal_error_ids).intersection(set(perturbed_error_ids))
        answer_dict, position_bias, position_bias_id, error_counts, error_id = analyze_compare_results(normal_results, perturbed_results)
        print("Total number of results: ", len(data))
        print("Answer Dictionary: ", answer_dict)
        print("Potential Position Bias: ", position_bias)
        print("Error Counts: ", error_counts)
        print("Parsing Errors: ", len(common_error_ids))
    elif args.type == 'reference':
        data = read_jsonl(args.file_name)
        results, value_counts, errors = analyze_reference_batch_result(data)
        
        print(f"Total number of results: {len(data)}")
        print(f"Value Counts: {value_counts}")
        if 10 in value_counts:
            print("Number 10: ", value_counts[10])
        if 9 in value_counts:
            print("Number 9: ", value_counts[9])
        if 8 in value_counts:
            print("Number 8: ", value_counts[8])
        print("Number less than 8: ", sum([value_counts[i] for i in range(1,8) if i in value_counts]))
        print(f"Errors: {errors}")
    else:
        print("Still pending")
        
        
if __name__ == "__main__":
    args = parse_args()
    main(args)