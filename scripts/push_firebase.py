import os
import argparse
import json
import firebase_admin
import pathlib
import traceback
import pandas as pd
from tqdm import tqdm
from firebase_admin import firestore, credentials

from langchain_core.output_parsers import JsonOutputParser
from perturbations.parsers import DirectError
from perturbations.parsers import CustomJsonOutputParser


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/safi/pertubation_checklist/fbi/scripts/firebase_key.json"
cred = credentials.Certificate("/home/safi/pertubation_checklist/fbi/scripts/firebase_key.json")

app = firebase_admin.initialize_app(cred)
db = firestore.Client(project="setu-dashboard", database="gpt4-eval-app")

def push_to_firebase(data):
    for record in data:
        doc_ref = db.collection("data").document(record['id'])
        doc_ref.set(
            {
                "id": record['id'],
                "question": record['question'],
                "perturbation_type": record['perturbation_type'],
                "expected_perturbation": record['expected_perturbation'],
                "original": record['original'],
                "perturbed": record['perturbed'],
                "user": record['user'],
                "status": "pending"
            }
        )
        
    print("Data pushed to firebase")
    
def generate_users(num_users):
    users = [f"user{i}" for i in range(1, num_users + 1)]
    while True:
        yield from users
    
def get_perturbed_type(p_type):
    perturbation_type_map = {
        "remove-fact": "Removing an important fact from the answer",
        "opposite-fact": "Introducing a negation or an opposite fact in the answer",
        "number-errors": "Introducing errors in the numbers in the answer",
        "incorrect-fact": "Introducing an incorrect fact in the answer",
        "entity-errors": "Introducing errors in the entities in the answer",
        "contextual-errors": "Replacing facts in the answer with contextually similar incorrect facts",
        "wrong-formula": "Introducing errors in the formula used in the answer",
        "operation-order": "Introducing errors in the order of operations in the answer",
        "final-answer-errors": "Introducing errors only in the final answer with all remaining steps same",
        "copying-numbers-errors": "Introducing errors by copying numbers incorrectly from the question in the answer",
        "incorrect-units": "Introducing errors in the units in the answer",
        "calculation-errors": "Introducing errors in the calculations in the answer",
    }
    
    return perturbation_type_map[p_type]
    
def parse_response(response):
    json_parser = JsonOutputParser(pydantic_object=DirectError)
    try:
        parsed_response = json_parser.invoke(response)
        return parsed_response
    except Exception as e:
        try:
            json_parser = CustomJsonOutputParser(pydantic_model=DirectError)
            parsed_response = json_parser.invoke(response)
            return parsed_response
        except Exception as e:
            raise
        


def prepare_doc(doc, testset):
    try:
        cdx = doc['custom_id']
        response = doc['response']['body']['choices'][0]['message']['content']
        question = testset[testset['cdx'] == cdx]['question'].values[0]
        original_answer = testset[testset['cdx'] == cdx]['answer'].values[0]
        parsed_response = parse_response(response)
        dict_ = {
            'id': cdx,
            'question': question,
            'perturbation_type': None,
            'expected_perturbation': parsed_response['explanation'],
            'original': original_answer,
            'perturbed': parsed_response['err_answer'],
            'user': None
        }
        return dict_
    except Exception as e:
        # print(e)
        raise
    
        
        
    
    

def parse_args():
    parser = argparse.ArgumentParser(description='Push to firebase')
    parser.add_argument('--file_path', type=str, help='File having paths of all the json files')
    parser.add_argument("--answers_path", type=str, default="/home/safi/pertubation_checklist/data/v0_perturbations/testset-v0-answers.tsv", help="Path to the answers file")
    parser.add_argument("--num_users", type=int, default=1, help="Number of users")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    return args

def main(args):
    
    file_path_list = []
    with open(args.file_path, 'r') as f:
        for line in f:
            file_path_list.append(line.strip())
            
    testset = pd.read_csv(args.answers_path, sep="\t")
    final_data = []
    error_data = []
    #process each file
    user_generator = generate_users(args.num_users)
    for file in tqdm(file_path_list):
        p_type = file.split("/")[-1].split("-temp1.0")[0]
        perturbed_type = get_perturbed_type(p_type)
        
        with open(file, "r") as f:
            data = [json.loads(line) for line in f]
        for d in data:
            try:
                doc = prepare_doc(d, testset)
                doc['id'] = f"{doc['id']}_{p_type}"
                doc['perturbation_type'] = perturbed_type
                doc['user'] = next(user_generator)
                final_data.append(doc)
            except Exception as e:
                error_data.append({
                    "file_name": file,
                    "document": d,
                    "error": traceback.format_exception(type(e), e, e.__traceback__)
                })
                continue
    #if debug mode, then write to dummy file
    if args.debug:
        with open("final_data.json", "w") as f:
            json.dump(final_data, f)
    else:
        push_to_firebase(final_data)
        
    if error_data:
        print(f"Number of errors: {len(error_data)}")
        with open("error_data.json", "w") as f:
            json.dump(error_data, f)
            
            
    
if __name__ == '__main__':
    args = parse_args()
    main(args)