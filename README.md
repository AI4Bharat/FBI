# FBI: Finding Blindspots in LLM Evaluations with Interpretable Checklists

[📜 Paper](https://arxiv.org/abs/2406.13439) | [🤗 HF Dataset](https://huggingface.co/datasets/ai4bharat/fbi)

We present FBI, our novel meta-evaluation framework designed to assess the robustness of evaluator LLMs across diverse tasks and evaluation strategies. Please refer to our [paper]() and [blog]() for more details.

<p align="center" width="100%">
      <img src="assets/imgs/FBI.jpg" alt="We present FBI, our novel meta-evaluation framework designed to assess the robustness of evaluator LLMs across diverse tasks and evaluation strategies." style="width: 75%; min-width: 200px; display: block; margin: auto;">
</p>

## Setup

To run perturbation generation and evaluation, you need to install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

Setup required API keys to make the API calls.
```bash
export OPENAI_API_KEY=[ADD_YOUR_OPENAI_API_KEY]
export CLAUDE_API_KEY=[ADD_YOUR_CLAUDE_API_KEY]
export GEMINI_API_KEY=[ADD_YOUR_GEMINI_API_KEY]
export LLAMA3_API_KEY=[ADD_LLAMA3_API_KEY]
export LLAMA3_BASE_URL=[ADD_LLAMA3_BASE_URL]
```
We use hosted services for using Llama-3-70B-Instruct Model.

## Generate Perturbations

To generate perturbations, specify the desired task ability, model, and other parameters. This will create a `FILENAME.jsonl`, that can be used to create a [Batch API](https://platform.openai.com/docs/guides/batch) call to get the model outputs.

Sample perturbations across all the 22 categories are available at [here](assets/sample_perturbations/)!!!

Use the following command to generate the jsonl:
```bash
python -m perturbations.perturbations \
  --data_dir DATA_DIR \
  --file_name PATH_TO_PROMPTS_AND_ANSWERS \
  --subset TASK_ABILITY \
  --model MODEL_NAME \
  --temperature TEMP \
  --top_p TOP_P \
  --max_tokens MAX_TOKENS \
  --frequency_penalty FREQ_PEN \
  --debug
```

- `DATA_DIR`: Directory containing the data.
- `PATH_TO_PROMPTS_AND_ANSWERS`: Path to the file with prompts and answers.
- `TASK_ABILITY`: Specify the type of task (choose from `['factual', 'reasoning', 'instruction-following', 'long-form']`).
- `MODEL_NAME`: Name of the model to use.
- `TEMP`: Sampling temperature (controls the randomness of predictions).
- `TOP_P`: Top-p sampling parameter (controls the diversity of predictions).
- `MAX_TOKENS`: Maximum number of tokens to generate.
- `FREQ_PEN`: Frequency penalty (controls the repetition of tokens).
- `--debug`: Optional flag to enable debug mode.



Once you have the `.jsonl` file, create a batch request using the following command:

```bash
python batch_call.py \
  --create_batch \
  --input_file_name FILENAME.jsonl \
  --data_path DATA_DIR \
  --job_desc "JOB_DESC"
```

- `--create_batch`: Flag to indicate the creation of a batch request.
- `FILENAME.jsonl`: The name of the input file (in `.jsonl` format) containing the outputs.
- `DATA_DIR`: Path to the directory where the data is stored.
- `JOB_DESC`: Description of the job for the batch request.

## Run Evaluation
To run evaluations, we again create a batch jsonl file `FILENAME.jsonl` and use this for getting model outputs either using Batch API or regular API. We support the following LLM Evaluation strategies (please refer to the paper for more details on each strategy):
### Single Answer Evaluation
- **Vanilla***: Scoring the answer without any explanation. (`single_vanilla`)
- **Vanilla**: Scoring the answer by first generating the explanation. (`single_vanilla_cot`)
- **Rubric**: Scoring the answer with the given rubrics. (`single_rubrics`)
- **Axis**: Scoring the given answer along a specified axis. (`single_axes`)
- **Axis+Rubrics**: Scoring the given answer along a specified axis with the given rubrics. (`single_axes_rubrics`)

### Pairwise Comparison
- **Pairwise***: Choosing the best answer without any explanation. (`compare_vanilla`)
- **Pairwise**: Choosing the best answer by first generating the explanation. (`compare_vanilla_cot`)
- **Rules**: Choosing the best answer according to the given rules. (`compare_rules`)
- **Axis**: Choosing the best answer along the specified axis. (`compare_axes`)
- **Axis+Rules**: Choosing the best answer along the specified answer and according to the given rules. (`compare_axes_rules`)

### Reference-guided Evaluation
- Reference: Scoring the answer, given the reference answer by first generating an explanation. (`reference_based`)

Use the below command to generate the jsonl file:
```bash
python llm_evaluators/<EVAL_METHOD>.py \
  --file_name PATH_TO_TSV_FILE_HAVING_DATA \
  --out_file_name PATH_TO_OUTPUT_JSONL \
  --model MODEL_NAME \
  --all \
  --axes "cont_qual" "task_qual" \    
  --p_mode 
```
- `file_name`: The path of the input data file to be evaluated. (We support data in tsv file format only right now)
- `out_file_name`: The path of the output JSONL file where the jsonl will be saved.
- `model`: The name of the model to be used for evaluation. Choices: `['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo-0125', 'llama3-70b', 'claude3-opus', 'gemini-1.5-flash', 'gemini-1.5-pro']`
- `all`: Run all available metrics. This option cannot be used together with `--axes`. (Supported only in Axis, Axis+Rubrics and Axis+Rules evaluators only)
- `axes`: Run specific metrics. Provide a list of metric names to be evaluated. This option cannot be used together with `--all`. (Supported only in Axis, Axis+Rubrics and Axis+Rules evaluators only)
- `p_mode`: Run in perturbed-first mode i.e., exchange the positions of gold answers and perturbed answers. (Supported only in Pairwise Comparison evaluators only)

Once the `.jsonl` file is created, we can create a batch request (supported by only OpenAI models) using the command: 
```bash
python batch_call.py \
  --create_batch \
  --input_file_name FILENAME.jsonl \
  --data_path DATA_DIR \
  --job_desc "JOB_DESC"
```

- `--create_batch`: Flag to indicate the creation of a batch request.
- `FILENAME.jsonl`: The name of the input file (in `.jsonl` format) containing the outputs.
- `DATA_DIR`: Path to the directory where the data is stored.
- `JOB_DESC`: Description of the job for the batch request.

For other models and for running in regular API mode, use the command:
```bash
python parallel_call.py \
  --input_file_name FILENAME.jsonl \
  --output_file_name PATH_FOR_OUTPUT_JSONL \
  --n_jobs NUMBER_OF_PARALLEL_REQUESTS
```

- `input_file_name`: The path of the input file (in `.jsonl` format).
- `output_file_name`: The path of the output file (for storing the outputs)
- `n_jobs`: Number of parallel requests to send to the model API. 

## Citation

If you used this repository or our models, please cite our work:

```bibtex
@misc{doddapaneni2024finding,
      title={Finding Blind Spots in Evaluator LLMs with Interpretable Checklists}, 
      author={Doddapaneni Sumanth and Mohammed Safi Ur Rahman Khan and Sshubam Verma and Mitesh M. Khapra},
      year={2024},
      eprint={2406.13439},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
