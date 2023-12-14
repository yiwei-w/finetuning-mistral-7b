import os
import json
import re
import pandas as pd
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Load Mistral tokenizer
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Load system prompt
with open('system_prompt.txt', 'r', encoding='utf-8') as file:
    system_prompt = file.read()

def list_files(directory):
    """ List all files in a given directory """
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def get_file_content(filename):
    """ Read the content of a file and clean up LaTeX math """
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        return content

def remove_latex_math_newline(text):
    """ Remove redundant newlines around LaTeX math delimiters """
    text = re.sub(r'\n\\\[', r'\[', text)    # Remove newline before '\['
    text = re.sub(r'\\\]\n', r'\]', text)    # Remove newline after '\]'
    return text

def mistral_tok(text):
    """ Tokenize text for Mistral instruct model (without EOS/BOS tokens) """
    return mistral_tokenizer.encode(text, add_special_tokens=False)

def prepare_mistral_prompt_tokens(problem, background):
    """ Prepare prompt tokens for Mistral instruct model """
    prompt = f"\n{system_prompt}\n\n## Problem:\n{problem}\n\n## Background:\n{background}\n"
    return [mistral_tokenizer.bos_token_id] + mistral_tok("[INST]") + \
           mistral_tok(prompt) + mistral_tok("[/INST]")

   

problems_folder = 'problems'
solutions_folder = 'ref_solutions'
background_folder = 'background'

problems = list_files(problems_folder)
data_rows = []

for problem_file in problems:
    # Extract section_id and problem_id
    problem_filename = Path(problem_file).stem
    section_id, problem_id = problem_filename.split('_')[:2]
    
    # Construct solution and background file names
    solution_file = f"{section_id}_{problem_id}_sol.txt"
    background_file = f"section_{section_id}.txt"

    # Read file contents
    problem_content = get_file_content(os.path.join(problems_folder, problem_file))
    solution_content = get_file_content(os.path.join(solutions_folder, solution_file))
    background_content = get_file_content(os.path.join(background_folder, background_file))

    problem_content = remove_latex_math_newline(problem_content)
    solution_content = remove_latex_math_newline(solution_content)
    background_content = remove_latex_math_newline(background_content)

    # Append data row
    data_rows.append({
        'section_id': section_id,
        'problem_id': problem_id,
        'problem': problem_content,
        'ref_solution': solution_content,
        'background': background_content
    })

# Create DataFrame
df = pd.read_json('eval_results_after.json', orient='records')


# sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
checkpoint_id = 72

after_finetune = LLM(model=f"/workspace/axolotl/finetuned-out/checkpoint-{checkpoint_id}", tensor_parallel_size=2)
prompt_tokens = [prepare_mistral_prompt_tokens(problem, background) for problem, background in zip(df['problem'], df['background'])]
after_finetune_outputs = after_finetune.generate(prompt_token_ids=prompt_tokens, sampling_params=sampling_params)
after_finetune_solutions = [output.outputs[0].text for output in after_finetune_outputs]
df[f'after_finetune_solution_{checkpoint_id}_fixed'] = after_finetune_solutions

df.to_json('eval_results_after.json', orient='records', indent=4)







