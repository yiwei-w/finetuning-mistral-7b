from rouge_score import rouge_scorer
import pandas as pd
from transformers import AutoTokenizer



# Load Mistral tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

df = pd.read_json('eval_results_with_gpt_indented_fixed.json', orient='records')

def avg_rouge_fmeasure(scores, metrics):
    report = {metric: sum([score[metric].fmeasure for score in scores]) / len(scores) for metric in metrics}
    return {m: round(report[m], 3) for m in metrics}

def avg_rouge_precision(scores, metrics):
    report = {metric: sum([score[metric].precision for score in scores]) / len(scores) for metric in metrics}
    return {m: round(report[m], 3) for m in metrics}

def avg_rouge_recall(scores, metrics):
    report = {metric: sum([score[metric].recall for score in scores]) / len(scores) for metric in metrics}
    return {m: round(report[m], 3) for m in metrics}

metrics = ['rouge1', 'rouge2', 'rougeL']
scorer = rouge_scorer.RougeScorer(metrics, tokenizer=tokenizer, use_stemmer=True)

gpt_scores = []
before_scores = []
after_scores = []
for _, row in df.iterrows():
    ref_solution = row['ref_solution']
    gpt_solution = row['gpt-3.5-turbo_solution']
    before_solution = row['before_finetune_solution']
    after_solution = row['after_finetune_solution_72']
    gpt_scores.append(scorer.score(target=ref_solution, prediction=gpt_solution))
    before_scores.append(scorer.score(target=ref_solution, prediction=before_solution))
    after_scores.append(scorer.score(target=ref_solution, prediction=after_solution))

print("GPT-3.5-Turbo Solutions:")

print("F measure: ", avg_rouge_fmeasure(gpt_scores, metrics))
print("Precision: ", avg_rouge_precision(gpt_scores, metrics))
print("Recall: ", avg_rouge_recall(gpt_scores, metrics))
print("="*50)
print("Before Finetuning:")
print("F measure: ", avg_rouge_fmeasure(before_scores, metrics))
print("Precision: ", avg_rouge_precision(before_scores, metrics))
print("Recall: ", avg_rouge_recall(before_scores, metrics))
print("="*50)
print("After Finetuning:")
print("F measure: ", avg_rouge_fmeasure(after_scores, metrics))
print("Precision: ", avg_rouge_precision(after_scores, metrics))
print("Recall: ", avg_rouge_recall(after_scores, metrics))