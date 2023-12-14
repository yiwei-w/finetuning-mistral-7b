from bert_score import BERTScorer
import pandas as pd

df = pd.read_json('eval_results_with_gpt_indented_fixed.json', orient='records')

model = "allenai/longformer-large-4096-finetuned-triviaqa"
scorer = BERTScorer(model_type=model, lang="en")

gpt_solutions = df['gpt-3.5-turbo_solution'].tolist()
before_solutions = df['before_finetune_solution'].tolist()
after_solutions = df['after_finetune_solution_72'].tolist()
ref_solutions = df['ref_solution'].tolist()

gpt_scores = scorer.score(cands=gpt_solutions, refs=ref_solutions)
before_scores = scorer.score(cands=before_solutions, refs=ref_solutions)
after_scores = scorer.score(cands=after_solutions, refs=ref_solutions)

print("GPT-3.5-Turbo Scores:")
P, R, F1 = gpt_scores
print(f"Precision: {P.mean():.4f}")
print(f"Recall: {R.mean():.4f}")
print(f"F1: {F1.mean():.4f}")

print("="*50)
print("Before Finetuning Scores:")
P, R, F1 = before_scores
print(f"Precision: {P.mean():.4f}")
print(f"Recall: {R.mean():.4f}")
print(f"F1: {F1.mean():.4f}")

print("="*50)
print("After Finetuning Scores:")
P, R, F1 = after_scores
print(f"Precision: {P.mean():.4f}")
print(f"Recall: {R.mean():.4f}")
print(f"F1: {F1.mean():.4f}")

