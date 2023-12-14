import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from voyageai import get_embedding
from numpy.linalg import norm
import numpy as np

model = "voyage-lite-01-instruct"

df = pd.read_json('eval_results.json', orient='records')

def avg_cos_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    kernel = cosine_similarity(a, b)
    return np.diag(kernel).mean()

ref_embeddings = []
gpt_embeddings = []
before_embeddings = []
after_embeddings = []
for _, row in df.iterrows():
    ref_solution = row['ref_solution']
    gpt_solution = row['gpt-3.5-turbo_solution']
    before_solution = row['before_finetune_solution']
    after_solution = row['after_finetune_solution_72']
    ref_embeddings.append(get_embedding(ref_solution, model, input_type="document"))
    gpt_embeddings.append(get_embedding(gpt_solution, model, input_type="document"))
    before_embeddings.append(get_embedding(before_solution, model, input_type="document"))
    after_embeddings.append(get_embedding(after_solution, model, input_type="document"))


df['ref_embeddings'] = ref_embeddings
df['gpt_embeddings'] = gpt_embeddings
df['before_embeddings'] = before_embeddings
df['after_embeddings'] = after_embeddings

# Save embeddings
df.to_json('eval_results_with_voyage-01-lite-instruct_embeddings.json', orient='records', indent=4)


print("Avg GPT-3.5-turbo cosine similarity: ", avg_cos_sim(ref_embeddings, gpt_embeddings))
print("Avg Before FT cosine similarity: ", avg_cos_sim(ref_embeddings, before_embeddings))
print("Avg After FT cosine similarity: ", avg_cos_sim(ref_embeddings, after_embeddings))



    
