import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from voyageai import get_embedding
from numpy.linalg import norm
import numpy as np

model = "voyage-lite-01-instruct"

df = pd.read_json(f'eval_results_with_{model}_embeddings.json', orient='records')

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
    ref_embeddings.append(row['ref_embeddings'])
    gpt_embeddings.append(row['gpt_embeddings'])
    before_embeddings.append(row['before_embeddings'])
    after_embeddings.append(row['after_embeddings'])



print("Avg GPT-3.5-turbo cosine similarity: ", avg_cos_sim(ref_embeddings, gpt_embeddings))
print("Avg Before FT cosine similarity: ", avg_cos_sim(ref_embeddings, before_embeddings))
print("Avg After FT cosine similarity: ", avg_cos_sim(ref_embeddings, after_embeddings))
# print("="*80)
# print("PREV Avg GPT-3.5-turbo cosine similarity: ", cosine_similarity(ref_embeddings, gpt_embeddings).mean())
# print("PREV Avg Before FT cosine similarity: ", cosine_similarity(ref_embeddings, before_embeddings).mean())
# print("PREV Avg After FT cosine similarity: ", cosine_similarity(ref_embeddings, after_embeddings).mean())



    
