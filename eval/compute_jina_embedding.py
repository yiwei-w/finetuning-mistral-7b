import pandas as pd
from transformers import AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from tqdm import tqdm
import numpy as np

# trust_remote_code is needed to use the encode method
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) 

df = pd.read_json('eval_results_with_gpt_indented_fixed.json', orient='records')

def avg_cos_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    kernel = cosine_similarity(a, b)
    return np.diag(kernel).mean()


ref_embeddings = []
gpt_embeddings = []
before_embeddings = []
after_embeddings = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    ref_solution = row['ref_solution']
    gpt_solution = row['gpt-3.5-turbo_solution']
    before_solution = row['before_finetune_solution']
    after_solution = row['after_finetune_solution_72']
    ref_embeddings.append(model.encode(ref_solution, max_length=4096))
    gpt_embeddings.append(model.encode(gpt_solution, max_length=4096))
    before_embeddings.append(model.encode(before_solution, max_length=4096))
    after_embeddings.append(model.encode(after_solution, max_length=4096))

df['ref_embeddings'] = ref_embeddings
df['gpt_embeddings'] = gpt_embeddings
df['before_embeddings'] = before_embeddings
df['after_embeddings'] = after_embeddings

# Save embeddings
df.to_json('eval_results_with_jina_embeddings.json', orient='records', indent=4)

print("Avg GPT-3.5-turbo cosine similarity: ", avg_cos_sim(ref_embeddings, gpt_embeddings))
print("Avg Before FT cosine similarity: ", avg_cos_sim(ref_embeddings, before_embeddings))
print("Avg After FT cosine similarity: ", avg_cos_sim(ref_embeddings, after_embeddings))