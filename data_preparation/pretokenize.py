import os
import random
import jsonlines
from tqdm import tqdm
#from transformers import LlamaTokenizer, CodeLlamaTokenizer
from transformers import AutoTokenizer

# random.seed(42)
max_seq_length = 8192

# Reading the MD Files
def read_md_files(folder_path):
    md_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]
    contents = []
    for f in tqdm(md_files, desc="Reading MD files"):
        with open(os.path.join(folder_path, f), 'r', encoding='utf-8') as file:
            contents.append(file.read())
    return contents

# Shuffling and Tokenizing
def tokenize_and_concat(files_content, tokenizer):
    random.shuffle(files_content)
    eos_token = tokenizer.eos_token_id
    bos_token = tokenizer.bos_token_id
    print("eos token ", tokenizer.eos_token)
    print("eos token id ", tokenizer.eos_token_id)
    print("bos token ", tokenizer.bos_token)
    print("bos token id ", tokenizer.bos_token_id)
    tokens = []
    # content_list = tokenizer(files_content).input_ids
    for content in tqdm(files_content, desc="Tokenizing"):
        # no need to add bos_token as it is already added in the tokenizer
        tokens.extend(tokenizer.encode(content))
        tokens.append(eos_token)
    # for item in content_list:
    #     assert item[0] == bos_token
    #     assert item[-1] != eos_token
    #     tokens.extend(item)
    #     tokens.append(eos_token)
    print("TOTAL tokens length: ", len(tokens))
    return tokens

# Chunking
def chunk_tokens(tokens, tokenizer, pad_token_id, chunk_size=max_seq_length):
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    # If the last chunk is smaller than 8192, pad it
    last_chunk_length = len(chunks[-1])
    if last_chunk_length < chunk_size:
        padding_length = chunk_size - last_chunk_length
        print("padding length ", padding_length)
        #chunks[-1].extend([tokenizer.pad_token_id] * padding_length)
        # pad to the left
        chunks[-1] = [pad_token_id] * padding_length + chunks[-1]
    
    return chunks

# Saving to JSONL
def save_to_jsonl(chunks, output_file, tokenizer, pad_token_id):
    with jsonlines.open(output_file, mode='w') as writer:
        for chunk in tqdm(chunks[:-1], desc="Saving to JSONL"):
            attention_mask = [1] * max_seq_length
            writer.write({
                "input_ids": chunk,
                "attention_mask": attention_mask,
                "labels": chunk
            })
        # Last chunk may need padding
        last_chunk = chunks[-1]
        pad_token_count = last_chunk.count(pad_token_id)
        print("pad token count ", pad_token_count)
        print("last token ", tokenizer.decode(last_chunk[-1]))
        # if pad_token_count > 0:
        #     assert last_chunk[:pad_token_count] == [pad_token_id] * pad_token_count
        # pad to the left
        attention_mask = [0] * pad_token_count + [1] * (max_seq_length - pad_token_count)
        writer.write({
            "input_ids": last_chunk,
            "attention_mask": attention_mask,
            "labels": last_chunk
        })

def main():
    raw_data_folder_path = "./qft_data/complete_files_md"
    output_file = "pretokenized_finetuning_data.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    #tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    #tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-13b-hf")
    # We need to add pad_token manually as it is not in Mistral tokenizer
    # tokenizer.add_special_tokens({'pad_token': '<unk>'})
    pad_token_id = tokenizer.eos_token_id
    files_content = read_md_files(raw_data_folder_path)
    tokens = tokenize_and_concat(files_content, tokenizer)
    chunks = chunk_tokens(tokens, tokenizer, pad_token_id, chunk_size=max_seq_length)
    save_to_jsonl(chunks, output_file, tokenizer, pad_token_id)

if __name__ == "__main__":
    main()
