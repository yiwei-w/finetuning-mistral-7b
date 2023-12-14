## Data Preparation
In the `data_preparation` folder, we provide the following scripts to prepare and clean the data for fine-tuning:
- `pretokenize.py`: Since we would like to use the whole context (8192 tokens) during fine-tuning, the default sample packing data pipeline in Axolotl will not work. Therefore, we implemented a custom script `data_preparation/pretokenize.py` that can be used to tokenize the dataset, concatenate the tokens with the EOS token in-between, chunk the tokens in to slices of length 8192, and save them (together with attention masks) in `.jsonl` format. Another tokenization detail is that we use the EOS token `</s>` as padding token, following Mistral's convention. Also, the pretokenization script pad to left by default as Mistral's implementation of Flash Attention assumes left-padded inputs.
- `ocr_script.py`: This script is used to fo the fist pass of OCR using [Nougat](https://github.com/facebookresearch/nougat). The output is in `.mmd` format (Mathpix Markdown). Note that it also saves a log during the OCR process, which can be used to track the bad pages that are skipped and any other OCR errors.
- `fill_missing_mathpix.py`: This script is used to do the second pass of OCR using Mathpix OCR API. It takes all skipped pages from the Nougat output, process them one by one using Mathpix OCR API, and re-insert them to the original location. Note that you need to have a Mathpix OCR API key to run this script.
- `clean_data.py`: This script is used to clean the data. It takes the output from the previous step, replace some special LaTeX syntax not supported by [KaTeX](https://github.com/KaTeX/KaTeX) as well as some format errors during OCR, and save the output to `.md` files
- `katex_errors/`: This folder contains some artifcats and scripts we used to get a sense of the KaTeX rendering errors before cleaning the data.

## Fine-tuning
For fine-tuning, we used the [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) library. After installing the library, you can run the following command to fine-tune the model on your own dataset:
```
accelerate launch -m axolotl.cli.train finetuning/finetuning_config.yaml --deepspeed deepspeed/zero2.json
```
The fine-tuning configuration file `finetuning_config.yaml` is located in the `finetuning` folder that covers all the training hyperparameters and setup.

## Evaluation
The `eval/` folder contains several scripts and files for evaluation:
- `eval_script_before.py`: This script is used to generate model's solutions on the benchmark before fine-tuning. It takes the problem and background pair from the evaluation dataset, format them into the proper prompt, and then generate the solution using [vLLM](https://github.com/vllm-project/vllm) for fast inference with Mistral 7B models. The results are saved to a `.json` file.
- `eval_script_after.py`: This script is used to generate model's solutions on the benchmark after fine-tuning. First, it loads the relevant model checkpoint. Then, it takes the problem and background pair from the evaluation dataset, format them into the proper prompt, and then generate the solution using [vLLM](https://github.com/vllm-project/vllm). The results are saved to a `.json` file.
- `gpt_eval_script.py`: This script is used to gather GPT-3.5's solutions to the questions in the benchmark. It takes the problem and background pair from the evaluation dataset, format them into the proper prompt, and then generate the solutions with OpenAI's API. The results are saved to a `.json` file.
- `gpt_system_prompt.txt`: This file contains the system message appended to the beginning of every conversation with GPT-3.5.
- `system_prompt.txt`: This file contains the zero-shot CoT instructions to solve QFT problems using background knowledge for both Mistral 7B and GPT-3.5.
- `compute_bertscore.py`: This script is used to compute the BERTScore (F1, recall, precision) using [allenai/longformer-large-4096-finetuned-triviaqa](https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa) model.
- `compute_ROUGE.py`: This script is used to compute the ROUGE-L, ROUGE-1, and ROUGE-2 scores with Mistral tokenizer.
- `compute_voyage_embedding`: This script is used to compute the and save the document-level embeddings based on `voyage-lite-01-instruct` model.
- `compute_jina_embedding`: This script is used to compute and save the document-level embeddings based on [jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) model.
- `verify_jina.py`: This script is used to calculate the average cosine similarity for Jina embeddings locally.
- `verify_voyage.py`: This script is used to calculate the average cosine similarity for Voyage embeddings locally.


