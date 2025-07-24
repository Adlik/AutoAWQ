import os
import json
import argparse
from datasets import load_dataset, load_from_disk
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import random


def get_common_calib_data(dataset_path, tokenizer, nsamples=512, max_seq_len=8192, seed=42):
    res_list = []
    print(f'using json to load data from {dataset_path}')
    f = open(dataset_path)
    dataset = json.load(f)
    random.seed(seed)
    random.shuffle(dataset)

    n_runs = 0
    for data in dataset:
        data = data["text"] if isinstance(data, dict) else data
        enc_data = tokenizer(data)
        if len(enc_data["input_ids"]) > max_seq_len:
            continue
        res_list.append(data)
        n_runs += 1
        if n_runs == nsamples:
            break

    print(f'sample length: {len(res_list)}')
    return res_list


def get_calib_data(dataset_path, tokenizer, nsamples=512, max_seq_len=8192, seed=42):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    # dataset = load_from_disk(dataset_path)
    print(f'using datasets to load data from {dataset_path}')
    dataset = dataset.shuffle(seed=seed)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer(line)
        if len(line_encoded["input_ids"]) > max_seq_len:
            continue
        samples.append(data["text"])
        n_run += 1
        if n_run == nsamples:
            break
    print(f'sample length: {len(samples)}')
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="quant_model")
    parser.add_argument("--dataset-path", type=str, default="val.jsonl.zst")
    parser.add_argument("--use-datasets", action="store_true")
    parser.add_argument("--save-quant", action="store_true")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

# model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
# quant_path = 'mistral-instruct-v0.2-awq'
    quant_config = { "zero_point": True, "q_group_size": args.group_size, "w_bit": 4, "version": "GEMM" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.use_datasets:
        print('using load_datasets')
        examples = get_calib_data(args.dataset_path, tokenizer, nsamples=args.max_samples,
                                  max_seq_len=args.max_seq_len, seed=args.seed)
    else:
        examples = get_common_calib_data(args.dataset_path, tokenizer, nsamples=args.max_samples,
                                         max_seq_len=args.max_seq_len, seed=args.seed)

    print(f'max_samples: {args.max_samples}, max_seq_len: {args.max_seq_len}')
    # Quantize
    model.quantize(tokenizer, quant_config=quant_config, export_compatible=not args.save_quant, calib_data=examples, max_calib_samples=args.max_samples, max_calib_seq_len=args.max_seq_len)

    # Save quantized model
    model.save_quantized(args.output_path, export_compatible=not args.save_quant)
    tokenizer.save_pretrained(args.output_path)
    os.system(f'cp {args.model_path}/tokenizer_config.json  {args.output_path}')  # add chat_template in tokenizer_config.json

    print(f'Model is quantized and saved at "{args.output_path}"')
