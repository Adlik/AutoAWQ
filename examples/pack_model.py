import os
import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="quant_model")
    parser.add_argument("--group-size", type=int, default=128)
    
    args = parser.parse_args()

    quant_config = { "zero_point": True, "q_group_size": args.group_size, "w_bit": 4, "version": "GEMM", "quant_method": "awq_marlin"}

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config, only_pack=True)

    # Save quantized model
    model.save_quantized(args.output_path, export_compatible=False)
    tokenizer.save_pretrained(args.output_path)
    os.system(f'cp {args.model_path}/tokenizer_config.json  {args.output_path}')  # add chat_template in tokenizer_config.json

    print(f'Model is quantized and saved at "{args.output_path}"')
