import os
import sys
import json
import argparse
import torch
from transformers import AutoTokenizer
from inference.generator import BrahmaGenerator
from models.model import BrahmaConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tokenizer import BrahmaTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with Brahma LLM model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, default=None, help="Path to model config")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate from")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Penalty for repeating tokens")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling (if false, uses greedy decoding)")
    parser.add_argument("--output_file", type=str, default=None, help="File to write output to (if not specified, prints to console)")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load tokenizer
    if args.tokenizer_path:
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = os.path.dirname(args.model_path)
    
    try:
        # Try to load from HuggingFace tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:
        # Fall back to custom tokenizer
        tokenizer = BrahmaTokenizer.from_pretrained(tokenizer_path)
    
    # Load config if provided
    config = None
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
            if "model_config" in config_dict:
                config = BrahmaConfig.from_dict(config_dict["model_config"])
            else:
                config = BrahmaConfig.from_dict(config_dict)
    
    # Initialize generator
    generator = BrahmaGenerator(
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tokenizer=tokenizer,
        config=config,
    )
    
    # Generate text
    print(f"Generating text from prompt: {args.prompt}")
    print(f"Model: {args.model_path}")
    print(f"Parameters: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    
    generated_texts = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        num_return_sequences=args.num_return_sequences,
    )
    
    # Print or save output
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for i, text in enumerate(generated_texts):
                f.write(f"=== Generated Text {i+1} ===\n")
                f.write(text)
                f.write("\n\n")
        print(f"Generated text saved to {args.output_file}")
    else:
        for i, text in enumerate(generated_texts):
            print(f"\n=== Generated Text {i+1} ===")
            print(text)
            print("\n")

if __name__ == "__main__":
    main()
