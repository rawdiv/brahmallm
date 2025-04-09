import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import BrahmaForCausalLM, BrahmaConfig
from inference.generator import BrahmaGenerator
from utils.tokenizer import BrahmaTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Brahma LLM model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, default=None, help="Path to model config")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--eval_dataset", type=str, required=True, 
                        help="Path to evaluation dataset (jsonl or HF dataset ID)")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name for prompts")
    parser.add_argument("--reference_column", type=str, default="completion", help="Column name for reference completions")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", 
                        help="File to write evaluation results to")
    return parser.parse_args()

def evaluate_perplexity(model, tokenizer, dataset, batch_size=4, device="cuda"):
    """Evaluate perplexity on dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating perplexity"):
            batch = dataset[i:min(i + batch_size, len(dataset))]
            
            # Tokenize inputs
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            
            # Get model outputs
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=inputs["input_ids"],
            )
            
            loss = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Calculate total loss and tokens
            total_loss += loss.item() * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    
    return perplexity.item()

def evaluate_generation(generator, dataset, prompt_column, reference_column, 
                        max_new_tokens=128, temperature=0.7, top_k=50, top_p=0.9):
    """Evaluate text generation quality."""
    results = []
    
    for item in tqdm(dataset, desc="Generating completions"):
        prompt = item[prompt_column]
        reference = item[reference_column]
        
        # Generate text
        generated = generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
        )[0]
        
        # Extract the generated part (not including prompt)
        if prompt in generated:
            generated_text = generated[len(prompt):].strip()
        else:
            generated_text = generated.strip()
        
        # Add to results
        results.append({
            "prompt": prompt,
            "reference": reference,
            "generated": generated_text,
        })
    
    return results

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location="cpu")
    
    if config is None:
        # Try to get config from checkpoint
        if "config" in checkpoint:
            config = checkpoint["config"]
            if isinstance(config, dict):
                config = BrahmaConfig.from_dict(config)
        else:
            config = BrahmaConfig()
    
    model = BrahmaForCausalLM(config)
    
    # Load model weights
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    # Initialize generator
    generator = BrahmaGenerator(
        model_path=args.model_path,
        device=device,
        tokenizer=tokenizer,
        config=config,
    )
    
    # Load dataset
    if os.path.isfile(args.eval_dataset) or os.path.isdir(args.eval_dataset):
        # Local file or directory
        dataset = load_dataset("json", data_files=args.eval_dataset)["train"]
    else:
        # HuggingFace dataset
        dataset = load_dataset(args.eval_dataset)["validation" if "validation" in dataset else "test"]
    
    print(f"Loaded evaluation dataset with {len(dataset)} examples")
    
    # Evaluate perplexity on the entire texts
    if args.reference_column in dataset.column_names:
        texts = dataset[args.reference_column]
        perplexity = evaluate_perplexity(model, tokenizer, texts, args.batch_size, device)
        print(f"Perplexity: {perplexity:.2f}")
    else:
        perplexity = None
        print("Skipping perplexity evaluation: reference column not found")
    
    # Evaluate generation
    if args.prompt_column in dataset.column_names and args.reference_column in dataset.column_names:
        generation_results = evaluate_generation(
            generator=generator,
            dataset=dataset,
            prompt_column=args.prompt_column,
            reference_column=args.reference_column,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        
        print(f"Generated {len(generation_results)} completions")
    else:
        generation_results = []
        print("Skipping generation evaluation: prompt or reference column not found")
    
    # Save results
    results = {
        "model_path": args.model_path,
        "dataset": args.eval_dataset,
        "perplexity": perplexity,
        "generation_samples": generation_results[:10],  # Save first 10 examples
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        }
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {args.output_file}")

if __name__ == "__main__":
    main()
