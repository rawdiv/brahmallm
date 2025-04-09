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
    parser = argparse.ArgumentParser(description="Chat with Brahma LLM model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, default=None, help="Path to model config")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Penalty for repeating tokens")
    parser.add_argument("--system_message", type=str, default="You are Brahma, a helpful AI assistant.", 
                       help="System message to use")
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
    
    # Print welcome message
    print("\nWelcome to Brahma LLM Chat!")
    print("Type 'exit', 'quit', or press Ctrl+C to end the conversation.\n")
    
    # Initialize chat history
    messages = [{"role": "system", "content": args.system_message}]
    
    # Chat loop
    try:
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Check if user wants to exit
            if user_input.lower() in ["exit", "quit"]:
                print("\nThank you for chatting with Brahma LLM!")
                break
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            response = generator.chat(
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            
            # Print response
            print(f"\nBrahma: {response}\n")
            
            # Add assistant message to history
            messages.append({"role": "assistant", "content": response})
    
    except KeyboardInterrupt:
        print("\n\nThank you for chatting with Brahma LLM!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
