import os
import json
import argparse
import torch
from transformers import AutoTokenizer
from utils.data_utils import load_and_prepare_dataset, create_dataloaders, set_seed, prepare_sample_data
from models.model import BrahmaForCausalLM, BrahmaConfig
from training.trainer import BrahmaTrainer
from utils.tokenizer import BrahmaTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Brahma LLM model")
    parser.add_argument("--config", type=str, default="config/default_config.json", help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="brahma-llm", help="W&B project name")
    parser.add_argument("--create_sample_data", action="store_true", help="Create sample data before training")
    parser.add_argument("--sample_data_size", type=int, default=100, help="Number of sample data points to create")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create directories
    os.makedirs(config["training_config"]["output_dir"], exist_ok=True)
    
    # Create sample data if requested
    if args.create_sample_data:
        print("Creating sample data...")
        prepare_sample_data(
            config["data_config"]["dataset_path"],
            num_samples=args.sample_data_size,
        )
    
    # Initialize tokenizer
    if config["tokenizer_config"]["tokenizer_name_or_path"]:
        # Use pretrained tokenizer from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_config"]["tokenizer_name_or_path"])
        
        # Make sure special tokens are set
        special_tokens = config["tokenizer_config"]["special_tokens"]
        special_tokens_dict = {k: v for k, v in special_tokens.items() if hasattr(tokenizer, k)}
        tokenizer.add_special_tokens(special_tokens_dict)
    else:
        # Use custom tokenizer (for demonstration)
        tokenizer = BrahmaTokenizer(
            vocab_size=config["tokenizer_config"]["vocab_size"],
            **config["tokenizer_config"]["special_tokens"]
        )
        
        # If there's some training data, we could train the tokenizer
        # But for simplicity, we'll just use the pre-initialized one
    
    # Initialize model configuration
    model_config = BrahmaConfig.from_dict(config["model_config"])
    
    # Update model vocab size to match tokenizer if needed
    if hasattr(tokenizer, "vocab_size"):
        model_config.vocab_size = tokenizer.vocab_size
    
    # Initialize model
    model = BrahmaForCausalLM(model_config)
    
    # Print model info
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load and prepare dataset
    print("Loading dataset...")
    tokenized_datasets = load_and_prepare_dataset(
        dataset_name_or_path=config["data_config"]["dataset_path"],
        tokenizer=tokenizer,
        text_column=config["data_config"]["text_column"],
        train_split=config["data_config"]["train_split"],
        validation_split=config["data_config"]["validation_split"],
        max_length=config["data_config"]["max_length"],
        preprocessing_num_workers=config["data_config"]["preprocessing_num_workers"],
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = create_dataloaders(
        tokenized_datasets=tokenized_datasets,
        tokenizer=tokenizer,
        batch_size=config["training_config"]["batch_size"],
        eval_batch_size=config["training_config"]["eval_batch_size"],
    )
    
    # Initialize trainer
    trainer = BrahmaTrainer(
        model=model,
        train_dataloader=dataloaders["train"],
        eval_dataloader=dataloaders.get("validation"),
        learning_rate=config["training_config"]["learning_rate"],
        weight_decay=config["training_config"]["weight_decay"],
        warmup_steps=config["training_config"]["warmup_steps"],
        max_steps=config["training_config"]["max_steps"],
        gradient_accumulation_steps=config["training_config"]["gradient_accumulation_steps"],
        gradient_clipping=config["training_config"]["gradient_clipping"],
        mixed_precision=config["training_config"]["mixed_precision"],
        log_interval=config["training_config"]["log_interval"],
        eval_interval=config["training_config"]["eval_interval"],
        save_interval=config["training_config"]["save_interval"],
        save_dir=config["training_config"]["output_dir"],
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
    
    # Resume from checkpoint if provided
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # Save tokenizer
    tokenizer.save_pretrained(config["training_config"]["output_dir"])
    
    # Save configuration
    with open(os.path.join(config["training_config"]["output_dir"], "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("Training complete!")

if __name__ == "__main__":
    main()
