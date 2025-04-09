import os
import json
import torch
import random
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_and_prepare_dataset(
    dataset_name_or_path: str,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "text",
    train_split: str = "train",
    validation_split: Optional[str] = "validation",
    test_split: Optional[str] = None,
    max_length: int = 2048,
    preprocessing_num_workers: int = 4,
    streaming: bool = False,
):
    """
    Load and prepare dataset for training.
    
    Args:
        dataset_name_or_path: HuggingFace dataset name or path to local dataset
        tokenizer: Tokenizer to use for tokenization
        text_column: Name of the column containing the text
        train_split: Name of the training split
        validation_split: Name of the validation split
        test_split: Name of the test split
        max_length: Maximum sequence length
        preprocessing_num_workers: Number of workers for preprocessing
        streaming: Whether to stream the dataset
    
    Returns:
        Tuple of tokenized datasets: train, validation, test (if provided)
    """
    # Load dataset
    if os.path.isdir(dataset_name_or_path):
        # Load from local jsonl files
        data_files = {}
        
        if os.path.exists(os.path.join(dataset_name_or_path, f"{train_split}.jsonl")):
            data_files[train_split] = os.path.join(dataset_name_or_path, f"{train_split}.jsonl")
        
        if validation_split and os.path.exists(os.path.join(dataset_name_or_path, f"{validation_split}.jsonl")):
            data_files[validation_split] = os.path.join(dataset_name_or_path, f"{validation_split}.jsonl")
        
        if test_split and os.path.exists(os.path.join(dataset_name_or_path, f"{test_split}.jsonl")):
            data_files[test_split] = os.path.join(dataset_name_or_path, f"{test_split}.jsonl")
        
        dataset = load_dataset("json", data_files=data_files, streaming=streaming)
    else:
        # Load from HuggingFace datasets
        dataset = load_dataset(dataset_name_or_path, streaming=streaming)
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
    
    # Tokenize datasets
    tokenized_datasets = {}
    
    if train_split in dataset:
        tokenized_datasets["train"] = dataset[train_split].map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers if not streaming else None,
            remove_columns=[col for col in dataset[train_split].column_names if col != text_column],
        )
    
    if validation_split and validation_split in dataset:
        tokenized_datasets["validation"] = dataset[validation_split].map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers if not streaming else None,
            remove_columns=[col for col in dataset[validation_split].column_names if col != text_column],
        )
    
    if test_split and test_split in dataset:
        tokenized_datasets["test"] = dataset[test_split].map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers if not streaming else None,
            remove_columns=[col for col in dataset[test_split].column_names if col != text_column],
        )
    
    return tokenized_datasets

def create_data_collator(tokenizer, padding_side="right", pad_to_multiple_of=8):
    """
    Create a data collator for language modeling.
    
    Args:
        tokenizer: Tokenizer to use for padding
        padding_side: Side to pad on
        pad_to_multiple_of: Pad to multiple of this value
    
    Returns:
        Data collator function
    """
    # Save original padding side
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    
    def collate_fn(examples):
        # Tokenize if needed
        if isinstance(examples[0], dict):
            examples = [example["input_ids"] for example in examples]
        
        # Create attention masks and pad
        batch = tokenizer.pad(
            {"input_ids": examples},
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Create labels
        batch["labels"] = batch["input_ids"].clone()
        
        return batch
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    return collate_fn

def create_dataloaders(
    tokenized_datasets,
    tokenizer,
    batch_size=8,
    eval_batch_size=None,
    shuffle=True,
    num_workers=4,
):
    """
    Create DataLoaders for training and evaluation.
    
    Args:
        tokenized_datasets: Dictionary of tokenized datasets
        tokenizer: Tokenizer to use for padding
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        shuffle: Whether to shuffle the training data
        num_workers: Number of workers for DataLoader
    
    Returns:
        Dictionary of DataLoaders
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    # Create data collator
    data_collator = create_data_collator(tokenizer)
    
    # Create DataLoaders
    dataloaders = {}
    
    if "train" in tokenized_datasets:
        dataloaders["train"] = DataLoader(
            tokenized_datasets["train"],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=data_collator,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    if "validation" in tokenized_datasets:
        dataloaders["validation"] = DataLoader(
            tokenized_datasets["validation"],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    if "test" in tokenized_datasets:
        dataloaders["test"] = DataLoader(
            tokenized_datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return dataloaders

def prepare_sample_data(output_dir, num_samples=100, seq_length=1024, vocab_size=32000):
    """
    Create sample data for testing training pipeline.
    
    Args:
        output_dir: Directory to save sample data
        num_samples: Number of samples to generate
        seq_length: Length of each sequence
        vocab_size: Size of vocabulary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create random text samples
    train_samples = []
    for _ in range(num_samples):
        tokens = torch.randint(0, vocab_size, (seq_length,)).tolist()
        train_samples.append({"text": " ".join(map(str, tokens))})
    
    val_samples = []
    for _ in range(num_samples // 10):
        tokens = torch.randint(0, vocab_size, (seq_length,)).tolist()
        val_samples.append({"text": " ".join(map(str, tokens))})
    
    # Save to jsonl files
    with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    
    with open(os.path.join(output_dir, "validation.jsonl"), "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Created sample data in {output_dir} with {num_samples} training and {num_samples // 10} validation samples")
