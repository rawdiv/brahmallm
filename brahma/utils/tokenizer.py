import os
import json
import torch
from typing import List, Dict, Optional, Union, Tuple
from transformers import PreTrainedTokenizer, AddedToken

class BrahmaTokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer for Brahma LLM.
    
    This is a simple byte-level tokenizer. For a real-world project, you might want to:
    1. Use sentencepiece or HuggingFace's tokenizers library
    2. Train a proper BPE/WordPiece/Unigram tokenizer on your corpus
    3. Or use an existing pretrained tokenizer
    
    This implementation is for demonstrative purposes to show how tokenization works.
    """
    
    vocab_files_names = {"vocab_file": "vocab.json"}
    
    def __init__(
        self,
        vocab_file=None,
        vocab_size=32000,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        **kwargs
    ):
        """
        Initialize a BrahmaTokenizer.
        
        Args:
            vocab_file: Path to vocabulary file
            vocab_size: Size of vocabulary
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            unk_token: Unknown token
            pad_token: Padding token
        """
        self.vocab_size = vocab_size
        
        # Load or initialize vocabulary
        if vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            
            # Make sure required special tokens are in vocab
            for token in [bos_token, eos_token, unk_token, pad_token]:
                if token not in vocab:
                    vocab[token] = len(vocab)
        else:
            # Initialize a basic vocabulary with special tokens and byte tokens
            vocab = {
                pad_token: 0,
                bos_token: 1,
                eos_token: 2,
                unk_token: 3,
            }
            
            # Add byte tokens (ASCII + common Unicode)
            for i in range(256):
                token = chr(i)
                if token not in vocab:
                    vocab[token] = len(vocab)
        
        # Create inverse vocabulary (id -> token)
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        # Initialize with the vocabulary
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs
        )
        
        # Set vocabulary
        self.vocab = vocab
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens."""
        # Simple character-level tokenization for demonstration
        return list(text)
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to vocabulary id."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert vocabulary id to token."""
        return self.inv_vocab.get(index, self.unk_token)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens to a string."""
        return ''.join(tokens)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save vocabulary to file."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load tokenizer from pretrained model."""
        # Check if pretrained_model_name_or_path is a directory
        if os.path.isdir(pretrained_model_name_or_path):
            vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
            if os.path.exists(vocab_file):
                return cls(vocab_file=vocab_file, *args, **kwargs)
        
        # If not a directory or vocab file not found, initialize new tokenizer
        return cls(*args, **kwargs)
    
    def train_from_texts(self, texts: List[str], vocab_size: int = 32000, save_path: Optional[str] = None):
        """
        Train tokenizer from texts (simplified byte-level implementation).
        
        In a real implementation, you would:
        1. Use a proper tokenization algorithm (BPE, WordPiece, Unigram)
        2. Count token frequencies
        3. Merge most frequent pairs
        4. Save trained vocabulary
        
        Args:
            texts: List of text samples
            vocab_size: Target vocabulary size
            save_path: Path to save vocabulary
        """
        # Initialize vocabulary with special tokens
        vocab = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }
        
        # Count character frequencies
        char_counts = {}
        for text in texts:
            for char in text:
                if char not in char_counts:
                    char_counts[char] = 0
                char_counts[char] += 1
        
        # Sort characters by frequency
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add most frequent characters to vocabulary up to vocab_size
        for char, _ in sorted_chars:
            if char not in vocab and len(vocab) < vocab_size:
                vocab[char] = len(vocab)
        
        # Update tokenizer's vocabulary
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        # Save vocabulary
        if save_path:
            self.save_vocabulary(save_path)
        
        return self
