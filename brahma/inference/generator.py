import os
import sys
import torch
import logging
from typing import List, Dict, Optional, Union, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import BrahmaForCausalLM, BrahmaConfig

class BrahmaGenerator:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer=None,
        max_length: int = 2048,
        config: Optional[Union[BrahmaConfig, Dict[str, Any]]] = None,
    ):
        """
        Text generator for Brahma LLM.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use for inference
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            config: Model configuration (if None, loaded from checkpoint)
        """
        self.device = device
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load model
        self.model = self._load_model(model_path, config)
        self.model.to(self.device)
        self.model.eval()
        
        # Set up logger
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: str, config: Optional[Union[BrahmaConfig, Dict[str, Any]]]):
        """Load model from checkpoint."""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # If config is provided, use it to initialize the model
            if config is not None:
                if isinstance(config, dict):
                    config = BrahmaConfig.from_dict(config)
                model = BrahmaForCausalLM(config)
            else:
                # Try to load config from the model checkpoint
                if "config" in checkpoint:
                    config = checkpoint["config"]
                    if isinstance(config, dict):
                        config = BrahmaConfig.from_dict(config)
                    model = BrahmaForCausalLM(config)
                else:
                    # Default config if none found
                    config = BrahmaConfig()
                    model = BrahmaForCausalLM(config)
            
            # Load model weights
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
            
            return model
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability cutoff for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or greedy decode
            num_return_sequences: Number of sequences to return
        
        Returns:
            List of generated texts
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for text generation")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=self.max_length - max_new_tokens,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device) if "attention_mask" in inputs else None
        
        with torch.no_grad():
            if hasattr(self.model, "generate"):
                # Use model's generate method if available
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                )
            else:
                # Manual generation if generate method not available
                outputs = self._manual_generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                )
        
        # Decode generated text
        generated_texts = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def _manual_generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        num_return_sequences=1,
    ):
        """Manual text generation implementation."""
        batch_size = input_ids.shape[0]
        
        # Expand input for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
        
        # Initialize generated sequences
        generated = input_ids.clone()
        
        # Initialize past key values
        past_key_values = None
        
        # Generate tokens
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is None:
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            else:
                outputs = self.model(
                    input_ids=generated[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            if isinstance(outputs, tuple):
                logits, past_key_values = outputs
            else:
                logits = outputs
                past_key_values = None
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for batch_idx in range(next_token_logits.shape[0]):
                    for prev_token in set(generated[batch_idx].tolist()):
                        next_token_logits[batch_idx, prev_token] /= repetition_penalty
            
            # Filter by top-k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("inf")
            
            # Filter by top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("inf")
            
            # Sample next token
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated
            generated = torch.cat((generated, next_token), dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=1
                )
            
            # Check if we've generated EOS token
            if (next_token == self.model.config.eos_token_id).any():
                # For sequences that have reached EOS, replace the rest with padding
                for batch_idx in range(generated.shape[0]):
                    if next_token[batch_idx, 0] == self.model.config.eos_token_id:
                        break
        
        return generated

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Generate a response in a chat format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability cutoff for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
        
        Returns:
            Generated response
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for chat generation")
        
        # Format messages into a prompt
        prompt = self._format_chat_prompt(messages)
        
        # Generate response
        response = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=1,
        )[0]
        
        # Extract assistant's response
        return self._extract_assistant_response(response, prompt)
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt."""
        formatted = []
        
        for message in messages:
            role = message["role"].lower()
            content = message["content"]
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"{role.capitalize()}: {content}")
        
        # Add assistant prefix for the response
        formatted.append("Assistant:")
        
        return "\n\n".join(formatted)
    
    def _extract_assistant_response(self, full_response: str, prompt: str) -> str:
        """Extract the assistant's response from the full generated text."""
        if prompt in full_response:
            response = full_response[len(prompt):].strip()
        else:
            # If the prompt isn't perfectly preserved, try to extract after the last "Assistant:"
            parts = full_response.split("Assistant:")
            if len(parts) > 1:
                response = parts[-1].strip()
            else:
                response = full_response
        
        # Remove any trailing user or system messages
        if "\n\nUser:" in response:
            response = response.split("\n\nUser:")[0].strip()
        if "\n\nSystem:" in response:
            response = response.split("\n\nSystem:")[0].strip()
        
        return response
