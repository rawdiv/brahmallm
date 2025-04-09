import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict

@dataclass
class BrahmaConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # For grouped-query attention
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1] // 2)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class BrahmaMLP(nn.Module):
    def __init__(self, config: BrahmaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class BrahmaAttention(nn.Module):
    def __init__(self, config: BrahmaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.kv_repeat = self.num_heads // self.num_kv_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):
        batch_size, seq_len = hidden_states.shape[:2]
        
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary position embeddings
        query_states, key_states = apply_rotary_emb(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            freqs_cis=freqs_cis,
        )
        
        # Transpose back
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        
        # Update cache if needed
        if cache_k is not None and cache_v is not None:
            key_states = torch.cat([cache_k, key_states], dim=1)
            value_states = torch.cat([cache_v, value_states], dim=1)
        
        if use_cache:
            cache_k, cache_v = key_states, value_states
        
        # Repeat the keys and values for multi-query attention
        if self.kv_repeat > 1:
            key_states = key_states.repeat_interleave(self.kv_repeat, dim=2)
            value_states = value_states.repeat_interleave(self.kv_repeat, dim=2)
        
        # Compute attention
        attn_weights = torch.matmul(
            query_states.transpose(1, 2), 
            key_states.transpose(1, 2).transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states.transpose(1, 2))
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        attn_output = self.o_proj(attn_output)
        
        if use_cache:
            return attn_output, (cache_k, cache_v)
        return attn_output


class BrahmaDecoderLayer(nn.Module):
    def __init__(self, config: BrahmaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = BrahmaAttention(config)
        self.mlp = BrahmaMLP(config)
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        if past_key_value is not None:
            cache_k, cache_v = past_key_value
        else:
            cache_k, cache_v = None, None
        
        if use_cache:
            hidden_states, (cache_k, cache_v) = self.self_attn(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                cache_k=cache_k,
                cache_v=cache_v,
                use_cache=use_cache,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
            )
        
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        if use_cache:
            return hidden_states, (cache_k, cache_v)
        return hidden_states


class BrahmaModel(nn.Module):
    def __init__(self, config: BrahmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([BrahmaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads // 2,
            config.max_position_embeddings,
            theta=config.rope_theta,
        )
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            if past_key_values is not None:
                position_ids = torch.arange(
                    past_key_values[0][0].shape[1],
                    seq_length + past_key_values[0][0].shape[1],
                    device=input_ids.device,
                )
            else:
                position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = torch.cat(
                [torch.ones_like(attention_mask[:, :1]), attention_mask], dim=-1
            )
            
            # Create causal mask
            seq_length_with_past = seq_length
            if past_key_values is not None:
                seq_length_with_past += past_key_values[0][0].shape[1]
            
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length_with_past), -float("inf"), device=attention_mask.device),
                diagonal=1,
            )
            attention_mask = attention_mask[:, None, None, :] + causal_mask[None, None, :, :]
        
        # Extract RoPE embeddings
        freqs_cis = self.freqs_cis.to(input_ids.device)
        freqs_cis = freqs_cis[position_ids]
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize past_key_values if not provided
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        
        # Initialize cache if use_cache
        if use_cache:
            new_past_key_values = []
        
        # Process through layers
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if use_cache:
                hidden_states, new_past_kv = layer(
                    hidden_states,
                    freqs_cis=freqs_cis,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
                new_past_key_values.append(new_past_kv)
            else:
                hidden_states = layer(
                    hidden_states,
                    freqs_cis=freqs_cis,
                    attention_mask=attention_mask,
                )
        
        # Apply final normalization
        hidden_states = self.norm(hidden_states)
        
        if use_cache:
            return hidden_states, tuple(new_past_key_values)
        return hidden_states


class BrahmaForCausalLM(nn.Module):
    def __init__(self, config: BrahmaConfig):
        super().__init__()
        self.config = config
        self.model = BrahmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Forward through the model
        if labels is not None:
            use_cache = False
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        if isinstance(outputs, tuple):
            hidden_states, past_key_values = outputs
        else:
            hidden_states = outputs
            past_key_values = None
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        if use_cache:
            return (loss, logits, past_key_values) if loss is not None else (logits, past_key_values)
        
        return (loss, logits) if loss is not None else logits
    
    def generate(
        self, 
        input_ids, 
        attention_mask=None,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        past_key_values = None
        generated = input_ids
        
        # Set to eval mode
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                if past_key_values is None:
                    outputs = self(
                        input_ids=generated,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                else:
                    outputs = self(
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
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("inf")
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("inf")
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated
                generated = torch.cat((generated, next_token), dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=1
                )
                
                # Check if we've generated EOS token
                if (next_token == self.config.eos_token_id).any():
                    break
        
        return generated
