import math
from utils.gpt_config import ModelConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, max_seq_length: int):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        
        self.d_k = self.d_model // self.n_heads

        self.query_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.key_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.value_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)

        self.out_proj = nn.Linear(self.n_heads * self.d_k, self.d_model, bias=False)

        self.register_buffer("mask", torch.tril(torch.ones(max_seq_length, max_seq_length)) * float('-inf'))


    def forward(self, x):
        B, T, C = x.shape

        key = self.key_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2) # B, n_heads, T, d_k
        query = self.query_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        value = self.value_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        #REPLACEd WITH FLASH ATTENTION

        # attn_weight = (query @ key.transpose(-1, -2)) / math.sqrt(self.d_k)  # B, n_heads, T, T
        # attn_weight = attn_weight.masked_fill(self.mask[:T, :T] == 0, float('-inf'))  # apply mask
        # attn_weight = F.softmax(attn_weight, dim=-1)  # softmax along the last dimension

        # out = (attn_weight @ value).transpose(1, 2).contiguous().view(B, T, C)  # B, T, C

        # out = self.out_proj(out)
        # return out

        out = F.scaled_dot_product_attention(query=query, key=key, value=value, attn_mask=self.mask, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.out_proj(out) # B, T, C @ (n_heads * d_k, d_model) -> B, T, C

        return out
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)

        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))



class AttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attention = CausalSelfAttention(config.d_model, config.n_heads, config.d_k, config.max_seq_len)
        self.layernorm1 = nn.LayerNorm(config.d_model)
        self.feedforward = FeedForward(config.d_model)
        self.layernorm2 = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = x + self.attention(self.layernorm1(x)) #in the model they use layernrm before the attention
        x = x + self.feedforward(self.layernorm2(x)) # layernorm before the feedforward + residual connection
        return x
    


class GPT2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(self.config.n_vocab, self.config.d_model)
        self.pos_embed = nn.Embedding(self.config.max_seq_len, self.config.d_model)
        self.blocks = nn.ModuleList([AttentionBlock(self.config) for _ in range(self.config.n_layers)])
        self.layernorm = nn.LayerNorm(self.config.d_model)
        self.head = nn.Linear(self.config.d_model, self.config.n_vocab)

    def forward(self, input_ids, target=None):
        B, T = input_ids.shape

        assert T <= self.config.max_seq_len, "Input sequence length must be less than or equal to the maximum sequence length"

        x = self.embed(input_ids) + self.pos_embed(torch.arange(T, device=input_ids.device))

        for block in self.blocks:
            x = block(x)

        x = self.layernorm(x)

        logits = self.head(x)

        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

        return logits, loss
