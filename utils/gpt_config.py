from dataclasses import dataclass

@dataclass
class ModelConfig:
    max_seq_len: int = 1024
    n_layers: int = 12
    d_k: int = 64
    n_heads: int = 12
    batch_size: int = 32
    d_model: int = 768
    n_vocab: int = 50257

