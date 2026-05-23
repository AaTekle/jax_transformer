from dataclasses import dataclass


@dataclass
class Config:
    # training
    batch_size: int = 32
    block_size: int = 128

    max_steps: int = 5000
    learning_rate: float = 3e-4

    # model
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4

    # dataset
    dataset_name: str = "roneneldan/TinyStories"

    # misc
    seed: int = 0
    eval_interval: int = 100


config = Config()