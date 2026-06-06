from dataclasses import dataclass

# Store training, model, and dataset settings in one object so the main
# code stays clean and these values can be changed from one place.
@dataclass
class Config:
    '''
    Number of training examples processed at the same time. A batch size
    of 32 is a common starting point because it gives a stable learning
    signal without using too much memory.
    '''
    batch_size: int = 32
    
    '''
    Number of tokens the model looks at in one sequence. A block size of
    128 gives the model enough context to learn short story patterns while
    keeping training cheaper than using very long sequences.
    '''
    block_size: int = 128
    
    '''
    Total number of training updates. 5000 steps is enough for a small
    experiment to show learning progress without making training too long.
    '''
    max_steps: int = 5000
    
    '''
    Controls how much model weights change after each update. 3e-4 means
    0.0003, a small step size that is often stable for transformer models:
    large enough to learn, but small enough to avoid unstable jumps.
    '''
    learning_rate: float = 3e-4
    
    '''
    Size of each token's learned vector. An embedding size of 256 gives
    the model enough space to represent word meaning and context while
    staying small enough for fast training.
    '''
    embed_dim: int = 256
    
    '''
    Number of attention heads. 4 heads let the model look for multiple
    types of token relationships at the same time, while keeping the model
    simple and efficient.
    '''
    num_heads: int = 4
    
    '''
    Number of transformer layers. 4 layers give the model enough depth to
    learn patterns beyond simple word matching without making the model
    too slow or memory-heavy.
    '''
    num_layers: int = 4
    
    # TinyStories Dataset used for training.
    dataset_name: str = "roneneldan/TinyStories"
    
    '''
    Random seed used to make results more repeatable. Using 0 is a simple
    default so random choices like initialization and data shuffling are
    consistent across runs.
    '''
    seed: int = 0
    
    '''
    How often to check model performance during training. Every 100 steps
    gives regular feedback without slowing training too much.
    '''
    eval_interval: int = 100


# Create one config object using the default values above.
config = Config()
