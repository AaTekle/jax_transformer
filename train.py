import pickle  # model checkpoints
import jax  
import jax.numpy as jnp  # Array operations
import optax  # Optimization library (Adam, SGD,)
from flax.training import train_state  # Manages model state during training
from config import config  # Training hyperparameters
from model import GPT  # GPT model architecture
from data import (
    load_text,  # Loading raw text file
    build_vocab,  # Creating character-to-token mappings
    encode,  # Convert text to token IDs
    get_batch  # Sample minibatches for training
)

# Checking available devices (GPU/TPU/CPU)
print(jax.devices())

# Loading raw text and build vocabulary
text = load_text()  # Raw text string from file
stoi, itos = build_vocab(text)  # stoi: char→ID, itos: ID→char
data = encode(text, stoi)  # Convert text to integer token IDs
vocab_size = len(stoi)  # Total number of unique characters

# Initializing GPT model architecture
model = GPT(
    vocab_size=vocab_size,  # Number of unique tokens
    block_size=config.block_size,  # Context window size
    embed_dim=config.embed_dim,  # Embedding dimension
    num_heads=config.num_heads,  # Number of attention heads
    num_layers=config.num_layers  # Number of transformer layers
)

# Initializing model parameters randomly
rng = jax.random.PRNGKey(config.seed)  # Random seed for reproducibility
params = model.init(
    rng,
    jnp.ones((1, config.block_size), dtype=jnp.int32)  # Dummy input to initialize
)

'''
AdamW is an deep learning optimizer algorithm that updates model parameters (weights) to reduce training loss.

Adam (the base)
Adam is the standard adaptive optimizer that:
- Adapts learning rate per parameter, parameters that change a lot get smaller updates, stable parameters get bigger updates
- Uses momentum, remembers past gradients to smooth out updates (like rolling a ball downhill)
- Works well in practice, faster convergence than basic SGD, less tuning needed

## The "W" — Weight Decay
Adam had a subtle issue: when you apply L2 regularization (penalty for large weights) to Adam, it doesn't work as intended. 

AdamW fixes this by applying weight decay directly (multiplying weights by a small factor like 0.99 each step) instead of through the loss function. 
This:
- Prevents weights from growing too large
- Improves generalization (model works better on unseen data)
- Decouples learning rate from weight decay strength

## In simple terms:

Adam    = Take adaptive steps toward lower loss
AdamW   = Take adaptive steps toward lower loss, 
           but also slightly shrink all weights

## Why use AdamW?
- Better generalization, prevents overfitting
- Standard choice, works across most deep learning tasks
- Stable training, less sensitive to learning rate than SGD

This optimizer is what actually improves the model during training by updating params based on gradients.
'''
# Set up optimizer (AdamW) and training state
tx = optax.adamw(config.learning_rate)  # AdamW optimizer with learning rate
state = train_state.TrainState.create(
    apply_fn=model.apply,  # Function to run forward pass
    params=params,  # Model parameters to optimize
    tx=tx  # Optimizer state
)

# Defining loss function: cross-entropy between predictions and targets
def loss_fn(params, x, y):
    """Compute cross-entropy loss between model predictions and true labels"""
    
    # Forward pass: get prediction scores for each token
    logits = model.apply(params, x)  # Shape: (batch_size, seq_len, vocab_size)
    
    # Compute cross-entropy loss and return mean across batch
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return loss.mean()

# JIT-compiled training step: compute gradients and update parameters
@jax.jit
def train_step(state, x, y):
    """Execute one gradient update step"""
    
    # Computing loss and gradients with respect to parameters
    grad_fn = jax.value_and_grad(loss_fn)  # Create gradient function
    loss, grads = grad_fn(state.params, x, y)  # Get loss value and gradients
    
    # Updating parameters using gradients and optimizer state
    state = state.apply_gradients(grads=grads)
    
    return state, loss

# Training loop: iterate through minibatches
for step in range(config.max_steps):
    # Sampling random minibatch from training data
    x, y = get_batch(
        data,
        config.batch_size,  # Number of samples per batch
        config.block_size  # Sequence length per sample
    )
    
    # Converting to JAX arrays
    x = jnp.array(x)  # Input token sequences
    y = jnp.array(y)  # Target token IDs (next token in sequence)
    
    # Updating model parameters
    state, loss = train_step(state, x, y)
    
    # Printing loss at intervals
    if step % config.eval_interval == 0:
        print(f"step={step} loss={float(loss):.4f}")

# Saving trained model and vocabulary to checkpoint
with open("checkpoints.pkl", "wb") as f:
    pickle.dump(
        {
            "params": state.params,  # Trained model weights
            "stoi": stoi,  # Character-to-token mapping
            "itos": itos  # Token-to-character mapping
        },
        f
    )

print("checkpoint saved")
