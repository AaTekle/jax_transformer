import pickle  # Loading saved model checkpoint
import jax 
import jax.numpy as jnp  # Array operations
from config import config  # Hyperparameters
from model import GPT  # Model architecture
from data import decode  # Convert token IDs to text

# checkpoint holding trained parameters and vocabulary mappings
with open("checkpoints.pkl", "rb") as f:
    checkpoint = pickle.load(f)

# extracting model weights, and character/token mappings
params = checkpoint["params"]  # Trained model parameters
stoi = checkpoint["stoi"]  # String-to-integer: char to token ID
itos = checkpoint["itos"]  # Integer-to-string: token ID to char
vocab_size = len(stoi)  # Total number of unique tokens


# GPT model
model = GPT(
    vocab_size=vocab_size,  # Number of unique tokens
    block_size=config.block_size,  # Maximum context length
    embed_dim=config.embed_dim,  # Embedding dimension
    num_heads=config.num_heads,  # Number of attention heads
    num_layers=config.num_layers  # Number of transformer layers
)


# JIT compile for fast execution, generates next token from current sequence
@jax.jit
def generate_step(params, idx):
    """Generate next token given model params and current token sequence"""
    
    # Forward pass, get logits (prediction scores) for all positions
    logits = model.apply(
        params,
        idx[:, -config.block_size:]  # using only last block_size tokens
    )
    
    # Extracting logits for only the last position (we only need next token prediction)
    logits = logits[:, -1, :]
    
    # Sampling next token from the probability distribution (higher logits = higher prob)
    next_token = jax.random.categorical(
        jax.random.PRNGKey(int(idx.shape[1])),  # Seed by sequence length
        logits
    )
    
    return next_token


# Converting prompt to token IDs and prepare for generation
prompt = "Once upon a time"
input_tokens = jnp.array([
    [stoi[c] for c in prompt]  # Converting each character to integer token ID
])

'''
Autoregressive generation

- sequential prediction loop used by Large Language Models (e.g., ChatGPT or Llama) to write text
- instead of generating an entire paragraph in a single split-second calculation, the model behaves like a human typing on a keyboard, it stops after every single word (token), reads everything written so far, picks the next word, and repeats the process until it generates a "stop" signal.

## autoregressive generation use cases and benefits:

- Language relies on context, you cannot accurately predict word #10 in a sentence without knowing exactly what words #1 through #9 are.
- Models cannot predict the future, a neural network can only calculate probabilities based on known data. Since the model doesn’t know what it is going to say until it says it, it must generate step-by-step to build its own context.
- It preserves coherence, by feeding its own outputs back into itself, the model maintains a continuous "train of thought," making sure that word #11 logically follows word #10.
'''
# Autoregressive generation, predicts next token 200 times, append each time
for _ in range(200):
    # Predicting next token based on current sequence
    next_token = generate_step(params, input_tokens)
    
    # Reshaping from (1,) to (1, 1) for concatenation
    next_token = next_token[:, None]
    
    # Appending new token to sequence
    input_tokens = jnp.concatenate(
        [input_tokens, next_token],
        axis=1  # Concatenating along sequence length dimension
    )


# Converting token IDs back to readable text
generated = decode(input_tokens[0], itos)

# Saving file and printing
with open("generated.txt", "w", encoding="utf-8") as f:
    f.write(generated)

print(generated)
print("\nsaved to generated.txt")
