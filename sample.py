import pickle

import jax
import jax.numpy as jnp

from config import config
from model import GPT
from data import decode


with open(
    "checkpoints.pkl",
    "rb"
) as f:
    checkpoint = pickle.load(f)

params = checkpoint["params"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

vocab_size = len(stoi)

model = GPT(
    vocab_size=vocab_size,
    block_size=config.block_size,
    embed_dim=config.embed_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers
)


@jax.jit
def generate_step(params, idx):
    logits = model.apply(
        params,
        idx[:, -config.block_size:]
    )

    logits = logits[:, -1, :]

    next_token = jax.random.categorical(
        jax.random.PRNGKey(
            int(idx.shape[1])
        ),
        logits
    )

    return next_token


prompt = "Once upon a time"

input_tokens = jnp.array([
    [stoi[c] for c in prompt]
])

for _ in range(200):
    next_token = generate_step(
        params,
        input_tokens
    )

    next_token = next_token[:, None]

    input_tokens = jnp.concatenate(
        [
            input_tokens,
            next_token
        ],
        axis=1
    )

generated = decode(
    input_tokens[0],
    itos
)

with open(
    "generated.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(generated)

print(generated)
print("\nsaved to generated.txt")