import pickle

import jax
import jax.numpy as jnp
import optax

from flax.training import train_state

from config import config

from model import GPT

from data import (
    load_text,
    build_vocab,
    encode,
    get_batch
)


print(jax.devices())

text = load_text()

stoi, itos = build_vocab(text)

data = encode(text, stoi)

vocab_size = len(stoi)

model = GPT(
    vocab_size=vocab_size,
    block_size=config.block_size,
    embed_dim=config.embed_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers
)

rng = jax.random.PRNGKey(
    config.seed
)

params = model.init(
    rng,
    jnp.ones(
        (1, config.block_size),
        dtype=jnp.int32
    )
)

tx = optax.adamw(
    config.learning_rate
)

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx
)


def loss_fn(params, x, y):
    logits = model.apply(
        params,
        x
    )

    loss = (
        optax
        .softmax_cross_entropy_with_integer_labels(
            logits,
            y
        )
    )

    return loss.mean()


@jax.jit
def train_step(state, x, y):
    grad_fn = jax.value_and_grad(
        loss_fn
    )

    loss, grads = grad_fn(
        state.params,
        x,
        y
    )

    state = state.apply_gradients(
        grads=grads
    )

    return state, loss


for step in range(
    config.max_steps
):
    x, y = get_batch(
        data,
        config.batch_size,
        config.block_size
    )

    x = jnp.array(x)
    y = jnp.array(y)

    state, loss = train_step(
        state,
        x,
        y
    )

    if (
        step
        % config.eval_interval
        == 0
    ):
        print(
            f"step={step} "
            f"loss={float(loss):.4f}"
        )


with open(
    "checkpoints.pkl",
    "wb"
) as f:
    pickle.dump(
        {
            "params": state.params,
            "stoi": stoi,
            "itos": itos
        },
        f
    )

print("checkpoint saved")