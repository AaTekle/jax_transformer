from datasets import load_dataset
import numpy as np

from config import config


def load_text():
    ds = load_dataset(config.dataset_name)

    text = " ".join(
        ds["train"]["text"][:5000]
    )

    return text


def build_vocab(text):
    chars = sorted(list(set(text)))

    stoi = {
        ch: i
        for i, ch in enumerate(chars)
    }

    itos = {
        i: ch
        for ch, i in stoi.items()
    }

    return stoi, itos


def encode(text, stoi):
    return np.array([
        stoi[c]
        for c in text
    ])


def decode(tokens, itos):
    return "".join([
        itos[int(t)]
        for t in tokens
    ])


def get_batch(
    data,
    batch_size,
    block_size
):
    ix = np.random.randint(
        0,
        len(data) - block_size - 1,
        size=batch_size
    )

    x = np.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = np.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y