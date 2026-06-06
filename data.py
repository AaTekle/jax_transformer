from datasets import load_dataset
import numpy as np

from config import config
# Load the training dataset and combine the first 5000 text examples into
# one large string. This creates a simple text source for character-level training.
def load_text():
    ds = load_dataset(config.dataset_name)

    text = " ".join(
        ds["train"]["text"][:5000]
    )

    return text


# Build a character vocabulary from the text. Each unique character gets 
# an integer ID so the model can work with numbers instead of raw text.
def build_vocab(text):
    chars = sorted(list(set(text)))

    # Create a character-to-index dictionary. ch is one unique character
    # from the dataset, and i is the numeric ID assigned to that character.
    stoi = {
        ch: i
        for i, ch in enumerate(chars)
    }

    # Create an index-to-character dictionary. ch is the original character,
    # and i is its numeric ID from stoi, so this reverses the mapping.
    itos = {
        i: ch
        for ch, i in stoi.items()
    }

    return stoi, itos


# Convert text into token IDs using the character-to-integer vocabulary. (b/c nn's don't understand raw textual information)
def encode(text, stoi):
    return np.array([
        # c is one character from the input text. stoi[c] looks up the
        # numeric token ID for that character.
        stoi[c]
        for c in text
    ])


# Convert token IDs back into readable text using the integer-to-character map. (b/c the model does not work with characters directly. It works with numbers (token IDs).)
def decode(tokens, itos):
    return "".join([
        # t is one token ID from the model/data. int(t) makes sure it can
        # be used as a dictionary key, then itos maps it back to a character.
        itos[int(t)]
        for t in tokens
    ])


# Create one training batch by sampling random text windows from the data. (training on random text windows is an efficient way to expose the model to many different parts of the dataset)
# x contains input tokens, and y contains the same tokens shifted one step
# forward so the model learns to predict the next character.
def get_batch(
    data,
    batch_size,
    block_size
):
    # Pick random starting positions for each sequence in the batch.
    ix = np.random.randint(
        0,
        len(data) - block_size - 1,
        size=batch_size
    )

    # Build input sequences (because the model needs input examples to learn from). 
    # (i is one random start position from ix, and data[i:i + block_size] takes block_size tokens starting at that position.
    x = np.stack([
        data[i:i + block_size]
        for i in ix
    ])

    # Build target sequences (because the model needs an answer key during training). i is the same start position, but the slice
    # starts one token later so each target is the "next character" answer.
    y = np.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y
