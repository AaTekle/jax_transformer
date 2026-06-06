import jax
import jax.numpy as jnp

from flax import linen as nn

'''
whole script is about moving information through clean, predictable tensor shapes at every step.
'''

class MultiHeadSelfAttention(nn.Module):
    # Embedding size for each token representation.
    embed_dim: int

    # Number of attention heads used to learn different token relationships.
    num_heads: int

    # decorator within JAX that allows you to define neural network submodules (making code look clean & concise)
    @nn.compact
    def __call__(self, x):
        # B = batch size, T = sequence length, C = embedding dimension.
        B, T, C = x.shape

        # Split the embedding across multiple attention heads.
        head_dim = (
            C // self.num_heads
        )

        # Create queries (q), keys (k), and values (v), the core inputs used by self-attention to determine which tokens matter most.
        qkv = nn.Dense(
            3 * C
        )(x)

        qkv = qkv.reshape(
            B,
            T,
            3,
            self.num_heads,
            head_dim
        )

        q, k, v = jnp.split(
            qkv,
            3,
            axis=2
        )

        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)

        # Computing attention scores between all token pairs.
        attn = jnp.einsum(
            "bthd,bshd->bhts",
            q,
            k
        )

        # Scaling scores for more stable training.
        attn = (
            attn / jnp.sqrt(head_dim)
        )

        # Preventing tokens from attending to future tokens during training.
        mask = jnp.tril(
            jnp.ones((T, T))
        )

        attn = jnp.where(
            mask == 0,
            -1e9,
            attn
        )

        # Converting attention scores into probabilities.
        attn = nn.softmax(
            attn,
            axis=-1
        )

        # Use attention weights to combine information from relevant tokens.
        out = jnp.einsum(
            "bhts,bshd->bthd",
            attn,
            v
        )

        out = out.reshape(
            B,
            T,
            C
        )

        return nn.Dense(C)(out)


class FeedForward(nn.Module):
    # Small neural network applied independently to each token.
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        # Expanding the representation, apply a non-linearity, then project
        # back down to the original embedding size.
        x = nn.Dense(
            4 * self.embed_dim
        )(x)

        x = nn.gelu(x)

        x = nn.Dense(
            self.embed_dim
        )(x)

        return x


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        # Residual connection allows attention to improve the representation
        # without losing the original information.
        x = x + MultiHeadSelfAttention(
            self.embed_dim,
            self.num_heads
        )(
            nn.LayerNorm()(x)
        )

        # Residual connections around the feed-forward network.
        x = x + FeedForward(
            self.embed_dim
        )(
            nn.LayerNorm()(x)
        )

        return x


class GPT(nn.Module):
    # Number of unique tokens the model can predict.
    vocab_size: int

    # Maximum sequence length supported by the model.
    block_size: int

    embed_dim: int
    num_heads: int
    num_layers: int

    @nn.compact
    def __call__(self, idx):
        # B = batch size, T = sequence length.
        B, T = idx.shape

        # Converting token IDs into learned vector representations.
        tok_emb = nn.Embed(
            self.vocab_size,
            self.embed_dim
        )(idx)

        # Generating token positions so the model knows token order.
        pos = jnp.arange(T)

        # Learn a representation for each position in the sequence.
        pos_emb = nn.Embed(
            self.block_size,
            self.embed_dim
        )(pos)

        # Combining token meaning with position information.
        x = tok_emb + pos_emb

        # Stacking multiple transformer blocks to learn complex language patterns.
        for _ in range(
            self.num_layers
        ):
            x = TransformerBlock(
                self.embed_dim,
                self.num_heads
            )(x)

        x = nn.LayerNorm()(x)

        # Producing scores for every possible next token.
        logits = nn.Dense(
            self.vocab_size
        )(x)

        return logits
