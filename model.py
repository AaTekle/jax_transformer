import jax
import jax.numpy as jnp

from flax import linen as nn


class MultiHeadSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape

        head_dim = (
            C // self.num_heads
        )

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

        attn = jnp.einsum(
            "bthd,bshd->bhts",
            q,
            k
        )

        attn = (
            attn / jnp.sqrt(head_dim)
        )

        mask = jnp.tril(
            jnp.ones((T, T))
        )

        attn = jnp.where(
            mask == 0,
            -1e9,
            attn
        )

        attn = nn.softmax(
            attn,
            axis=-1
        )

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
    embed_dim: int

    @nn.compact
    def __call__(self, x):
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
        x = x + MultiHeadSelfAttention(
            self.embed_dim,
            self.num_heads
        )(
            nn.LayerNorm()(x)
        )

        x = x + FeedForward(
            self.embed_dim
        )(
            nn.LayerNorm()(x)
        )

        return x


class GPT(nn.Module):
    vocab_size: int
    block_size: int
    embed_dim: int
    num_heads: int
    num_layers: int

    @nn.compact
    def __call__(self, idx):
        B, T = idx.shape

        tok_emb = nn.Embed(
            self.vocab_size,
            self.embed_dim
        )(idx)

        pos = jnp.arange(T)

        pos_emb = nn.Embed(
            self.block_size,
            self.embed_dim
        )(pos)

        x = tok_emb + pos_emb

        for _ in range(
            self.num_layers
        ):
            x = TransformerBlock(
                self.embed_dim,
                self.num_heads
            )(x)

        x = nn.LayerNorm()(x)

        logits = nn.Dense(
            self.vocab_size
        )(x)

        return logits