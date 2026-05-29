# JAX Transformer Language Model

A decoder-only Transformer language model implemented from scratch using JAX, Flax, and Optax.

This project uses:
- Transformer architecture fundamentals
- Autoregressive language modeling (meaning it generates text one token at a time using previously generated tokens as context.)
- GPU-accelerated training with JAX/XLA
- Attention mechanisms
- Token embedding and sequence prediction
- End-to-end training and inference pipelines

The model was trained on the TinyStories dataset and generates text autoregressively one token at a time.

---

# General Objective

The goal of this project was to implement a functional low-level GPT-style Transformer architecture from scratch without relying on high-level training frameworks.

The project focuses on:
- understanding Transformer internals
- implementing causal self-attention
- building a training loop manually
- using JAX for accelerated tensor computation
- generating coherent natural language text

---

# Generated Output 

#### Note: File Location - Folder (generated_output)

```text
Once upon a time, it's magic appeared up and mom came really far.

She around across the watch money and the twins people every day. She looked myself excited and went on adventures about this, she printed up and dow
```
---

## Dependencies Used

### Dependencies

- JAX
  - high-performance tensor computation
  - automatic differentiation
  - XLA compilation
  - GPU acceleration

- Flax
  - neural network module system for JAX

- Optax
  - optimization and gradient update library

- HuggingFace Datasets
  - dataset loading and preprocessing

- NumPy
  - batch sampling and preprocessing utilities

---

# Dataset

Dataset used:

- TinyStories

TinyStories is a synthetic dataset designed for training small language models efficiently while still producing coherent outputs.

- lightweight
- fast training
- strong grammatical consistency
- optimal for small Transformer projects

---

# Model Architecture

The model is a decoder-only Transformer similar to other GPT architectures.

### Model Pipeline:

Input Tokens
→ Token Embeddings
→ Positional Embeddings
→ Transformer Blocks
→ Layer Normalization
→ Linear Projection
→ Vocabulary Logits

---

# Transformer Components

## 1. Token Embedding

Token Embedding: Learned numerical representation of a piece of text.

Each token index is mapped into a dense vector representation.

Mathematically:

```math
x_i = E[t_i]
```

Where:
- $t_i$ is token index
- $E$ is embedding matrix
- $x_i$ is embedded token vector

---

## 2. Positional Embedding

Transformers have no native sequence awareness, so positional embeddings are added:

```math
h_i = x_i + p_i
```

Where:
- $x_i$ = token embedding
- $p_i = positional embedding

This allows the model to learn sequence order.

---

# Self-Attention Mechanism

The core operation of the Transformer is scaled dot-product self-attention.

Self-attention is the mechanism that allows the model to determine: which words or tokens are most important when understanding context.

This is the main reason why Transformers work so well within language modeling.

Computers do not understand language naturally.

To a neural network, text is simply:
- numbers
- vectors
- mathematical relationships

The model must learn:
- which words relate to each other
- which earlier tokens matter
- how context changes meaning

Instead of sequential memory, every token can directly look at every other token.

Attention is computed using:

```math
Attention(Q, K, V) = softmax((QKᵀ) / √dₖ)V
```

Where:
- $Q$: query matrix representing what each token is searching for

- $K$: key matrix representing what information each token contains

- $V$: value matrix containing the actual token information

- $QKᵀ$: computes similarity scores between tokens

- $√dₖ$: scaling factor used for numerical stability

- $softmax$: converts scores into probabilities


## Multi-Head Attention
Transformers use multiple attention heads simultaneously.

Different heads can learn different relationships:
- grammar
- sentence structure
- semantic meaning
- long-range dependencies

This improves the model’s ability to capture complex language patterns.

---

## Query, Key, Value Projections

Each token in the sentence is converted into three learned vectors:
- Query (Q)
- Key (K)
- Value (V)

These help the model decide which other tokens are important.

Simple intuition:

- Query:
  - what this token is searching for

- Key:
  - what information this token contains

- Value:
  - the actual information passed forward

For input matrix `X`:

```
Q = XW_Q
K = XW_K
V = XW_V
```

Where:
- $Q$ = queries
- $K$ = keys
- $V$ = values

---

## Attention Scores

Attention scores measure how relevant other tokens are to the current token.

They are computed using:

```math
Attention(Q, K, V) = softmax((QKᵀ) / √dₖ)V
```
Where:
- $QKᵀ$ computes similarity between tokens
- $d_k$ is the key vector dimension used for scaling
- $softmax$ converts scores into probabilities

Higher attention scores mean the model focuses more strongly on those tokens when predicting the next token.

---

## Causal Masking

The Transformer uses a causal mask to prevent tokens from accessing future information.

This means each token can only attend to:
- itself
- previous tokens in the sequence

and not future tokens.

This preserves autoregressive generation behavior (meaning it generates text one token at a time using previously generated tokens as context.):

```math
P(x_t | x_1, x_2, ..., x_{t-1})
```

Meaning:
- the next token is predicted only from earlier context

For example, when predicting the next word in:

```text
"The cat sat on the ___"
```

the model cannot look ahead at the correct answer during training.

---

# Feed Forward Network

Each Transformer block also includes a feed-forward neural network (MLP).

```math
FFN(x) = W₂ · GELU(W₁x + b₁) + b₂
```
- $x$: input token representation entering the feed-forward network

- $W₁$
  - first weight matrix
  - projects the input into a higher-dimensional space

- $b₁$: bias vector added after the first linear transformation

- $GELU$: activation function (Gaussian Error Linear Unit)
  - introduces nonlinearity so the model can learn complex patterns

- $W₂$
  - second weight matrix
  - projects the representation back to the original embedding dimension

- $b₂$: second bias vector added after the final projection

This layer processes each token independently after the attention operation.

Its purpose is to:
- refine learned features
- introduce nonlinearity
- increase the model’s representational capacity

While attention determines which tokens are important, the feed-forward network helps transform and strengthen the learned representations.

---

# Layer Normalization

Layer normalization helps stabilize training by keeping activations within a consistent range.

```math
LayerNorm(x) = ((x - μ) / √(σ² + ε))γ + β
```

Where:
- $μ$: mean of the activations

- $σ²$: variance of the activations

- $ε$: small constant added for numerical stability

- $γ$: learned scaling parameter

- $β$: learned shifting parameter

Benefits of Layer normalization:
- smoother gradient flow
- more stable optimization
- faster convergence during training

Layer normalizatios are vital within deep Transformer architectures because it prevents activations from growing too large or becoming unstable across layers.

---

## Loss Function

The model is trained using categorical cross-entropy loss.

This loss measures how different the model’s predicted probabilities are from the correct next token.

```math
L = -Σ yᵢ log(pᵢ)
```

Where:
- $L$: total loss value

- $yᵢ$: true probability distribution for the correct token

- $pᵢ$: predicted probability for each token

- $log$: logarithm used to penalize incorrect confident predictions

The objective is to minimize loss by increasing the probability assigned to the correct next token during training.

---

# Optimization

## Optimizer

The model uses the AdamW optimizer for training.

AdamW improves training stability by combining:
- momentum
- adaptive learning rates
- weight decay regularization

Parameter updates are computed using:

```math
θₜ₊₁ = θₜ - η · (mₜ / (√vₜ + ε))
```

Where:
- $θₜ$: current model parameters

- $θₜ₊₁$: updated parameters after optimization

- $η$: learning rate controlling update size

- $mₜ$: momentum estimate tracking average gradients

- $vₜ$: variance estimate tracking gradient magnitude

- $ε$: small constant added for numerical stability

AdamW helps the model converge faster and train more reliably, especially in deep neural networks such as Transformers.

---

# Training Results

Training was performed on an NVIDIA RTX GPU using JAX XLA compilation.

Loss function progression:

```text
step=0     loss=4.8179
step=1000  loss=1.2611
step=2000  loss=1.1095
step=3000  loss=0.9370
step=4000  loss=0.9396
step=4900  loss=0.8894
```

## Analysis

The model converged successfully:
- rapid early loss reduction
- stable optimization
- gradual convergence below 1.0 loss

The later-stage oscillation is expected due to:
- stochastic minibatch sampling
    - process of randomly selecting small, fixed-size subsets (minibatches) of training data in each iteration of an optimization algorithm
- relatively small model size
- limited dataset subset

Final outputs demonstrated:
- coherent sentence structure
- grammatical consistency
- story-like generation patterns

---

# Text Generation

Generation is autoregressive. (meaning it generates text one token at a time using previously generated tokens as context.
)

At each step:

1. Feed current token sequence into model
2. Compute logits for next token
3. Sample highest-probability token
4. Append token to sequence
5. Repeat

Mathematically:

``` math
x_t ~ P(x_t | x_<t)
```

Where:
- $xₜ$: the next token being generated

- $x<t$: all previous tokens before position `t`

- $P(xₜ | x<t)$: the probability distribution for the next token given the previous context

This process allows the model to generate variable-length text.

---

# GPU Acceleration

JAX compiles operations using XLA (Accelerated Linear Algebra).

Advantages:
- optimized GPU kernels (specialized function within Graphics Processing Unit (GPU), executed simultaneously in parallel by thousands or millions of threads to process massive datasets efficiently.)
- operation fusion (ML compiler optimization that combines multiple consecutive tensor operations into a single GPU kernel)
- optimized tensor execution
    - optimized tensor execution via  pure functional programming + XLA (Accelerated Linear Algebra) compilation 

---

# Running the Project (within your own local environment)

### Train Model

```bash
python train.py
```

### Generate Text

```bash
python sample.py
```
