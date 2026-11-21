# A transformer language model built from first principles

A small decoder-only transformer written in PyTorch, where the core operations have hand-derived backward passes instead of relying on autograd. I trained it on the complete works of Shakespeare, first on a MacBook and now on an RTX 5070.

The goal was to understand every piece of a modern language model well enough to write it myself: attention, normalization, positional encoding, the training loop and the diagnostics that show whether training worked.

## What's implemented

| Component | Implementation |
|---|---|
| Architecture | Decoder-only transformer, pre-norm residual blocks |
| Attention | Multi-head causal self-attention (4 heads, d_k = 32) |
| Normalization | RMSNorm with a **hand-written backward pass** |
| Softmax | Numerically stable softmax with a **hand-written backward pass** |
| QKV projection | Batched per-head projection with a **hand-written backward pass** |
| Embedding | Lookup with a **hand-written backward pass** (scatter-add of gradients) |
| Activation | Softplus ("smooth ReLU") with a **hand-written backward pass** |
| Positional encoding | Sinusoidal |
| Tokenizer | Byte-level BPE, 2,000-token vocabulary (Hugging Face `tokenizers`) |
| Training | AdamW, LR warmup + cosine decay, dropout, periodic evaluation, best/last checkpointing, CSV logs, loss plots |
| Inference | Temperature sampling |

Each custom op is a `torch.autograd.Function` whose `backward` I derived on paper. For example, for RMSNorm y = γ · x / rms(x), the input gradient is

```
dL/dx = (g·γ) / rms  −  x · Σ(g·γ·x) / (d · rms³)
```

## First run

Model: 1.30M parameters (4 layers, width 128, 512-token context). Data: 1.78M training tokens, 30k validation tokens. Training took 2 h 16 min on Apple Silicon (MPS) at about 19k tokens/s.

| Model | Validation loss (nats/token) |
|---|---|
| Uniform guess | 7.60 |
| Unigram baseline | 6.13 |
| Bigram baseline | 4.31 |
| **This transformer (best checkpoint)** | **3.70** (perplexity 40.6, 1.85 bits/byte) |

![Loss curves](assets/loss_curves.png)

**What I learned from the run.** Validation loss bottomed out at step 1800 and then rose while training loss kept falling. The model saw its training set 86 times, and the train–validation gap grew almost linearly from step ~400. That is classic overfitting from data repetition. Training itself was stable, with no loss spikes.

![Train-validation gap](assets/train_val_gap.png)

## Experiments since the first run

Same model and data, 3,000 steps each, now trained on an RTX 5070 (CUDA): the same run takes about 10 minutes there, against 2 h 16 min on the MacBook.

| Run | Best validation loss | Final train–val gap |
|---|---|---|
| Baseline (constant LR) | 3.704 | 0.75 |
| + LR warmup and cosine decay | 3.706 | 0.43 |
| + Dropout 0.1 (attention and residual) | **3.673** | **0.21** |

Warmup and cosine decay barely moved the best validation loss. Dropout gave the best result so far and cut the overfitting gap by more than two thirds, so it's now on by default. I also tried tying the input embedding and output layer weights; at this size it made results noticeably worse, so I've left it out for now and will retry it at GPT-2 scale.

## Current work

**Now:** finishing the Shakespeare model. Next up are random-window batching (sampling `(batch, context)` windows at random offsets instead of fixed chunks, as a cheap regularizer on such a small dataset), gradient clipping, a byte-level tokenizer alphabet so unseen characters never break encoding, a `tests/` script (gradcheck on every custom op, tokenizer round-trip, causal-mask check), and saving text samples with each run.

**Roadmap:** I'm working through a staged plan from this model toward a modern LLM:

1. **The GPT recipe:** my own BPE tokenizer, a GPT-2 small (124M) model trained on FineWeb-Edu on the 5070
2. **Scaling laws:** a small-model sweep, fitted loss curves, μP
3. **Llama-style block:** SwiGLU, RoPE, grouped-query attention, KV cache
4. **Speed and long context:** profiling, FlashAttention, the Muon optimizer

## Files

| File | Purpose |
|---|---|
| `Transformer.py` | Model and custom autograd functions |
| `Training.py` | Training loop, evaluation, logging, checkpoints |
| `Inference.py` | Generate text from a prompt |
| `byte_pair_encoding.py` | Train the BPE tokenizer |

## Running it

```bash
pip install torch tokenizers matplotlib
python byte_pair_encoding.py   # train the tokenizer
python Training.py             # train the model
python Inference.py            # generate text
```
