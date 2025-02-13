# AtomicGPT: A Lightweight Decoder-Only Transformer for Coherent Text Generation

![AtomicGPT](https://img.shields.io/badge/Parameters-1M-green) 

AtomicGPT is a highly efficient decoder-only transformer model designed for coherent text generation at a microscopic scale (1M parameters). Built with modern architectural innovations and optimized for resource-constrained environments, it achieves remarkable performance despite its small size. Ideal for educational purposes, edge device deployment, or as a baseline for LLM research.

## Key Features
- 🦺 **1 Million Parameter** architecture
- 🧠 **Decoder-Only** transformer design
- 🚀 Modern techniques: Grouped Query Attention, SwiGLU, Rotary Positional Embeddings
- 📖 Trained on 60M tokens from TinyStories dataset
- ⚡ Full CUDA/CUDA support via PyTorch

## Dependencies
- Python 3.8+
- `torch` (v2.0+)
- `tokenizers` (v0.15+)
- `numpy`
- `tqdm` (for training)

Install requirements:

```bash
pip install torch tokenizers numpy tqdm
```

## Quick Start: Inference
```python
# prompt = "Once upon a time there was a clever fox"
python inference.py
```
Output:
```
Once upon a time there was a clever fox who discovered a hidden grove filled with magical berries...
```

## Architecture Overview
### Core Components
- **Decoder-Only Structure** (4 layers, 160-dim embeddings)
- **Grouped Query Attention** (8 heads, 2 groups)
- **SwiGLU Activation** in Feed-Forward Networks
- **Rotary Positional Embeddings** (RoPE)
- **RMSNorm** for stable training
- **BPE Tokenizer** (12k vocab, byte-level fallback)

### Hyperparameters
| Component          | Specification              |
|--------------------|----------------------------|
| Embedding Dimension| 160                        |
| Attention Heads    | 8                          |
| Query Groups       | 2                          |
| FFN Hidden Dim     | 170                        |
| Layers             | 4                          |
| Dropout            | 0.2                        |
| Learning Rate      | 3e-4 (cosine schedule)     |
| Batch Size         | 64                         |
| Context Length     | 256 tokens                 |

## Training Configuration
- **Dataset**: TinyStories (60M tokens)
- **Optimizer**: AdamW with gradient clipping (‖1.0)
- **Schedule**: 650-step warmup + cosine decay
- **Regularization**: Dropout (0.2), Weight Tying
- **Hardware**: Trains on single GPU in ~2 hours

## Advanced Features
### Memory-Efficient Attention
```python
class MultiHeadAttention:
    def __init__(self, num_groups=2):
        # GQA implementation
        self.k_linear = nn.Linear(d_model, num_groups * head_dim)
        self.v_linear = nn.Linear(d_model, num_groups * head_dim)
```

### SwiGLU FFN
```python
class FeedForward:
    def forward(self, x):
        x_gate, x_value = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x_gate) * x_value)  # SwiGLU
```

## Training Workflow
1. Prepare tokenizer:
```python
# tokenizer.py
tokenizer.train(files=["Data_small.txt"], trainer=trainer)
tokenizer.save("tokenizer.json")
```

2. Start training:
```python
# Model.py
train(resume_checkpoint="checkpoint_epoch_0.pt")
```

3. Monitor metrics:
```
Epoch 0 | Batch 512 | LR 2.95e-4 | Loss 3.214 | Grad Norm 0.92
```

## Inference Customization
Adjust generation parameters in `inference.py`:
```python
max_length = 250    # Max generation length
temperature = 0.75  # 0.1-1.5 range recommended
prompt = "In a futuristic city..."  # Custom input
```

## Model Comparisons
| Metric           | AtomicGPT       | NanoGPT (15M) |
|------------------|-----------------|---------------|
| Parameters       | 1.1M            | 15M           |
| Perplexity       | 12.4            | 9.8           |
| VRAM Usage       | 450MB           | 1.2GB         |


## Acknowledgements
- TinyStories dataset (https://huggingface.co/datasets/roneneldan/TinyStories)
- Llama architecture reference
- PyTorch Lightning team for optimization tips
```
