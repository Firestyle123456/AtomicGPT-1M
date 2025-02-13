import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from torch.utils.checkpoint import checkpoint

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.scale

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_groups if num_groups else num_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, self.num_groups * self.head_dim, bias=False)
        self.v_linear = nn.Linear(d_model, self.num_groups * self.head_dim, bias=False)
        self.out_linear = nn.Linear(d_model, d_model, bias=False)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False) 

    def _apply_rope(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :] 
        sin = emb.sin()[None, None, :, :]
        return apply_rope(x, sin, cos)


    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)

        if self.num_groups != self.num_heads:
            k = k.repeat_interleave(self.num_heads//self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_heads//self.num_groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_linear(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.2):
        super().__init__()
        hidden_dim = int(2 * d_ff / 3)
        self.w1 = nn.Linear(d_model, hidden_dim * 2, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_gate, x_value = self.w1(x).chunk(2, dim=-1)
        x = F.silu(x_gate) * x_value
        x = self.dropout(x)
        return self.w2(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.2, num_groups=None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, num_groups)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_norm = self.norm1(x)
        attn_output = self.self_attn(x_norm, mask)
        x = x + self.dropout(attn_output)
        
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=160, num_layers=4, num_heads=8, 
                 d_ff=170, dropout=0.2, use_checkpointing=True, 
                 num_groups=4):  #Hyperparameters
        super().__init__()
        self.d_model = d_model
        self.use_checkpointing = use_checkpointing

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, num_groups)
            for _ in range(num_layers)
        ])

        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        self.linear.weight = self.embedding.weight
        self.norm_final = RMSNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model**-0.5)
                
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(x)
        
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                x = checkpoint(layer, x, mask, use_reentrant=False) 
            else:
                x = layer(x, mask)
                
        return self.linear(self.norm_final(x))

    def generate_mask(self, sz):
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=self.embedding.weight.device), diagonal=1)
    
class TextDataset(Dataset):
    def __init__(self, tokenizer_path, text_file, block_size=256):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.text_file = text_file
        self.block_size = block_size
        self.file_size = os.path.getsize(text_file)
        self.total_blocks = (self.file_size // (block_size * 4))  
        
    def __len__(self):
        return self.total_blocks

    def __getitem__(self, idx):
        start_pos = idx * self.block_size * 4  
        with open(self.text_file, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(start_pos)
            text_chunk = f.read(self.block_size * 10)  # Read small chunk
            encodings = self.tokenizer.encode(text_chunk).ids
        
        if len(encodings) < self.block_size + 1:
            encodings += [0] * (self.block_size + 1 - len(encodings))
            
        inputs = torch.tensor(encodings[:self.block_size], dtype=torch.long)
        targets = torch.tensor(encodings[1:self.block_size+1], dtype=torch.long)
        return inputs, targets
    
def train(resume_checkpoint=None):

    # Hyperparameters
    warmup_steps = 650
    batch_size = 64
    epochs = 2
    max_lr = 3e-4
    save_dir = "saved_models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = Tokenizer.from_file("./tokenizer.json")
    model = Transformer(
        vocab_size=tokenizer.get_vocab_size(),
        num_groups=2,
        use_checkpointing=True
    ).to(device)

    # Dataset and DataLoader
    train_dataset = TextDataset("tokenizer.json", "Data_small.txt")
    generator = torch.Generator().manual_seed(42) 
    
    start_epoch = 0
    start_batch_idx = 0
    total_batches_processed = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.RandomSampler(
            train_dataset, 
            generator=generator
        ),
        shuffle=False 
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=0.0)
    total_training_steps = epochs * len(train_loader)

    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch_idx = checkpoint['batch_idx'] + 1
        total_batches_processed = checkpoint['total_batches']
        generator = checkpoint['dataloader_generator']
        torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_train_loss = 0.0
        
        data_iter = enumerate(train_loader)
        if epoch == start_epoch and start_batch_idx > 0:
            for _ in range(start_batch_idx):
                next(data_iter)

        for batch_idx, (inputs, targets) in data_iter:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            total_batches_processed += 1

            if total_batches_processed < warmup_steps:
                lr = max_lr * (total_batches_processed / warmup_steps)
            else:
                decay_ratio = (total_batches_processed - warmup_steps) / (total_training_steps - warmup_steps)
                lr = max_lr * 0.5 * (1 + math.cos(math.pi * decay_ratio))
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            mask = model.generate_mask(inputs.size(1))
            outputs = model(inputs, mask=mask)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()

            if total_batches_processed % 1000 == 0:
                checkpoint = {
                    'total_batches': total_batches_processed,
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'lr': lr,
                    'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    'numpy_rng_state': np.random.get_state(),
                    'python_rng_state': random.getstate(),
                    'dataloader_generator': generator.get_state(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(save_dir, f"intermediate_batch_{total_batches_processed}.pt")
                )
                print(f"\nSaved checkpoint at batch {total_batches_processed}")

            print(f"Epoch {epoch} | Batch {batch_idx} | LR {lr:.2e} | "
                      f"Loss {loss.item():.4f} | Grad Norm {grad_norm:.2f}")

        epoch_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss / len(train_loader),
            'total_batches': total_batches_processed,
            'dataloader_generator': generator.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        torch.save(
            epoch_checkpoint,
            os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        )

# To start training, uncomment the line below
# train()