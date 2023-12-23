import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout = 0.2

class FeedForward(nn.Module):
    def __init__(self, n_embed):
       super().__init__()
       self.net = nn.Sequential(
          nn.Linear(n_embed, 4*n_embed),
          nn.ReLU(),
          nn.Linear(4*n_embed, n_embed),
          nn.Dropout(dropout)
       )
    
    def forward(self, x):
       return self.net(x)

class AttentionHead(nn.Module):
    def __init__(self, n_embed, n_heads):
      super().__init__()
      self.key = nn.Linear(n_embed, n_heads, bias= False)
      self.value = nn.Linear(n_embed, n_heads, bias= False)
      self.query = nn.Linear(n_embed, n_heads, bias= False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).to(device))
      self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
      B, T, C = x.shape
      k = self.key(x)
      v = self.value(x)
      q = self.query(x)
    
      wei = q @ k.transpose(-2, -1) / (C ** 0.5)

      wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
      wei = F.softmax(wei, dim=-1)
      wei = self.dropout(wei)

      out = wei @ v
      return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads, head_size):
      super().__init__()
      self.heads = nn.ModuleList([AttentionHead(n_embed, head_size) for _ in range(num_heads)])
      self.proj = nn.Linear(n_embed, n_embed)
      self.dropout = nn.Dropout(dropout)

    def forward(self,x):
       out =  torch.cat([head(x) for head in self.heads], dim=-1)
       out = self.dropout(self.proj(out))
       return out
