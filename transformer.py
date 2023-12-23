import torch
import torch.nn as nn
from layers import MultiHeadAttention, FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.attn = MultiHeadAttention(n_embed, n_head, head_size) # Communication
        self.ffwd = FeedForward(n_embed) # Computation (Think about the communication and relations )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self,x):
        x = x + self.attn(self.ln1(x)) # res + compution
        x = x + self.ffwd(self.ln2(x))

        return x