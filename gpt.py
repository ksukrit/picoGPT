import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer import TransformerBlock

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = max_iters//10
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200
n_embed = 384
n_head = 6
no_blocks = 6
torch.manual_seed(1337)

# Download data source
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt') as f:
  text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

# encoder decoder (Tokenization equivalent for Char-level) BPE or SentancePiece in the future

# We can either have very small codebooks (mappings) and longer sequence lenghts or the opposite
ctoi = {ch:i for i,ch in enumerate(chars)}
itoc = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [ctoi[c] for c in s]
decode = lambda l: ''.join([itoc[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
train_size = int(len(text)*0.9)
train_data = data[:train_size]
test_data = data[train_size:]

def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data)- block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # 4x8 tensor batch_size,block_size
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_losses():
    out = {}
    model.eval()
    for split in ['train', 'test']:
       losses = torch.zeros(eval_iters)
       for i in range(eval_iters):
          X, Y = get_batch(split)
          _, loss = model(X, Y)
          losses[i] = loss.item()
          out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch('train')


# for b in range(batch_size): # batch dim
#   for t in range(block_size): # time dim
#     context = xb[b, :t+1]
#     target = yb[b, t]
#     print(f"Context is {context} and target is {target}")


class BigramModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding = nn.Embedding(block_size, n_embed)

    self.blocks = nn.Sequential(*[TransformerBlock(n_embed, n_head) for _ in range(no_blocks)])
    self.ln_f = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size) # To get logits
  def forward(self, idx, targets=None):
    B,T = idx.shape

    # idx (B,T)
    token_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emdb = self.position_embedding(torch.arange(T, device=device))
    x = token_emb + pos_emdb # (B,T,C)
    x = self.blocks(x) # (B,T,C)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B,T,vocab_size)

    B,T,C = logits.shape

    if targets is None:
        loss = None
    else:
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits,targets)

    return logits,loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits, _ = self(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

model = BigramModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_losses()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
    
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(idx = context, max_new_tokens=500)[0].tolist()))