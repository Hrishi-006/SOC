import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)


max_iters = 1000
n_head = 6
n_layer = 6
dropout = 0.2
n_embd = 384

batch_size =64 
block_size =256 

with open('conversation_clean.txt', 'r', encoding='utf-8') as f:
    text = f.read()
def preprocess_conversations(text):
    # Split into conversation turns
    turns = [turn.strip() for turn in text.split('\n\n') if turn.strip()]

    processed = []
    for turn in turns:
        if turn.startswith('Human:'):
            processed.append(f"[HUMAN] {turn[len('Human:'):].strip()}")
        elif turn.startswith('Bot:'):
            processed.append(f"[BOT] {turn[len('Bot:'):].strip()}")

    return ' '.join(processed)

text = preprocess_conversations(text)
static_tokens = "[HUMAN][BOT]"
extra_chars = list(static_tokens + "0123456789.,!?():'\"- ")
chars = sorted(list(set(text).union(set(extra_chars))))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    try:
        return [stoi[c] for c in s]
    except KeyError as e:
        print(f"KeyError: Character {e} not found in vocab.")
        missing = set(c for c in s if c not in stoi)
        print(f"Missing characters: {missing}")
        raise

decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])


    return x.to(device), y.to(device)


eval_iters = 200
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

learning_rate = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

eval_interval = 100
for iter in range(max_iters):


    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()  
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
def train_model():
  for iter in range(max_iters):
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

train_model()

def chat(model, prompt, history=None, max_length=100):
    if history is None:
        history = []

    conv_history = "\n\n".join(
        f"[HUMAN] {h['human']}\n[BOT] {h['bot']}"
        for h in history
    )

    full_prompt = f"{conv_history}\n\n[HUMAN] {prompt}\n[BOT]" if history else f"[HUMAN] {prompt}\n[BOT]"
    input_ids = torch.tensor([encode(full_prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_length)

    response = decode(output[0][input_ids.shape[1]:].tolist())
    response = response.strip().split("[HUMAN]")[0].strip() 
    return response



history = []
max_history = 3  

while True:
    prompt = input("You: ").strip()
    if prompt.lower() in ["exit", "quit"]:
        break

    reply = chat(model, prompt, history[-max_history:] if history else None)
    print("Bot:", reply)

    history.append({'human': prompt, 'bot': reply})
    if len(history) > max_history + 2:  
        history = history[-max_history:]




def chat(model, prompt, history=None, max_length=100):
    if history is None:
        history = []

    conv_history = "\n\n".join(
        f"[HUMAN] {h['human']}\n[BOT] {h['bot']}"
        for h in history
    )

    full_prompt = f"{conv_history}\n\n[HUMAN] {prompt}\n[BOT]" if history else f"[HUMAN] {prompt}\n[BOT]"
    input_ids = torch.tensor([encode(full_prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_length)

    response = decode(output[0][input_ids.shape[1]:].tolist())
    response = response.strip().split("[HUMAN]")[0].strip() 
    return response



history = []
max_history = 3 ]

while True:
    prompt = input("You: ").strip()
    if prompt.lower() in ["exit", "quit"]:
        break

    reply = chat(model, prompt, history[-max_history:] if history else None)
    print("Bot:", reply)

    history.append({'human': prompt, 'bot': reply})
    if len(history) > max_history + 2: 
        history = history[-max_history:]
