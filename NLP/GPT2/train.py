import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import model
from data import get_shakespeare_dataloader
import time

num_return_sequences = 5
max_length = 30

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# autodetect device
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print("using device: %s" % device)

# load dataset and dataloader
batch_size = 524288
mini_batch_size = 16
seq_length = 1024
assert batch_size % (mini_batch_size * seq_length) == 0, "batch size must be divisible by mini batch size"
grad_accum_steps = batch_size // (mini_batch_size * seq_length)
shakespeare_text = open('tiny_shakespeare.txt').read()
dataloader = get_shakespeare_dataloader(batch_size, seq_length, shakespeare_text)

torch.set_float32_matmul_precision('high')

# load the model
llm = model.GPT(model.GPTConfig(vocab_size=50304))
llm.to(device)
llm = torch.compile(llm)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# train
optimizer = llm.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_name)
step = 0
lr = 0.0
for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
    t0 = time.time()

    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
    
    # forward then backward
    optimizer.zero_grad()
    with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
        logits, loss = llm(input_seq, target_seq)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)

    # step the scheduler
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    # timing and logging
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # ms
    tokens_per_sec = batch_size * seq_length / (t1 - t0)
    print("batch %d, loss %.3f, lr %.2e, grad_norm %.4f, time %.2f ms, tok/sec %.2f" % (batch_idx, loss.item(), lr, grad_norm, dt, tokens_per_sec))

    step += 1

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (T,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (B, T)
x = tokens.to(device)
