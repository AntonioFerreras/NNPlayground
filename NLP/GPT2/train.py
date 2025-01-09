import torch
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
batch_size = 16
seq_length = 1024
shakespeare_text = open('tiny_shakespeare.txt').read()
dataloader = get_shakespeare_dataloader(batch_size, seq_length, shakespeare_text)

torch.set_float32_matmul_precision('high')

# load the model
llm = model.GPT(model.GPTConfig(vocab_size=50304))
llm.to(device)
llm = torch.compile(llm)

# train
optimizer = torch.optim.AdamW(llm.parameters(), lr=3e-4)
for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
    t0 = time.time()

    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
    
    optimizer.zero_grad()
    with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
        logits, loss = llm(input_seq, target_seq)
    loss.backward()
    optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # ms
    tokens_per_sec = batch_size * seq_length / (t1 - t0)
    print("batch %d, loss %.3f, time %.2f ms, tok/sec %.2f" % (batch_idx, loss.item(), dt, tokens_per_sec))

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (T,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (B, T)
x = tokens.to(device)
