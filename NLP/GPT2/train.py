import torch
import torch.nn as nn
import torch.nn.functional as F
import model
from data import get_shakespeare_dataloader

num_return_sequences = 5
max_length = 30

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# autodetect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: %s" % device)

# load dataset and dataloader
batch_size = 128
seq_length = 32
shakespeare_text = open('tiny_shakespeare.txt').read()
dataloader = get_shakespeare_dataloader(batch_size, seq_length, shakespeare_text)

# calculate number of batches in an epoch
num_batches = len(dataloader)
print("Number of batches in an epoch: %d" % num_batches)

# load the model
# llm = model.GPT.from_pretrained('gpt2')
llm = model.GPT(model.GPTConfig())
llm.to(device)

# train
optimizer = torch.optim.AdamW(llm.parameters(), lr=3e-4)
for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
    
    optimizer.zero_grad()
    logits, loss = llm(input_seq, target_seq)
    print("batch %d, loss %.3f" % (batch_idx, loss.item()))

    loss.backward()
    optimizer.step()

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (T,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (B, T)
x = tokens.to(device)
