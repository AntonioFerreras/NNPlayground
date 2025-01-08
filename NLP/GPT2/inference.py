import torch
import torch.nn as nn
import torch.nn.functional as F
import model

num_return_sequences = 5
max_length = 30

# autodetect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: %s" % device)

# llm = model.GPT.from_pretrained('gpt2')
llm = model.GPT(model.GPTConfig())
llm.eval()
llm.to(device)

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (T,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (B, T)
x = tokens.to(device)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = llm(x)
        # take the logits at the last position
        next_token_logits = logits[:, -1, :]  # (B, vocab_size)
        # convert to probabilities
        next_token_probs = F.softmax(next_token_logits, dim=-1)  # (B, vocab_size)
        # topk 
        topk_probs, topk_indices = torch.topk(next_token_probs, 50, dim=-1)  # (B, topk)
        # sample from the topk
        idx = torch.multinomial(topk_probs, 1)  # (B, 1)
        xcol = torch.gather(topk_indices, -1, idx)  # (B, 1)

        x = torch.cat((x, xcol), dim=1)

# print the generated sequences
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    print('>', enc.decode(tokens))