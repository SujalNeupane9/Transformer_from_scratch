from ReformerV2 import Reformer
import torch
import torch.nn as nn
import torch.nn.functional as F

vocab_size = 10000
num_hashes = 6
num_buckets = 8
emb_dim = 512
num_heads = 8
hidden_dim = 2048
num_layers = 12
max_seq_len = 512
chunk_size = 64

model = Reformer(vocab_size, emb_dim, num_heads, hidden_dim, num_layers, max_seq_len, num_hashes, num_buckets, chunk_size)

input_seq = torch.randint(low=0, high=vocab_size, size=(4, 512))

output = model(input_seq)
print(output.shape) # should be torch.Size([4, 512, 10000])