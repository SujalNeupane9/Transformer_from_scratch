import torch
import torch.nn as nn
import torch.nn.functional as F

class LSH(nn.Module):
    def __init__(self, num_hashes, emb_dim, num_buckets):
        super(LSH, self).__init__()
        self.num_hashes = num_hashes
        self.emb_dim = emb_dim
        self.num_buckets = num_buckets

        # create random projection vectors and bias terms
        self.projections = nn.Parameter(torch.randn(num_hashes, emb_dim))
        self.biases = nn.Parameter(torch.randn(num_hashes))

    def forward(self, x):
        # project the input vectors to the hash space
        hashes = torch.matmul(x, self.projections.transpose(0, 1))
        hashes += self.biases
        hashes = torch.floor_divide(hashes, self.num_buckets)

        # return the hashes as a long tensor
        return hashes.long()

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, emb_dim, max_seq_len):
        super(AxialPositionalEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        
        self.row_embeddings = nn.Parameter(torch.randn(max_seq_len, emb_dim ))  # //2
        self.col_embeddings = nn.Parameter(torch.randn(max_seq_len, emb_dim ))  # //2
        
    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}"
        
        row_pos = torch.arange(seq_len , device=x.device) #//2
        col_pos = torch.arange(seq_len , device=x.device) #//2
        row_embs = self.row_embeddings[row_pos, :].unsqueeze(0).expand(batch_size, -1, -1)
        col_embs = self.col_embeddings[col_pos, :].unsqueeze(0).expand(batch_size, -1, -1)
        
        if seq_len % 2 == 0:
            x_even = x[:, 0::2, :]
            x_odd = x[:, 1::2, :]
            x = torch.cat([x_even @ row_embs, x_odd @ col_embs], dim=1)
        else:
            x_even = x[:, 0::2, :]
            x_odd = x[:, 1::2, :]
            x = torch.cat([x_even @ row_embs, x_odd @ col_embs, x[:, -1, :].unsqueeze(1)], dim=1)
        
        return x

class ReformerLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, hidden_dim, chunk_size, dropout=0.1):
        super(ReformerLayer, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.chunk_size = chunk_size
        
        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(p=dropout)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        # Chunk input tensor along sequence length dimension
        chunks = x.chunk(x.size(1) // self.chunk_size, dim=1)
        chunked_output = []
        for chunk in chunks:
            chunk = chunk.permute(1, 0, 2)
            chunk, _ = self.self_attn(chunk, chunk, chunk)
            chunk = F.dropout(chunk, p=self.dropout, training=self.training)
            chunk = chunk.permute(1, 0, 2)
            chunked_output.append(chunk)
        x = torch.cat(chunked_output, dim=1)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        # Chunk input tensor along sequence length dimension
        chunks = x.chunk(x.size(1) // self.chunk_size, dim=1)
        chunked_output = []
        for chunk in chunks:
            chunked_output.append(self.feed_forward(chunk))
        x = torch.cat(chunked_output, dim=1)
        x = x + residual
        return x



class Reformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, hidden_dim, num_layers, max_seq_len, num_hashes, num_buckets,chunk_size, dropout=0.1):
        super(Reformer, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.dropout = dropout

        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.axial_position_emb = AxialPositionalEmbedding(emb_dim, max_seq_len)
        self.lsh = LSH(num_hashes, emb_dim, num_buckets)

        self.layers = nn.ModuleList([
            ReformerLayer(emb_dim, num_heads, hidden_dim,chunk_size, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        # encode the input sequence
        x = self.token_emb(x)
        x = self.axial_position_emb(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # generate LSH hashes for the input sequence
        hashes = self.lsh(x.view(-1, self.emb_dim))
        hashes = hashes.view(x.size(0), x.size(1), self.num_hashes)

        # apply the transformer layers
        for layer in self.layers:
            residual = x
            x = layer(x)
            x += residual

        # compute the logits and return the output
        x = self.fc(x)
        return x

