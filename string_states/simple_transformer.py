import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, input_size, d_model, d_head):
        super().__init__()
        self.d_head = d_head
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)
        self.W_O = nn.Linear(d_head, d_head, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(input_size, input_size)))

        # xavier initialization
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)


    def forward(self, x):
        # x is of size (batch_size, input_size, d_model)
        # get the query, key, and value
        Q = self.W_Q(x) # (batch_size, input_size, d_head)
        K = self.W_K(x) # (batch_size, input_size, d_head)
        V = self.W_V(x) # (batch_size, input_size, d_head)
        # get the attention weights
        A = torch.einsum("bid,bjd->bij", Q, K) / (self.d_head**0.5) 
        A = A.masked_fill(self.mask==0, float("-inf"))
        A = F.softmax(A, dim=-1) # the rows of A sum to 1
        # apply the attention weights
        O = torch.einsum("bij,bjd->bid", A, V) # this is the output of the attention head, we weight the values by the attention weights
        O = self.W_O(O) # (batch_size, input_size, d_model)
        return O 
    
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)

        # xavier initialization
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)

        
    def forward(self, x):
        # x is of size (batch_size, input_size, d_model)
        x = self.W_in(x) # (batch_size, input_size, d_mlp)
        x = F.relu(x)
        x = self.W_out(x) # (batch_size, input_size, d_model)
        return x

class Transformer(nn.Module):
    def __init__(self, d_vocab=2, d_model=16, input_size=3, d_head=4, n_head=4, d_mlp=4*16):
        super().__init__()
        assert d_model == d_head * n_head, "d_model should be equal to the product of d_head and n_head"
        self.input_size = input_size
        self.embedding = nn.Embedding(d_vocab, d_model)
        self.pos_embedding = nn.Embedding(input_size, d_model)
        self.heads = nn.ModuleList([Head(input_size, d_model, d_head) for _ in range(n_head)])
        self.mlp = MLP(d_model, d_mlp)
        self.unembedding = nn.Linear(d_model, d_vocab)

        # xavier initialization
        nn.init.xavier_uniform_(self.unembedding.weight)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)

    def forward(self, x):
        # x is of size (batch_size, input_size)
        # embed the input
        x = self.embedding(x) + self.pos_embedding(torch.arange(self.input_size, device=x.device))
        # apply the attention heads, stack them
        x = x + torch.cat([head(x) for head in self.heads], dim=-1) # (batch_size, input_size, d_model)
        # apply the MLP
        x = x + self.mlp(x)
        # unembed the output
        x = self.unembedding(x)
        return x
    
    def predict_probs(self, x):
        # pass input through the model
        logits = self.forward(x)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        return probs
    


class TransformerLayer(nn.Module):
    def __init__(self, d_model, input_size, d_head, n_head, d_mlp):
        super().__init__()
        assert d_model == d_head * n_head, "d_model should be equal to the product of d_head and n_head"
        self.heads = nn.ModuleList([Head(input_size, d_model, d_head) for _ in range(n_head)])
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x):
        # apply the attention heads, stack them
        x = x + torch.cat([head(x) for head in self.heads], dim=-1) # (batch_size, input_size, d_model)
        # apply the MLP
        x = x + self.mlp(x)
        return x

    
class MultilayerTransformer(nn.Module):
    def __init__(self, d_vocab=2, d_model=16, input_size=3, d_head=4, n_head=4, d_mlp=4*16, n_layers=2):
        super().__init__()
        self.input_size = input_size
        self.embedding = nn.Embedding(d_vocab, d_model)
        self.pos_embedding = nn.Embedding(input_size, d_model)
        self.layers = nn.ModuleList([TransformerLayer(d_model, input_size, d_head, n_head, d_mlp) for _ in range(n_layers)])
        self.unembedding = nn.Linear(d_model, d_vocab)

    def forward(self, x):
        # x is of size (batch_size, input_size)
        # embed the input
        x = self.embedding(x) + self.pos_embedding(torch.arange(self.input_size, device=x.device))
        # pass through each transformer layer
        for layer in self.layers:
            x = layer(x)
        # unembed the output
        x = self.unembedding(x)
        return x
    
    def predict_probs(self, x):
        # pass input through the model
        logits = self.forward(x)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        return probs

class MultilayerTransformer(nn.Module):
    def __init__(self, d_vocab=2, d_model=16, input_size=3, d_head=4, n_head=4, d_mlp=4*16, n_layers=2):
        super().__init__()
        self.input_size = input_size
        self.embedding = nn.Embedding(d_vocab, d_model)
        self.pos_embedding = nn.Embedding(input_size, d_model)
        self.layers = nn.ModuleList([TransformerLayer(d_model, input_size, d_head, n_head, d_mlp) for _ in range(n_layers)])
        self.unembedding = nn.Linear(d_model, d_vocab)

    def forward(self, x):
        # x is of size (batch_size, input_size)
        # embed the input
        x = self.embedding(x) + self.pos_embedding(torch.arange(self.input_size, device=x.device))
        # pass through each transformer layer
        for layer in self.layers:
            x = layer(x)
        # unembed the output
        x = self.unembedding(x)
        return x
    
    def predict_probs(self, x):
        # pass input through the model
        logits = self.forward(x)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        return probs




