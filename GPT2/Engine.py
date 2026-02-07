import torch
from EngineHelpers import LayerNorm, FeedForward

class MMHA(torch.nn.Module):
    def __init__(self, context_length, d_in, d_out, num_heads, drop_rate, qkv_bias):
        super().__init__()
        assert d_out%num_heads==0, "d_out should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads
        self.W_query = torch.nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias = qkv_bias)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.out_proj = torch.nn.Linear(d_out, d_out)

    def forward(self, x): #[bs,cl,d_in]
        bs, cl, _ = x.shape
        q = self.W_query(x) #[bs,cl,d_out]
        k = self.W_key(x)
        v = self.W_value(x)
        q = q.view(bs, cl, self.num_heads, self.head_dim).transpose(1,2) #[bs,nh,cl,hd]
        k = k.view(bs, cl, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(bs, cl, self.num_heads, self.head_dim).transpose(1,2)
        att_scores = q @ k.transpose(2,3) #[bs,nh,cl,hd] @ [bs,nh,hd,cl] = [bs,nh,cl,cl]
        att_scores.masked_fill_(self.mask.bool()[:cl,:cl], -torch.inf)
        att_weights = torch.softmax(att_scores, dim = -1) #[bs,nh,cl,cl]
        context_vectors = att_weights @ v #[bs,nh,cl,cl] @ [bs,nh,cl,hd] = [bs,nh,cl,hd]
        context_vectors = context_vectors.transpose(1,2).contiguous().view((bs,cl,self.num_heads*self.head_dim)) #[bs,cl,d_out]
        return self.out_proj(context_vectors)

class TfmrBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.att = MMHA(config["context_length"], config["emb_dim"], config["emb_dim"], config["n_heads"], config["drop_rate"], config["qkv_bias"])
        self.ff = FeedForward(config["emb_dim"])
        self.dropout = torch.nn.Dropout(config["drop_rate"])

    def forward(self, x):
        shortcut = x

        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x+shortcut

        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x+shortcut

        return x

class GPTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = torch.nn.Embedding(config["context_length"], config["emb_dim"])
        self.trf_blocks = torch.nn.Sequential(*[TfmrBlock(config) for _ in range(config["n_layers"])])
        self.dropout = torch.nn.Dropout(config["drop_rate"])
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = torch.nn.Linear(config["emb_dim"], config["vocab_size"])
        
    def forward(self, x):
        bs, cl = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(0, cl, device=x.device))
        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        x = self.out_head(x)
        return x



def main():
    print("mmha test")
    mmha = MMHA(6,4,4,2,0.0,False)
    a = torch.ones(2,6,4)
    print(a.shape)
    out = mmha(a)
    print(out.shape)

    print("\n\ntfmr test")
    GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
    }   
    tfmrb = TfmrBlock(GPT_CONFIG_124M)
    print(tfmrb(torch.ones(1,6,GPT_CONFIG_124M["emb_dim"])).shape)

    print("\n\ngpt model test")
    gptm = GPTModel(GPT_CONFIG_124M)
    print(gptm(torch.ones((2,6), dtype=torch.long)).shape)

    print(gptm.pos_emb_layer.weight.shape[0])

if __name__ == '__main__':
    main()