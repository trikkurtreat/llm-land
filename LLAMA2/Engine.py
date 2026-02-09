import torch
from EngineHelpers import RMSNorm, FeedForward, compute_rope, precompute_rope_params

class MMHA(torch.nn.Module):
    def __init__(self, context_length, d_in, d_out, num_heads, dtype = None):
        super().__init__()
        assert d_out%num_heads==0, "d_out should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads
        self.W_query = torch.nn.Linear(d_in, d_out, dtype = dtype, bias = False)
        self.W_key = torch.nn.Linear(d_in, d_out, dtype = dtype, bias = False)
        self.W_value = torch.nn.Linear(d_in, d_out, dtype = dtype, bias = False)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.out_proj = torch.nn.Linear(d_out, d_out, bias = False, dtype = dtype)

        sin, cos = precompute_rope_params(self.head_dim, context_length)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    def forward(self, x): #[bs,cl,d_in]
        bs, cl, _ = x.shape
        q = self.W_query(x) #[bs,cl,d_out]
        k = self.W_key(x)
        v = self.W_value(x)
        q = q.view(bs, cl, self.num_heads, self.head_dim).transpose(1,2) #[bs,nh,cl,hd]
        k = k.view(bs, cl, self.num_heads, self.head_dim).transpose(1,2)

        q = compute_rope(q, self.sin, self.cos)
        k = compute_rope(k, self.sin, self.cos)

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
        self.norm1 = RMSNorm(config["emb_dim"])
        self.norm2 = RMSNorm(config["emb_dim"])
        self.att = MMHA(config["context_length"], config["emb_dim"], config["emb_dim"], config["n_heads"], config["dtype"])
        self.ff = FeedForward(config)

    def forward(self, x):
        shortcut = x

        x = self.norm1(x)
        x = self.att(x)
        x = x+shortcut

        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = x+shortcut

        return x

class LlamaModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(config["vocab_size"], config["emb_dim"], dtype = config["dtype"])
        self.trf_blocks = torch.nn.Sequential(*[TfmrBlock(config) for _ in range(config["n_layers"])])
        self.final_norm = RMSNorm(config["emb_dim"])
        self.out_head = torch.nn.Linear(config["emb_dim"], config["vocab_size"], bias = False, dtype = config["dtype"])
        
    def forward(self, x):
        bs, cl = x.shape
        tok_emb = self.tok_emb(x)
        x = tok_emb
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        x = self.out_head(x)
        return x

def model_memory_size(model, model_dtype = torch.float32):
    parameters, gradients = 0.0, 0.0
    for p in model.parameters():
        parameters+=p.numel()
        if p.requires_grad:
            gradients +=  p.numel()
    buffers = sum(b.numel() for b in model.buffers())
    el_size = torch.tensor(0, dtype = model_dtype).element_size()
    return (parameters + gradients + buffers) * el_size / (1024**3)

def main():
    LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
    }
    llama = LlamaModel(LLAMA2_CONFIG_7B)
    total_params = sum(p.numel() for p in llama.parameters())
    print("total number of parameters in llama 7B : ", total_params)
    print("size of model when using float 32 : ", model_memory_size(llama, torch.float32))
    print("size of model when using bfloat 16 : ", model_memory_size(llama, torch.bfloat16))

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")
    print(device)

    llama.to(device);g


if __name__ == '__main__':
    main()