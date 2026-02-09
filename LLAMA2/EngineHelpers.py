import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(emb_dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = self.eps + torch.mean(x**2, dim = -1, keepdim=True)
        normalized = x * torch.rsqrt(mean)
        return (normalized * self.weight).to(dtype = x.dtype)

    
class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForward(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = torch.nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype = cfg["dtype"], bias = False)
        self.fc2 = torch.nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype = cfg["dtype"], bias = False)
        self.silu = SiLU()
        self.fc3 = torch.nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype = cfg["dtype"], bias = False)
        
    def forward(self, x):
        f1 = self.fc1(x)
        f2 = self.fc2(x)
        int_out = self.silu(f1) * f2
        f3 = self.fc3(int_out)
        return f3

def precompute_rope_params(head_dim, context_length, base_rot = 10000):
    assert head_dim%2 == 0, "head dim should be divisible by 2"
    inv_freq = 1.0 / base_rot ** (torch.arange(0,head_dim, 2)[:head_dim//2].float()/head_dim)
    positions = torch.arange(context_length).float()
    angles = positions.unsqueeze(1) @ inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], -1)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return sin, cos

def compute_rope(x, sin, cos):
    # x [bs, num_heads, seq_len, head_dim]
    head_dim = x.shape[-1]
    seq_len = x.shape[-2]
    x1 = x[:, :, :, :head_dim//2]
    x2 = x[:, :, :, head_dim//2:]
    modified_x = torch.cat([-x2,x1], dim = -1) # [bs, cl, head_dim]
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, head_dim]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated_x = x * cos + modified_x * sin
    return rotated_x.to(dtype = x.dtype)


def main():
    precompute_rope_params(4, 6)



if __name__ == "__main__":
    main()
