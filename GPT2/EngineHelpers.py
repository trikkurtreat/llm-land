import torch

class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = torch.mean(x, dim = -1, keepdim = True)
        var = torch.var(x, dim = -1, keepdim = True, unbiased = False)
        normx = (x-mean)/(torch.sqrt(var)+self.eps)
        return normx * self.scale + self.shift

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x *(1 + torch.tanh((2/torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)))

class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim * 4)
                                          , GELU()
                                          , torch.nn.Linear(emb_dim * 4, emb_dim))
        
    def forward(self, x):
        return self.layers(x)

def main():
    a = torch.ones((2,2,4))
    print(a)
    ln = LayerNorm(a.shape[-1])
    print(ln(a))
    gl = GELU()
    print(gl(a))
    ff = FeedForward(a.shape[-1])
    print(ff(a).shape)

if __name__ == "__main__":
    main()
