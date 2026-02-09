import torch
import sentencepiece as spm

class LlamaTokenizer():
    def __init__(self, tokfile):
        sp = spm.SentencePieceProcessor()
        sp.load(tokfile)
        self.tokenizer = sp
    def encode(self, x):
        return self.tokenizer.encode(x, out_type = int)
    def decode(self, x):
        return self.tokenizer.decode(x)

def text_to_tok(text, tokenizer):
    toks = torch.unsqueeze(torch.tensor(tokenizer.encode(text)), dim = 0)
    print(text, toks)
    return toks

def tok_to_text(toks, tokenizer):
    toks = torch.squeeze(toks,dim=0).tolist()
    return tokenizer.decode(toks)

def generate_text(inp, model, device, max_gen_tokens, context_length, temperature = 1.0, top_k = None):
    inp = inp.to(device)
    model.to(device)
    model.eval();
    with torch.no_grad():
        for i in range(max_gen_tokens):
            cur_inp = inp[:, -context_length:]
            logits = model(cur_inp)
            logits = logits[:, -1, :] #[bs, cl, vocabsize] -> [bs, vocabsize]
            if temperature != 1.0:
                logits = logits / (temperature + 1e-5)
            if top_k is not None:
                tops, _ = torch.topk(logits, top_k)
                maxtops = tops[:, -1]
                logits = torch.where(condition = logits<maxtops, input = torch.tensor(-torch.inf), other = logits)
                probs = torch.softmax(logits, dim = -1)
                next_toks = torch.multinomial(probs, num_samples=1)

            else : 
                next_toks = torch.argmax(logits, dim = -1, keepdim = True)
            inp = torch.concat((inp,next_toks), dim = -1)
    return inp
