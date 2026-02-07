import torch
import tiktoken
from Engine import GPTModel
from Helpers import text_to_tok, tok_to_text, generate_text
from Data import get_train_val_test_split_data

def calc_loss_batch(x, y, model, device):
    x = x.to(device)
    y = y.to(device)
    model.to(device)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(torch.flatten(logits, 0, 1), torch.flatten(y, 0, 1))
    return loss

def calc_loss_loader(dl, model, device, num_batches = None):
    if num_batches is None:
        num_batches = len(dl)
    else:
        num_batches = min(len(dl), num_batches)
    totloss = 0.0
    for i, (x, y) in enumerate(dl):
        if i < num_batches:
            totloss = totloss + calc_loss_batch(x, y, model, device)
        else:
            break
    return totloss/num_batches

def trainloop(traindl, valdl, model, device, epochs, optimizer, val_freq, tokenizer, startseq = "Trump is the president of"):
    trainlosses, vallosses, toksseenhist = [],[],[]
    globalstep, toksseen = 0,0
    for e in range(epochs):
        print("Epoch : ", e)
        for x,y in traindl:
            model.train();
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()
            globalstep+=1
            toksseen += torch.numel(x)
            if globalstep % val_freq == 0:
                trainloss, valloss = get_losses_val_step(traindl, valdl, model, device)
                trainlosses.append(trainloss)
                vallosses.append(valloss)
                toksseenhist.append(toksseen)
                print_text_val_step(model, device, startseq, tokenizer)
                print("trainloss : ", trainloss, ", valloss : ", valloss)
    return trainlosses, vallosses, toksseenhist


def get_losses_val_step(traindl, valdl, model, device):
    model.eval();
    with torch.no_grad():
        trainloss = calc_loss_loader(traindl, model, device, 5)
        valloss = calc_loss_loader(valdl, model, device, 5)
    return trainloss, valloss

def print_text_val_step(model, device, startseq, tokenizer):
    toks = text_to_tok(startseq, tokenizer)
    gentoks = generate_text(toks, model, device, 50, 1.0, 20)
    gentext = tok_to_text(gentoks, tokenizer)
    print("Val Step Generation : ", gentext)


def main():
    GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
    }
    tokenizer = tiktoken.get_encoding("gpt2")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    traindl, valdl, testdl = get_train_val_test_split_data(GPT_CONFIG_124M["context_length"])
    model = GPTModel(GPT_CONFIG_124M)
    optimizer = torch.optim.AdamW(model.parameters(), 0.0005, weight_decay=0.1)
    trainlosses, vallosses, toksseenhist = trainloop(traindl, valdl, model, device, 10, optimizer, 10, tokenizer)

    save_path = "./savedweights.pth"
    torch.save(model, save_path)
    print("model saved at path : ", save_path)

if __name__ == '__main__':
    main()