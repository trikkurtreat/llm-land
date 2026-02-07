import requests
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch

def download_save_get_data():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    with open("the-verdict.txt", "wb") as f:
        response = requests.get(url)
        if response.status_code == 200:
            f.write(response.content)

    with open("the-verdict.txt", "r", encoding = "utf-8") as f:
        raw_text = f.read()
    return raw_text

class DS(Dataset):
    def __init__(self, text, tokenizer, context_length, stride):
        super().__init__()
        self.inputs = []
        self.targets = []
        text = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})
        t_len = len(text)
        for i in range(0, t_len-context_length-1, stride):
            cur_input = text[i:i+context_length]
            cur_target = text[i+1:i+context_length+1]
            self.inputs.append(torch.tensor(cur_input))
            self.targets.append(torch.tensor(cur_target))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def get_data_loader(text, context_length, stride, batch_size, shuffle = False, num_workers = 0, drop_last = True):
    tokenizer = tiktoken.get_encoding("gpt2")
    ds = DS(text, tokenizer, context_length, stride)
    dl = DataLoader(ds, batch_size, shuffle, num_workers=num_workers, drop_last=drop_last)
    return dl

def get_train_val_test_split_data(context_length, train_split = 0.7, test_split = 0.1):
    rtext = download_save_get_data()
    #print(rtext[-99:])

    totlen = len(rtext)

    train_end_index = int(totlen*train_split)
    test_start_index = int((1-test_split)*totlen)

    traintext = rtext[:train_end_index]
    valtext = rtext[train_end_index:test_start_index]
    testtext = rtext[test_start_index:]

    traindl = get_data_loader(traintext, context_length, context_length, 2)
    valdl = get_data_loader(valtext, context_length, context_length, 2)
    testdl = get_data_loader(testtext, context_length, context_length, 2)

    return traindl, valdl, testdl

def main():
    rd = download_save_get_data()
    print(rd[-99:])

    traindl, valdl, testdl = get_train_val_test_split_data(256)

    for i, (x,y) in enumerate(traindl):
        print(i, x.shape, y.shape)

if __name__ == '__main__':
    main()