from Engine import LlamaModel
from huggingface_hub import login, hf_hub_download
from Helpers import LlamaTokenizer, text_to_tok, tok_to_text, generate_text
import json
import torch

def login_into_hf_hub():
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        access_token = config["HF_ACCESS_TOKEN"]
    login(token=access_token)

def assign(left, right, tensor_name="unknown"):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
    
    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right)
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

    return left 


def permute(w: torch.Tensor, n_heads, out_dim, in_dim):
    return (w.view(n_heads, out_dim // n_heads // 2, 2, in_dim)
             .transpose(1, 2)          # put axis 2 next to heads
             .reshape(out_dim, in_dim))


def load_weights_into_llama(model, param_config, params):

    cfg = param_config
    
    model.tok_emb.weight = assign(model.tok_emb.weight, params["tok_embeddings.weight"])

    for l in range(param_config["n_layers"]):

        # The original Meta/Llama checkpoints store Q and K so that the two numbers 
        # that form one complex RoPE pair sit next to each other inside the head dimension ("sliced" layout).
        # Our RoPE implementation, similar to the one in Hugging Face, expects an interleaved layout
        # For example, with n_heads=2 and head_dim = 8
        #                         ┌── pair 0 ──┐      ┌── pair 1 ──┐
        # Meta (sliced):    [ h0:  r0 r1 r2 r3,   h1:  r0 r1 r2 r3  ]
        # Ours & HF (interleaved):  [ h0: r0 r0 r1 r1 r2 r2 r3 r3  , h1: ... ]
        # For more information, please see the discussion in the PR: https://github.com/rasbt/LLMs-from-scratch/pull/747 
        
        # So, below, for q_raw and k_raw, we must re‑order the checkpoint weights using the slices_to_interleave helper

        q_raw = params[f"layers.{l}.attention.wq.weight"]
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            permute(q_raw, cfg["n_heads"], cfg["emb_dim"], cfg["emb_dim"])
        )
        k_raw = params[f"layers.{l}.attention.wk.weight"]
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            permute(k_raw, cfg["n_heads"], cfg["emb_dim"], cfg["emb_dim"])
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"layers.{l}.attention.wv.weight"]
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"layers.{l}.attention.wo.weight"]
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"layers.{l}.attention_norm.weight"]
        )

        # Load FeedForward weights
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"layers.{l}.feed_forward.w1.weight"]
        )
        # For some reason w2 and w3 are provided in the wrong order in the weights file
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"layers.{l}.feed_forward.w3.weight"]
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"layers.{l}.feed_forward.w2.weight"]
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"layers.{l}.ffn_norm.weight"]
        )

    # Load output layer weights
    model.final_norm.weight = assign(model.final_norm.weight, params["norm.weight"])
    model.out_head.weight = assign(model.out_head.weight, params["output.weight"])


def main():
    login_into_hf_hub()

    tokenizer_file = hf_hub_download(
        repo_id="meta-llama/Llama-2-7b",
        filename="tokenizer.model",
        local_dir="Llama-2-7b"
        )
    print("tokenizer downloaded")
    tokenizer = LlamaTokenizer(tokenizer_file)

    device = torch.device("cpu")
    LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
    }

    #llama = LlamaModel(LLAMA2_CONFIG_7B)
    
    # generate before weight load
    # outs = generate_text(text_to_tok("Barak Obama is the ", tokenizer), llama, device, 3, LLAMA2_CONFIG_7B["context_length"], 1.0, 50)
    # print(tok_to_text(outs, tokenizer))
    
    # load pretrained weights

    # weights_file = hf_hub_download(
    #     repo_id="meta-llama/Llama-2-7b",
    #     filename="consolidated.00.pth",
    #     local_dir="Llama-2-7b"
    #     )
    
    # weights = torch.load(weights_file, weights_only=True)
    # load_weights_into_llama(llama, LLAMA2_CONFIG_7B, weights)
    # llama.to(device);

    # torch.manual_seed(123)

    # token_ids = generate_text(
    #         model=llama,
    #         idx=text_to_tok("Every effort", tokenizer).to(device),
    #         max_new_tokens=10,
    #         context_size=LLAMA2_CONFIG_7B["context_length"],
    #         top_k=1,
    #         temperature=0.)

    # print("Output text:\n", tok_to_text(token_ids, tokenizer))

    # load instruction finetuned weights

    weights_file = hf_hub_download(
        repo_id="meta-llama/Llama-2-7b-chat",
        filename="consolidated.00.pth",
        local_dir="Llama-2-7b-chat"
        )
    print("instruct weigts file downloaded")
    weights = torch.load(weights_file, weights_only=True)
    model = LlamaModel(LLAMA2_CONFIG_7B)
    load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights)
    model.to(device);
    print("model loaded with weights and starting to generate")
    torch.manual_seed(123)

    token_ids = generate_text(text_to_tok("What do llamas eat?", tokenizer).to(device), model, device, 5
                              , LLAMA2_CONFIG_7B["context_length"], 1.0, 100)

    print("Output text:\n", tok_to_text(token_ids, tokenizer))

if __name__ == "__main__":
    main()