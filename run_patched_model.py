import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface.load_patched_model import patch_mlp_with_tc 


model_name = "codellama/CodeLlama-7b-Instruct-hf"
sae_weights_path = "/scratch/core/minxue/transcoder_circuits/codellama-transcoders-4ef/60segxwj/final_sparse_autoencoder_CodeLlama-7b-Instruct-hf_blocks.19.ln2.hook_normalized_16384.pt"

model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
sae_weights = torch.load(sae_weights_path)
layer_index = 19

patched_model = patch_mlp_with_tc(model, sae_weights, layer_index) 
tok = AutoTokenizer.from_pretrained(model_name)
print(patched_model)
