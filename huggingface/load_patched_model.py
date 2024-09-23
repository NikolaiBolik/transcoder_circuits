import torch
import sys
import os
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sae_training
 
def patch_mlp_with_tc(model, sae_weights, layer_index):
    """
    Patch the model's MLP layer at the given layer_index with new weight matrices from the SAE weights.
    
    Args:
    - model: Pretrained model to be patched.
    - sae_weights: SAE weights loaded from file.
    - layer_index: Index of the model layer to be patched.
    
    Returns:
    - Patched model.
    """
    with torch.no_grad():
        print(f"Patching layer {layer_index}...")

        mlp = model.model.layers[layer_index].mlp

        W_enc = sae_weights['state_dict']['W_enc']
        W_dec = sae_weights['state_dict']['W_dec']
        
        tc_dim = W_enc.shape[1]
        input_dim = W_enc.shape[0]

        print("==>>>in_features:", input_dim)
        print("==>>>out_features:", tc_dim)

        mlp.up_proj = nn.Linear(in_features=input_dim, out_features=tc_dim, bias=False)
        mlp.down_proj = nn.Linear(in_features=tc_dim, out_features=input_dim, bias=False)
        mlp.gate_proj = nn.Linear(in_features=input_dim, out_features=tc_dim, bias=False)

        mlp.up_proj.weight.data.copy_(W_enc.T)
        mlp.down_proj.weight.data.copy_(W_dec.T)
        mlp.gate_proj.weight.data.fill_(1)

        print(f"Patched MLP layer {layer_index} with tc_dim={tc_dim}")

    return model
    

def load_model_and_extract_activations(model, input_text, layer_index, activation_name="up_proj"):
    """
    Helper function to extract the activation of a specific layer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(input_text, return_tensors="pt")
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    layer = model.model.layers[layer_index].mlp.__getattr__(activation_name)
    hook_handle = layer.register_forward_hook(get_activation(activation_name))

    with torch.no_grad():
        model(**inputs)

    activation = activations[activation_name]
    hook_handle.remove()

    return activation


def compare_weights(upd_model, orig_model, sae_weights, layer_index):
    """
    Compare the MLP weights of the updated model with the original model and SAE weights.
    """
    def show_mlp_weights(layer_index, model_to_show, model_name):
        print(f"====================>> {model_name} <<======================")
        print("mlp.up_proj.weight:", model_to_show.model.layers[layer_index].mlp.up_proj.weight)
        print("mlp.down_proj.weight:", model_to_show.model.layers[layer_index].mlp.down_proj.weight)
        print("mlp.gate_proj.weight:", model_to_show.model.layers[layer_index].mlp.gate_proj.weight)
        print("mlp.up_proj.weight.shape:", model_to_show.model.layers[layer_index].mlp.up_proj.weight.shape)
        print("mlp.down_proj.weight.shape:", model_to_show.model.layers[layer_index].mlp.down_proj.weight.shape)
        print("mlp.gate_proj.weight.shape:", model_to_show.model.layers[layer_index].mlp.gate_proj.weight.shape)

    show_mlp_weights(layer_index, upd_model, "Updated Model")
    show_mlp_weights(layer_index, orig_model, "Original Model")

    print("====================>> SAE Weights <<======================")
    print("W_enc.shape:", sae_weights['state_dict']['W_enc'].shape)
    print("W_dec.shape:", sae_weights['state_dict']['W_dec'].shape)
    print("W_enc:", sae_weights['state_dict']['W_enc'])
    print("W_dec:", sae_weights['state_dict']['W_dec'])


def compare_activations(upd_model, orig_model, input_text, layer_index):
    """
    Compare the activations of the updated model and the original model.
    """
    print("====================>> Activations with Updated Weights <<======================")
    up_proj_activation_upd = load_model_and_extract_activations(upd_model, input_text, layer_index)
    print("up_proj_activation_upd:", up_proj_activation_upd)
    print("up_proj_activation_upd.shape:", up_proj_activation_upd.shape)

    print("====================>> Activations with Original Weights <<======================")
    up_proj_activation_ori = load_model_and_extract_activations(orig_model, input_text, layer_index)
    print("up_proj_activation_ori:", up_proj_activation_ori)
    print("up_proj_activation_ori.shape:", up_proj_activation_ori.shape)



if __name__ == '__main__':
    model_name = "codellama/CodeLlama-7b-Instruct-hf"
    sae_weights_path = "/scratch/core/minxue/transcoder_circuits/codellama-transcoders-4ef/60segxwj/final_sparse_autoencoder_CodeLlama-7b-Instruct-hf_blocks.19.ln2.hook_normalized_16384.pt"
    input_text = "hello world"
    layer_index = 19  # The index of the layer to patch

    model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
    sae_weights = torch.load(sae_weights_path)
    patched_model = patch_mlp_with_tc(model, sae_weights, layer_index)

    # Compare the weights and activations
    orig_model = AutoModelForCausalLM.from_pretrained(model_name)
    compare_weights(patched_model, orig_model, sae_weights, layer_index)
    compare_activations(patched_model, orig_model, input_text, layer_index)
