import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class MyCustomModelForCausalLM(AutoModelForCausalLM):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, layer_index, sae_weights_path, *model_args, **kwargs):
        kwargs['ignore_mismatched_sizes'] = True
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        with torch.no_grad():
            print("Applying custom transformation to layer 19")

            mlp = model.model.layers[layer_index].mlp
            
            mlp.up_proj = torch.nn.Linear(in_features=4096, out_features=65536, bias=False)
            mlp.down_proj = torch.nn.Linear(in_features=65536, out_features=4096, bias=False)
            mlp.gate_proj = torch.nn.Linear(in_features=4096, out_features=65536, bias=False)
            mlp.gate_proj.weight.data.fill_(1) 
            
            sae_weights = torch.load(sae_weights_path)

            W_enc = sae_weights['state_dict']['W_enc']  # shape: [4096, 65536]
            W_dec = sae_weights['state_dict']['W_dec']  # shape: [65536, 4096]

            mlp.up_proj.weight.data.copy_(W_enc.T)  
            mlp.down_proj.weight.data.copy_(W_dec.T)  

 
        return model



def load_model_and_extract_activation(model, input_text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(input_text, return_tensors="pt")
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    layer = model.model.layers[19].mlp.up_proj
    hook_handle = layer.register_forward_hook(get_activation("up_proj"))

    with torch.no_grad():
        model(**inputs)

    up_proj_activation = activations["up_proj"]
    hook_handle.remove()

    return up_proj_activation




model_name = "codellama/CodeLlama-7b-Instruct-hf"
sae_weights_path = "/scratch/core/minxue/transcoder-distribute/codellama-transcoders-layer19-ef16-codeparrot/d8lcir9q/final_sparse_autoencoder_CodeLlama-7b-Instruct-hf_blocks.19.ln2.hook_normalized_65536.pt"
layer_index = 19
input_text = "hello world"

upd_model = MyCustomModelForCausalLM.from_pretrained(model_name, layer_index, sae_weights_path)
model_ori = AutoModelForCausalLM.from_pretrained(model_name)
data_sae = torch.load(sae_weights_path)

print("====================>>check: upd weight<<======================")
print("mlp.up_proj.weight:", upd_model.model.layers[19].mlp.up_proj.weight)   
print("mlp.down_proj.weight:", upd_model.model.layers[19].mlp.down_proj.weight) 
print("mlp.gate_proj.weight:", upd_model.model.layers[19].mlp.gate_proj.weight) 
print("mlp.up_proj.weight.shape:", upd_model.model.layers[19].mlp.up_proj.weight.shape) 
print("mlp.down_proj.weight.shape:", upd_model.model.layers[19].mlp.down_proj.weight.shape) 
print("mlp.gate_proj.weight.shape:", upd_model.model.layers[19].mlp.gate_proj.weight.shape) 

print("====================>>check: sae weight<<======================")
print("W_enc.shape:", data_sae['state_dict']['W_enc'].shape)
print("W_dec.shape:", data_sae['state_dict']['W_dec'].shape)
print("W_enc:", data_sae['state_dict']['W_enc'])
print("W_dec:", data_sae['state_dict']['W_dec'])


print("====================>>check: ori weight<<======================")
print("mlp.up_proj.weight:", model_ori.model.layers[19].mlp.up_proj.weight)   
print("mlp.down_proj.weight:", model_ori.model.layers[19].mlp.down_proj.weight) 
print("mlp.gate_proj.weight:", model_ori.model.layers[19].mlp.gate_proj.weight) 
print("mlp.up_proj.weight.shape:", model_ori.model.layers[19].mlp.up_proj.weight.shape) 
print("mlp.down_proj.weight.shape:", model_ori.model.layers[19].mlp.down_proj.weight.shape) 
print("mlp.gate_proj.weight.shape:", model_ori.model.layers[19].mlp.gate_proj.weight.shape) 

print("====================>>check: cache activation from upd weight<<======================")
up_proj_activation_upd = load_model_and_extract_activation(upd_model, input_text)
print("up_proj_activation_upd:", up_proj_activation_upd)
print("up_proj_activation_upd.shape:", up_proj_activation_upd.shape)


print("====================>>check: cache activation from ori weight<<======================")
up_proj_activation_ori = load_model_and_extract_activation(model_ori, input_text)
print("up_proj_activation_ori:", up_proj_activation_ori)
print("up_proj_activation_ori.shape:", up_proj_activation_ori.shape)
