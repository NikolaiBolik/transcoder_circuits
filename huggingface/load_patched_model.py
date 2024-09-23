import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class MyCustomModelForCausalLM(AutoModelForCausalLM):

    #todo: move the new args to kwargs to keep the signature compatible with the parent class
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, layer_index, sae_weights_path, *model_args, **kwargs):
        kwargs['ignore_mismatched_sizes'] = True
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        with torch.no_grad():
            print("MyCustomModelForCausalLM:from_pretrained: Patching layer ", layer_index)

            mlp = model.model.layers[layer_index].mlp

            # Todo: get the shape of the weight matrices from the tc model instead of hardcoding
            #tc_dim = 65536
            tc_dim = 16384  # Artur's tc_dim
            # Create and add the new weight matrices to the model
            mlp.up_proj = torch.nn.Linear(in_features=4096, out_features=tc_dim, bias=False)
            mlp.down_proj = torch.nn.Linear(in_features=tc_dim, out_features=4096, bias=False)
            mlp.gate_proj = torch.nn.Linear(in_features=4096, out_features=tc_dim, bias=False)
            mlp.gate_proj.weight.data.fill_(1) 
            
            sae_weights = torch.load(sae_weights_path)

            # todo: remove the hardcoded nums in the comments
            W_enc = sae_weights['state_dict']['W_enc']  
            W_dec = sae_weights['state_dict']['W_dec']  

            mlp.up_proj.weight.data.copy_(W_enc.T)  
            mlp.down_proj.weight.data.copy_(W_dec.T)  

 
        return model


### Debugging helper methods
# Debug method to extract the activation of a specific layer
def load_model_and_extract_activations(model, input_text, layer_index, activation_name="up_proj"):
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


def compare_weights(upd_model, orig_model, data_sae, layer_index):
    def show_mlp_weights(layer_index, model_to_show, model_name):
        print(f"====================>>check: {model_name}<<======================")
        print("mlp.up_proj.weight:", model_to_show.model.layers[layer_index].mlp.up_proj.weight)
        print("mlp.down_proj.weight:", model_to_show.model.layers[layer_index].mlp.down_proj.weight)
        print("mlp.gate_proj.weight:", model_to_show.model.layers[layer_index].mlp.gate_proj.weight)
        print("mlp.up_proj.weight.shape:", model_to_show.model.layers[layer_index].mlp.up_proj.weight.shape)
        print("mlp.down_proj.weight.shape:", model_to_show.model.layers[layer_index].mlp.down_proj.weight.shape)
        print("mlp.gate_proj.weight.shape:", model_to_show.model.layers[layer_index].mlp.gate_proj.weight.shape)

    show_mlp_weights(layer_index, upd_model, "updated model")

    print("====================>>check: sae weight<<======================")
    print("W_enc.shape:", data_sae['state_dict']['W_enc'].shape)
    print("W_dec.shape:", data_sae['state_dict']['W_dec'].shape)
    print("W_enc:", data_sae['state_dict']['W_enc'])
    print("W_dec:", data_sae['state_dict']['W_dec'])

    show_mlp_weights(layer_index, orig_model, "original model")


def compare_activations(upd_model, orig_model, input_text, layer_index):
    print("====================>>check: cache activation from upd weight<<======================")
    up_proj_activation_upd = load_model_and_extract_activations(upd_model, input_text, layer_index)
    print("up_proj_activation_upd:", up_proj_activation_upd)
    print("up_proj_activation_upd.shape:", up_proj_activation_upd.shape)

    print("====================>>check: cache activation from ori weight<<======================")
    up_proj_activation_ori = load_model_and_extract_activations(orig_model, input_text, layer_index)
    print("up_proj_activation_ori:", up_proj_activation_ori)
    print("up_proj_activation_ori.shape:", up_proj_activation_ori.shape)


if __name__ == '__main__':
    model_name = "codellama/CodeLlama-7b-Instruct-hf"
    sae_weights_path_min = "/scratch/core/minxue/transcoder-distribute/codellama-transcoders-layerlayer_index-ef16-codeparrot/d8lcir9q/final_sparse_autoencoder_CodeLlama-7b-Instruct-hf_blocks.layer_index.ln2.hook_normalized_65536.pt"
    sae_weights_path_artur = '/nfs/data/shared/codellama-transcoders/12ok1dny/final_sparse_autoencoder_CodeLlama-7b-Instruct-hf_blocks.19.ln2.hook_normalized_16384.pt'
    layer_index = 19
    input_text = "hello world"

    upd_model = MyCustomModelForCausalLM.from_pretrained(model_name, layer_index, sae_weights_path_artur)
    orig_model = AutoModelForCausalLM.from_pretrained(model_name)
    data_sae = torch.load(sae_weights_path_artur)

    compare_weights(upd_model, orig_model, data_sae, layer_index)
    compare_activations(upd_model, orig_model, input_text, layer_index)
