import torch
from transformers import AutoModelForCausalLM, pipeline

from huggingface.transcoder_adapter import TranscoderAdapter

if __name__ == '__main__':
    from_hub = False
    if from_hub:
        model = AutoModelForCausalLM.from_pretrained("LeStoe11/CodeLlama-7b-Instruct-hf_blocks.19.ln2.hook_normalized_65536", torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", torch_dtype=torch.float16)
        adapter = TranscoderAdapter.load("/scratch/share/final_sparse_autoencoder_CodeLlama-7b-Instruct-hf_blocks.19.ln2.hook_normalized_65536.pt")
        device = next(model.base_model.layers[19].mlp.parameters()).device
        # TODO: The model work extremely well even if we replace the wrong layer
        model.base_model.layers[19].mlp = adapter.to(model.dtype).to(device)

    generator = pipeline("text-generation", model=model,
                         device="cuda", tokenizer="codellama/CodeLlama-7b-Instruct-hf")
    print(generator('def remove_non_ascii(s: str) -> str:\n    """ <FILL_ME>\n    return result', max_new_tokens=128)[0]["generated_text"])