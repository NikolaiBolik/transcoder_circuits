import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface.transcoder_adapter import TranscoderAdapter


def guess_layer_based_on_config(model, cfg):
    """It's a really good guess"""
    return model.base_model.layers[cfg.hook_point_layer]


def convert_to_hf(transcoder_path, hf_model_name):
    # TODO: This works locally, but the upload process seems not to pick up on the changes properly - need to fix
    model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    adapter = TranscoderAdapter.load(transcoder_path).to(model.dtype)
    cfg = torch.load(transcoder_path)['cfg']
    layer = guess_layer_based_on_config(model, cfg)
    layer.mlp = adapter

    return model


def upload(model, tokenizer, upload_path):
    model.push_to_hub(upload_path)
    tokenizer.push_to_hub(upload_path)


def main(args):
    model = convert_to_hf(
        transcoder_path=args.transcoder_path,
        hf_model_name=args.hf_model_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    upload(model, tokenizer, args.upload_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'transcoder_path',
        type=str,
        default='/scratch/share/final_sparse_autoencoder_CodeLlama-7b-Instruct-hf_blocks.19.ln2.hook_normalized_65536.pt'
    )
    parser.add_argument(
        'hf_model_name',
        type=str,
        default='codellama/CodeLlama-7b-Instruct-hf'
    )
    parser.add_argument(
        'upload_path',
        type=str,
        default=None
    )
    parser.add_argument(
        'hf_username',
        type=str,
        default=None
    )

    args = parser.parse_args()
    assert args.upload_path is not None or args.hf_username is not None

    if args.upload_path is None:
        args.upload_path = f"{args.hf_username}/{os.path.basename(args.transcoder_path).replace('.', '_')}"

    main(args)