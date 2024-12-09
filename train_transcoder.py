# Transcoder training sample code

"""
This sample script can be used to train a transcoder on a model of your choice.
This code, along with the transcoder training code more generally, was largely
    adapted from an older version of Joseph Bloom's SAE training repo, the latest
    version of which can be found at https://github.com/jbloomAus/SAELens.
Most of the parameters given here are the same as the SAE training parameters
    listed at https://jbloomaus.github.io/SAELens/training_saes/.
Transcoder-specific parameters are marked as such in comments.

"""
import argparse
from dataclasses import asdict

import torch
import os
import sys
import numpy as np
import wandb
from wandb.cli.cli import offline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_training.train_sae_on_language_model import train_sae_on_language_model

def main(args):
    lr = 4e-4  # learning rate
    l1_coeff = 1e-6  # l1 sparsity regularization coefficient 1.4e-4
    expansion_factor = 16

    batch_size = 4096
    per_device_batch_size = None
    total_training_tokens = 81_380_000
    l1_warm_up_steps = 5000

    cfg = LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distibuion)

        # "hook_point" is the TransformerLens HookPoint representing
        #    the input activations to the transcoder that we want to train on.
        # Here, "ln2.hook_normalized" refers to the activations after the
        #    pre-MLP LayerNorm -- that is, the inputs to the MLP.
        # You might alternatively prefer to train on "blocks.8.hook_resid_mid",
        #    which corresponds to the input to the pre-MLP LayerNorm.
        hook_point="blocks.19.ln2.hook_normalized",
        hook_point_layer=19,
        d_in=4096,
        dataset_path="codeparrot/github-code",
        is_dataset_tokenized=False,
        model_name='CodeLlama-7b-Instruct-hf',

        # Transcoder-specific parameters.
        is_transcoder=True,  # We're training a transcoder here.
        # "out_hook_point" is the TransformerLens HookPoint representing
        #    the output activations that the transcoder should reconstruct.
        # In our use case, we're using transcoders to interpret MLP sublayers.
        # This means that our transcoder will take in the input to an MLP and
        #    attempt to spit out the output of the MLP (but in the form of a
        #    sparse linear combination of feature vectors).
        # As such, we want to grab the "hook_mlp_out" activations from our
        #    transformer, which (as the name suggests), represent the
        #    output activations of the original MLP sublayer.
        out_hook_point="blocks.19.hook_mlp_out",
        out_hook_point_layer=19,
        d_out=4096,

        # SAE Parameters
        expansion_factor=expansion_factor,
        b_dec_init_method="mean",

        # Training Parameters
        lr=lr,
        l1_coefficient=l1_coeff,
        lr_scheduler_name="constantwithwarmup",
        train_batch_size=batch_size,
        per_device_batch_size=per_device_batch_size,
        context_size=128,
        lr_warm_up_steps=l1_warm_up_steps,

        # Activation Store Parameters
        n_batches_in_buffer=32,  # Must be large enough so that n_batches_in_buffer * store_batch_size * context_size >= 2 * train_batch_size
        total_training_tokens=total_training_tokens,
        store_batch_size=1,
        data_column="code",
        improve_mixing=True,  # Disabling this mean you can get as low as n_batches_in_buffer * store_batch_size * context_size >= train_batch_size

        # Dead Neurons and Sparsity
        use_ghost_grads=True,
        feature_sampling_method=None,
        feature_sampling_window=1000,
        resample_batches=1028,
        dead_feature_window=5000,
        dead_feature_threshold=1e-8,
        top_k=None,

        # WANDB
        log_to_wandb=True,
        wandb_project="Transcoder_Codellama",
        wandb_entity="pvs-shared",
        wandb_group=None,
        wandb_log_frequency=10,

        # Misc
        use_tqdm=True,
        device="cuda:0",
        seed=42,
        n_checkpoints=0,
        checkpoint_path="/nfs/data/shared/codellama-transcoders",  # change as you please
        dtype=torch.float32,
        model_dtype=torch.float16,
        model_device="cuda:1",
        lazy_device_loading=False,
    )


    print(f"About to start training with lr {lr} and l1 {l1_coeff}")
    print(f"Checkpoint path: {cfg.checkpoint_path}")
    print(cfg)

    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, group=cfg.wandb_group, config=asdict(cfg),
                   mode="offline" if args.offline else "online")

    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sparse_autoencoder, activations_loader = loader.load_session()

    # train SAE
    sparse_autoencoder = train_sae_on_language_model(
        model, sparse_autoencoder, activations_loader,
        n_checkpoints=cfg.n_checkpoints,
        batch_size=cfg.train_batch_size,
        feature_sampling_method=cfg.feature_sampling_method,
        feature_sampling_window=cfg.feature_sampling_window,
        feature_reinit_scale=cfg.feature_reinit_scale,
        dead_feature_threshold=cfg.dead_feature_threshold,
        dead_feature_window=cfg.dead_feature_window,
        use_wandb=cfg.log_to_wandb,
        wandb_log_frequency=cfg.wandb_log_frequency
    )

    # save sae to checkpoints folder
    path = f"{cfg.checkpoint_path}/final_{sparse_autoencoder.get_name()}.pt"
    sparse_autoencoder.save_model(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_device", type=str, default="cuda")

    main(parser.parse_args())