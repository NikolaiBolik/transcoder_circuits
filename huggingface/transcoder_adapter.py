from pathlib import Path

import einops
import torch
from torch import nn
from transformers import AutoModelForCausalLM, pipeline

from sae_training.config import LanguageModelSAERunnerConfig


class TranscoderAdapter(nn.Module):
    def __init__(self, cfg: LanguageModelSAERunnerConfig):
        super().__init__()

        self.cfg = cfg
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}"
            )
        self.d_sae = cfg.d_sae
        self.dtype = cfg.dtype

        # transcoder stuff
        self.d_out = self.d_in
        if cfg.is_transcoder and cfg.d_out is not None:
            self.d_out = cfg.d_out

        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=self.dtype)
            )
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype)
        )

        self.down_proj = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(self.d_sae, self.d_out, dtype=self.dtype)
            )
        )

        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.down_proj.data /= torch.norm(self.down_proj.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype)
        )

        self.b_dec_out = nn.Parameter(
            torch.zeros(self.d_out, dtype=self.dtype)
        )

    def forward(self, x):
        sae_in = x - self.b_dec

        hidden_pre = einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            ) + self.b_enc

        feature_acts = torch.nn.functional.relu(hidden_pre)

        sae_out = einops.einsum(
                feature_acts,
                self.down_proj,
                "... d_sae, d_sae d_out -> ... d_out",
            ) + self.b_dec_out

        return sae_out

    @classmethod
    def load(cls, path: Path):
        cfg_and_states = torch.load(path, map_location="cpu")
        module = cls(cfg_and_states["cfg"])

        state_dict = cfg_and_states["state_dict"]
        state_dict['down_proj'] = state_dict['W_dec']
        del state_dict['W_dec']

        module.load_state_dict(state_dict)
        return module
