"""
Next token prediction variant of BCAT model.
"""

import torch
import torch.nn as nn

from .attention_utils import (
    CustomTransformerEncoder,
    CustomTransformerEncoderLayer,
    CacheCustomTransformerEncoder,
    CacheCustomTransformerEncoderLayer,
)
from .embedder import get_embedder
from logging import getLogger
from functools import partial
from torch.nn.attention.flex_attention import create_block_mask
from .kv_cache import KVCache

logger = getLogger()


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class BCAT_causal(nn.Module):
    """
    Next token prediction variant of BCAT model.
    """

    def __init__(self, config, x_num, max_output_dim, max_data_len=1):
        super().__init__()
        self.config = config
        self.x_num = x_num
        self.max_output_dim = max_output_dim

        self.embedder = get_embedder(config.embedder, x_num, max_output_dim)
        self.flex_attn = config.get("flex_attn", False)

        match config.get("norm", "layer"):
            case "rms":
                norm = nn.RMSNorm
            case _:
                norm = nn.LayerNorm

        kwargs = {
            "d_model": config.dim_emb,
            "nhead": config.n_head,
            "dim_feedforward": config.dim_ffn,
            "dropout": config.dropout,
            "activation": config.get("activation", "gelu"),
            "norm_first": config.norm_first,
            "norm": norm,
            "rotary": config.rotary,
            "qk_norm": config.get("qk_norm", False),
            "flex_attn": self.flex_attn,
        }

        if config.kv_cache:
            self.transformer = CacheCustomTransformerEncoder(
                CacheCustomTransformerEncoderLayer(**kwargs),
                num_layers=config.n_layer,
                norm=norm(config.dim_emb, eps=1e-5) if config.norm_first else None,
                config=config,
            )
        else:
            self.transformer = CustomTransformerEncoder(
                CustomTransformerEncoderLayer(**kwargs),
                num_layers=config.n_layer,
                norm=norm(config.dim_emb, eps=1e-5) if config.norm_first else None,
                config=config,
            )

        self.seq_len_per_step = config.embedder.patch_num**2
        self.max_seq_len = self.seq_len_per_step * max_data_len

        if self.flex_attn:
            block_size = config.patch_num**2
            seq_len = block_size * max_data_len
            self.block_mask = create_block_mask(causal, None, None, seq_len, seq_len)

    def summary(self):
        s = "\n"
        s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        s += f"\tTransformer:    {sum([p.numel() for p in self.transformer.parameters() if p.requires_grad]):,}"
        return s

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data, input_len: int, **kwargs):
        """
        Inputs:
            data:          Tensor     (bs, input_len+output_len, x_num, x_num, data_dim)
            input_len:     How many timesteps to use as input, for training this should be 1

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """

        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data_input (bs, t_num, x_num, x_num, data_dim) -> (bs, data_len, dim)
                       data_len = (input_len + output_len) * patch_num * patch_num
        """
        pos_embeddings = self.embedder.get_pos_embeddings(data.size(1))
        data = self.embedder.encode(data) + pos_embeddings  # (bs, data_len, dim)

        """
        Step 2: Transformer
            data_input:   Tensor     (bs, data_len, dim)
        """
        if self.flex_attn:
            block_mask = self.block_mask
            data_encoded = self.transformer(data, block_mask=block_mask)  # (bs, data_len, dim)
        else:
            data_encoded = self.transformer(data, is_causal=True)  # (bs, data_len, dim)

        """
        Step 3: Decode data
        """

        input_seq_len = input_len * self.seq_len_per_step
        data_output = data_encoded[:, (input_seq_len - 1) : -1]  # (bs, output_len*patch_num*patch_num, dim)

        data_output = self.embedder.decode(data_output)  # (bs, output_len, x_num, x_num, data_dim)
        return data_output

    @torch.compiler.disable()
    def generate(self, data_input, times, input_len: int, data_mask, **kwargs):
        """
        Inputs:
            data_input:    Tensor     (bs, input_len, x_num, x_num, data_dim)
            times:         Tensor     (bs/1, input_len+output_len, 1)
            data_mask:     Tensor     (1, 1, 1, 1, data_dim)
            carry_over_c:  int        Indicate channel that should be carried over,
                                        not masked out or from output (e.g. boundary mask channel)

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """

        t_num = times.size(1)
        input_token_len = input_len * self.seq_len_per_step
        output_len = t_num - input_len
        output_token_len = output_len * self.seq_len_per_step
        bs, _, x_num, _, data_dim = data_input.size()

        pos_emb = self.embedder.get_pos_embeddings(t_num)  # (1, t*p*p, d)

        all_patches = torch.zeros(
            bs,
            t_num * self.embedder.config.patch_num**2,
            self.embedder.patch_dim,
            dtype=data_input.dtype,
            device=data_input.device,
        )
        all_patches[:, :input_token_len] = self.embedder.encode(data_input, proj=False)
        cur_len = input_token_len

        # (1, 1, patch_dim)
        token_mask = self.embedder.encode(
            data_mask.expand(-1, -1, self.embedder.patch_resolution, self.embedder.patch_resolution, -1), proj=False
        )[0, 0]

        config = self.config
        if config.kv_cache:
            cache = KVCache(
                config.n_layer, data_input.shape[0], self.max_seq_len, config.n_head, config.dim_emb // config.n_head
            )

        for i in range(output_token_len):
            # (bs, cur_len*p*p, patch_dim=data_dim*patch_res^2) -> (bs, data_len=cur_len*p*p, d)
            skip_len = cache.size if config.kv_cache else 0
            cur_data_input = self.embedder.pre_proj(all_patches[:, skip_len:cur_len]) + pos_emb[:, skip_len:cur_len]

            if self.config.kv_cache:
                cur_data_encoded = self.transformer(cur_data_input, is_causal=True, cache=cache)
            else:
                cur_data_encoded = self.transformer(cur_data_input, is_causal=True)  # (bs, data_len, d)

            # (bs, 1, d) -> (bs, 1, patch_dim)
            new_output = self.embedder.post_proj(cur_data_encoded[:, -1:])

            new_output = new_output * token_mask  # (bs, 1, patch_dim)
            all_patches[:, cur_len : cur_len + 1] = new_output
            cur_len += 1

        output_patches = all_patches[:, input_token_len:]  # (bs, output_len*p*p, patch_dim)

        return self.embedder.decode(output_patches, proj=False)
