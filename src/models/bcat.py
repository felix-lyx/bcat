"""
Autoregressive BCAT model.
"""

from logging import getLogger
from functools import partial

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask

from .attention_utils import (
    CustomTransformerEncoder,
    CustomTransformerEncoderLayer,
    CacheCustomTransformerEncoder,
    CacheCustomTransformerEncoderLayer,
    DynamicTanh,
)
from .embedder import get_embedder
from .kv_cache import KVCache


logger = getLogger()


def block_lower_triangular_mask(block_size, block_num, use_float=False):
    """
    Create a block lower triangular boolean mask. (upper right part will be 1s, and represent locations to ignore.)
    """
    matrix_size = block_size * block_num
    lower_tri_mask = torch.tril(torch.ones(matrix_size, matrix_size, dtype=torch.bool))
    block = torch.ones(block_size, block_size, dtype=torch.bool)
    blocks = torch.block_diag(*[block for _ in range(block_num)])
    final_mask = torch.logical_or(lower_tri_mask, blocks)

    if use_float:
        return torch.zeros_like(final_mask, dtype=torch.float32).masked_fill_(~final_mask, float("-inf"))
    else:
        return ~final_mask


def block_causal(b, h, q_idx, kv_idx, block_size):
    return (q_idx // block_size) >= (kv_idx // block_size)


class BCAT(nn.Module):
    """
    Wrapper for the autoregressive BCAT model.
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
            case "dyt":
                norm = DynamicTanh
            case _:
                norm = nn.LayerNorm

        kwargs = {
            "d_model": config.dim_emb,
            "nhead": config.n_head,
            "dim_feedforward": config.dim_ffn,
            "dropout": config.dropout,
            "attn_dropout": config.get("attn_dropout", 0),
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
        mask = block_lower_triangular_mask(self.seq_len_per_step, max_data_len, use_float=True)
        self.register_buffer("mask", mask, persistent=False)

        if self.flex_attn:
            block_size = config.patch_num**2
            seq_len = block_size * (max_data_len - 1)
            self.block_mask = create_block_mask(
                partial(block_causal, block_size=block_size), None, None, seq_len, seq_len
            )
            # seq_len_eval = block_size * 10  # NOTE: hardcoded input length
            # self.block_mask_eval = create_block_mask(
            #     partial(block_causal, block_size=block_size), None, None, seq_len_eval, seq_len_eval
            # )

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

    def fwd(self, data, times, input_len: int, **kwargs):
        """
        Inputs:
            data:          Tensor     (bs, input_len+output_len, x_num, x_num, data_dim)
            times:         Tensor     (bs/1, input_len+output_len, 1)
            input_len:     How many timesteps to use as input, for training this should be 1

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """

        data = data[:, :-1]  # ignore last timestep for autoregressive training (b, t_num-1, x_num, x_num, data_dim)
        times = times[:, :-1]  # (bs/1, t_num-1, 1)

        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data_input (bs, t_num-1, x_num, x_num, data_dim) -> (bs, data_len, dim)
                       data_len = (input_len + output_len - 1) * patch_num * patch_num
        """

        data = self.embedder.encode(data, times)  # (bs, data_len, dim)

        """
        Step 2: Transformer
            data_input:   Tensor     (bs, data_len, dim)
        """
        data_len = data.size(1)
        if self.flex_attn:
            block_mask = self.block_mask
            data_encoded = self.transformer(data, block_mask=block_mask)  # (bs, data_len, dim)
        else:
            mask = self.mask[:data_len, :data_len]
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                data_encoded = self.transformer(data, mask=mask)  # (bs, data_len, dim)

        """
        Step 3: Decode data
        """

        input_seq_len = (input_len - 1) * self.seq_len_per_step
        data_output = data_encoded[:, input_seq_len:]  # (bs, output_len*patch_num*patch_num, dim)

        data_output = self.embedder.decode(data_output)  # (bs, output_len, x_num, x_num, data_dim)
        return data_output

    @torch.compiler.disable()
    def generate(self, data_input, times, input_len: int, data_mask, carry_over_c=-1, **kwargs):
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
        output_len = t_num - input_len
        bs, _, x_num, _, data_dim = data_input.size()

        data_all = torch.zeros(bs, t_num, x_num, x_num, data_dim, dtype=data_input.dtype, device=data_input.device)
        data_all[:, :input_len] = data_input
        cur_len = input_len
        prev_len = 0

        config = self.config
        if config.kv_cache:
            cache = KVCache(
                config.n_layer, data_input.shape[0], self.mask.size(0), config.n_head, config.dim_emb // config.n_head
            )

        for i in range(output_len):
            cur_data_input = data_all[:, :cur_len]  # (bs, cur_len, x_num, x_num, data_dim)

            # (bs, cur_len, x_num, x_num, data_dim) -> (bs, data_len=cur_len*p*p, dim)
            skip_len = prev_len if self.config.kv_cache else 0
            cur_data_input = self.embedder.encode(
                cur_data_input, times[:, :cur_len], skip_len=skip_len
            )  # (bs, data_len, dim)

            mask = None
            if (not self.config.kv_cache) or i == 0:
                data_len = cur_len * self.seq_len_per_step
                mask = self.mask[:data_len, :data_len]

            if self.config.kv_cache:
                cur_data_encoded = self.transformer(cur_data_input, mask, cache=cache)
            else:
                cur_data_encoded = self.transformer(cur_data_input, mask)  # (bs, data_len, dim)

            new_output = cur_data_encoded[:, -self.seq_len_per_step :]  # (bs, patch_num*patch_num, dim)
            new_output = self.embedder.decode(new_output)  # (bs, 1, x_num, x_num, data_dim)

            new_output = new_output * data_mask  # (bs, 1, x_num, x_num, data_dim)

            if carry_over_c >= 0:
                new_output[:, 0, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

            data_all[:, cur_len : cur_len + 1] = new_output
            prev_len = cur_len
            cur_len += 1

        return data_all[:, input_len:]
