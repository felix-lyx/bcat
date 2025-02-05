"""
Space Time models. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_utils import CustomTransformerEncoder, MultiheadAttention, get_activation, GroupNorm, FFN
from .embedder import get_embedder
from einops import rearrange
from logging import getLogger

from functools import partial

logger = getLogger()


class A:
    pass


class ST_Block(nn.Module):
    """
    A single layer for processing space and time information.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5,
        norm_first=True,
        bias=True,
        norm=nn.LayerNorm,
        time_module_type="attn",
        space_module_type="attn",
        qk_norm=1,
        modes=32,
    ):
        super().__init__()
        self.norm_first = norm_first
        assert norm_first

        self.time_module_type = time_module_type
        match time_module_type:
            case "attn":
                self.time_module = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, qk_norm=qk_norm)
            case _:
                raise ValueError(f"Unsupported time module: {time_module_type}")

        self.space_module_type = space_module_type
        match space_module_type:
            case "attn":
                self.space_module = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, qk_norm=qk_norm)
            case _:
                raise ValueError(f"Unsupported space module: {space_module_type}")

        # norms
        self.norm1 = norm(d_model, eps=layer_norm_eps)
        self.norm2 = norm(d_model, eps=layer_norm_eps)
        self.norm3 = norm(d_model, eps=layer_norm_eps)

        # feedforward
        self.ffn = FFN(d_model, dim_feedforward, activation, dropout)

        # dropout
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, src, is_causal=False, rotary_emb=None, **kwargs):
        """
        src: Tensor (b, t, p, p, d)
        """
        b, _, p, _, _ = src.size()
        x = src

        # if self.norm_first:
        nx = self.norm1(x)
        nx = rearrange(nx, "b t h w c -> (b h w) t c")
        nx = self.dropout1(self.time_module(nx, nx, nx, is_causal=is_causal, rotary_emb=rotary_emb))
        nx = rearrange(nx, "(b h w) t c -> b t h w c", b=b, h=p)
        x = x + nx

        nx = self.norm2(x)
        match self.space_module_type:
            case "attn":
                nx = rearrange(nx, "b t h w c -> (b t) (h w) c", h=p, w=p)
                nx = self.dropout2(self.space_module(nx, nx, nx, rotary_emb=rotary_emb))
                nx = rearrange(nx, "(b t) (h w) c -> b t h w c", b=b, h=p, w=p)

            case "afno":
                nx = rearrange(nx, "b t h w c -> (b t) h w c", h=p, w=p)
                nx = self.dropout2(self.space_module(nx))
                nx = rearrange(nx, "(b t) h w c -> b t h w c", b=b)
        x = x + nx

        x = x + self.ffn(self.norm3(x))
        # else:
        #     x = rearrange(x, "b t h w c -> (b h w) t c")
        #     x = self.norm1(x + self.dropout1(self.time_module(x, x, x, is_causal=is_causal, rotary_emb=rotary_emb)))

        #     match self.space_module_type:
        #         case "attn":
        #             x = rearrange(x, "(b h w) t c -> (b t) (h w) c", h=p, w=p)
        #             x = self.norm2(x + self.dropout2(self.space_module(x, x, x, rotary_emb=rotary_emb)))
        #             x = rearrange(x, "(b t) (h w) c -> b t h w c", b=b, h=p, w=p)

        #         case "afno":
        #             x = rearrange(x, "(b h w) t c -> (b t) h w c", h=p, w=p)
        #             x = self.norm2(x + self.dropout2(self.space_module(x)))
        #             x = rearrange(x, "(b t) h w c -> b t h w c", b=b)

        #     x = self.norm3(x + self.ffn(x))
        return x


class ST_auto(nn.Module):
    """
    Wrapper for the autoregressive Time-then-Space model.
    """

    def __init__(self, config, x_num, max_output_dim, max_data_len=1):
        super().__init__()
        self.config = config
        self.x_num = x_num
        self.max_output_dim = max_output_dim

        self.embedder = get_embedder(config.embedder, x_num, max_output_dim)

        match config.get("norm", "layer"):
            case "rms":
                norm = nn.RMSNorm
            case "group":
                norm = partial(GroupNorm, 8)
            case _:
                norm = nn.LayerNorm

        self.transformer = CustomTransformerEncoder(
            ST_Block(
                d_model=config.dim_emb,
                nhead=config.n_head,
                dim_feedforward=config.dim_ffn,
                dropout=config.dropout,
                activation=config.get("activation", "gelu"),
                norm_first=config.norm_first,
                norm=norm,
                time_module_type=config.time_module,
                space_module_type=config.space_module,
                qk_norm=config.qk_norm,
                modes=config.modes,
            ),
            num_layers=config.n_layer,
            norm=norm(config.dim_emb, eps=1e-5) if config.norm_first else None,
            config=config,
        )

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
            input_len:     How many timesteps to use as input

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """

        data = data[:, :-1]  # ignore last timestep for autoregressive training (b, t_num-1, x_num, x_num, data_dim)
        times = times[:, :-1]  # (bs/1, t_num-1, 1)

        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data (bs, t_num-1, x_num, x_num, data_dim) -> (bs, t_num-1, patch_num, patch_num, dim)
        """

        data = self.embedder.encode(data, times)  # (b, t, p, p, d)

        """
        Step 2: Transformer
            data:   Tensor     (bs, t_num-1, patch_num, patch_num, dim)
        """
        data_encoded = self.transformer(data, is_causal=True)  # (b, t, p, p, d)

        """
        Step 3: Decode data
        """

        data_output = data_encoded[:, (input_len - 1) :]  # (bs, output_len, patch_num, patch_num, dim)

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

        for _ in range(output_len):
            cur_data_input = data_all[:, :cur_len]  # (bs, cur_len, x_num, x_num, data_dim)

            # (bs, cur_len, x_num, x_num, data_dim) -> (bs, cur_len, p, p, dim)
            cur_data_input = self.embedder.encode(cur_data_input, times[:, :cur_len])

            cur_data_encoded = self.transformer(cur_data_input, is_causal=True)  # (bs, cur_len, p, p, dim)

            new_output = cur_data_encoded[:, -1:]  # (bs, 1, p, p, dim)
            new_output = self.embedder.decode(new_output)  # (bs, 1, x_num, x_num, data_dim)

            new_output = new_output * data_mask  # (bs, 1, x_num, x_num, data_dim)

            if carry_over_c >= 0:
                new_output[:, 0, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

            data_all[:, cur_len : cur_len + 1] = new_output
            cur_len += 1

        return data_all[:, input_len:]
