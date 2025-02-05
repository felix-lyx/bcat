"""
This file contains complete transformer encoder/decoder modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_utils import (
    Embedding,
    OperatorDecoderLayer,
    SinusoidalPE,
    LearnablePE,
    CustomTransformerEncoder,
    CustomTransformerEncoderLayer,
    get_block_attn_mask,
    get_embeddings,
)
from logging import getLogger
from functools import partial

logger = getLogger()

"""
Transformer Data modules

"""


class TransformerDataEncoder(nn.Module):
    """
    Encoder Transformer for data
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if config.n_layer == 0:
            self.transformer_encoder = None
        else:
            match config.get("norm", "layer"):
                case "rms":
                    norm = nn.RMSNorm
                case _:
                    norm = nn.LayerNorm

            self.transformer_encoder = CustomTransformerEncoder(
                CustomTransformerEncoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation=config.get("activation", "gelu"),
                    norm_first=config.norm_first,
                    norm=norm,
                    rotary=config.rotary,
                    qk_norm=config.get("qk_norm", False),
                ),
                num_layers=config.n_layer,
                norm=norm(config.dim_emb, eps=1e-5) if config.norm_first else None,
                config=config,
            )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

    def forward(self, x, mask=None, src_key_padding_mask=None, is_causal=False):
        """
        x: Tensor (bs, slen, dim)
        """

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)  # (bs, slen, dim)

        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x, mask, src_key_padding_mask, is_causal)

        return x  # (bs, slen, dim)


class DataOperatorDecoder(nn.Module):
    """
    Operator Decoder for data
    """

    def __init__(self, config, output_len=1, space_len=None):
        super().__init__()

        self.config = config

        self.dim = config.dim_emb

        self.time_embed_type = config.get("time_embed", "continuous")
        if self.time_embed_type == "continuous":
            self.time_proj = nn.Sequential(
                nn.Linear(config.query_dim, self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim),
            )
        else:
            self.time_embed = get_embeddings((1, config.get("max_time_len", 10), 1, self.dim))

        if space_len is None:
            space_len = config.patch_num_output**2

        self.patch_position_embeddings = get_embeddings((1, 1, space_len, self.dim))

        if config.self_attn > 0:
            # self attn + cross attn + ffn

            self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )

            if config.self_attn == 1:
                # self attn is restricted to patches with the same t
                self_attn_mask = get_block_attn_mask(
                    block_size=config.patch_num_output * config.patch_num_output, n_repeat=output_len
                )
                self.register_buffer("self_attn_mask", self_attn_mask)
        else:
            # cross attn + ffn

            match config.get("norm", "layer"):
                case "rms":
                    norm = nn.RMSNorm
                case _:
                    norm = nn.LayerNorm

            self.transformer_decoder = nn.TransformerDecoder(
                OperatorDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=config.norm_first,
                    custom_attn=config.get("custom_attn", 0),
                    norm=norm,
                ),
                num_layers=config.n_layer,
                # norm=norm(config.dim_emb) if config.norm_first else None,
                norm=norm(config.dim_emb) if (config.norm_first and config.final_ln) else None,
            )

    def get_query_emb(self, times):
        """
        Input:
            times:     Tensor (bs/1, output_len, 1)
        Output:
            query_emb: Tensor (bs/1, query_len, dim)
                       query_len = output_len * patch_num * patch_num
        """

        bs, output_len, query_dim = times.size()

        if self.time_embed_type == "continuous":
            times = self.time_proj(times)[:, :, None]  # (bs/1, output_len, 1, dim)
        else:
            times = self.time_embed[:, :output_len]  # (1, input_len, 1, dim)

        return (times + self.patch_position_embeddings).reshape(bs, -1, self.dim)

    def forward(self, src, query_emb, src_key_padding_mask=None, tgt_mask=None):
        """
        src:         Tensor (bs, src_len, dim)
        query_emb:   Tensor (bs, query_len, dim)
        src_key_padding_mask: Optional[Tensor] (bs, src_len)
        tgt_mask:             Optional[Tensor] (query_len, query_len) or (bs*n_head, query_len, query_len)
        """

        if tgt_mask is None and self.config.self_attn == 1:
            tgt_mask = self.self_attn_mask

        x = self.transformer_decoder(query_emb, src, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)

        return x  # (bs, query_len, dim)


"""
Transformer Symbol Modules

"""


class TransformerSymbolEncoder(nn.Module):
    """
    Encoder Transformer for Symbols
    """

    def __init__(self, config, id2word):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if config.n_layer == 0:
            self.transformer_encoder = None
        else:
            match config.get("norm", "layer"):
                case "rms":
                    norm = nn.RMSNorm
                case _:
                    norm = nn.LayerNorm

            self.transformer_encoder = CustomTransformerEncoder(
                CustomTransformerEncoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation=config.get("activation", "gelu"),
                    norm_first=config.norm_first,
                    norm=norm,
                    rotary=config.rotary,
                    qk_norm=config.get("qk_norm", False),
                ),
                num_layers=config.n_layer,
                norm=norm(config.dim_emb) if config.norm_first else None,
                config=config,
            )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

        # dictionary

        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.bos_index = self.word2id["<BOS>"]
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]
        self.n_words = len(self.id2word)

        self.word_embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)

    def forward(self, x, mask=None, src_key_padding_mask=None, is_causal=False):
        """
        x:                    LongTensor (bs, slen)
        mask:                 Optional[Tensor] (bs, slen, slen)
        src_key_padding_mask: Optional[BoolTensor] (bs, slen)         (positions with value True will be ignored)
        """

        x = self.word_embeddings(x)  # (bs, slen, dim)

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)  # (bs, slen, dim)

        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x, mask, src_key_padding_mask, is_causal)

        return x  # (bs, slen, dim)


"""
Transformer Fusion Module

"""


class TransformerFusion(nn.Module):
    """
    Fusion Transformer
    """

    def __init__(self, config, num_types=2):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if config.n_layer == 0:
            self.transformer_encoder = None
        else:
            if config.get("custom_encoder", 0):
                match config.get("norm", "layer"):
                    case "rms":
                        norm = nn.RMSNorm
                    case _:
                        norm = nn.LayerNorm

                self.transformer_encoder = CustomTransformerEncoder(
                    CustomTransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        norm_first=config.norm_first,
                        rotary=config.rotary,
                        norm=norm,
                    ),
                    num_layers=config.n_layer,
                    norm=norm(config.dim_emb) if config.norm_first else None,
                    config=config,
                )
            else:
                self.transformer_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                    ),
                    num_layers=config.n_layer,
                    norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
                )

        if config.type_embeddings:
            self.type_embeddings = Embedding(num_types, self.dim)
        else:
            self.type_embeddings = None

    def forward(self, x0, x1, key_padding_mask0=None, key_padding_mask1=None):
        """
        x0: Tensor (bs, slen0, dim)
        x1: Tensor (bs, slen1, dim)
        key_padding_mask0: Optional[BoolTensor] (bs, slen0)           (True for positions that should be ignored)
        key_padding_mask1: Optional[BoolTensor] (bs, slen1)
        """

        bs = x0.size(0)

        if self.type_embeddings is not None:
            type0 = torch.zeros(1, 1, dtype=torch.long, device=x0.device)
            type1 = torch.ones(1, 1, dtype=torch.long, device=x1.device)
            x0 = x0 + self.type_embeddings(type0).expand_as(x0)
            x1 = x1 + self.type_embeddings(type1).expand_as(x1)

        x = torch.cat([x0, x1], dim=1)  # (bs, slen0+slen1, dim)

        if key_padding_mask0 is None and key_padding_mask1 is None:
            fused_mask = None
        else:
            if key_padding_mask0 is None:
                key_padding_mask0 = torch.zeros(bs, x0.size(1), dtype=torch.bool, device=x0.device)
            if key_padding_mask1 is None:
                key_padding_mask1 = torch.zeros(bs, x1.size(1), dtype=torch.bool, device=x1.device)
            fused_mask = torch.cat([key_padding_mask0, key_padding_mask1], dim=1)  # (bs, slen0+slen1)

        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=fused_mask)

        return x, fused_mask  # (bs, slen0+slen1, dim)
