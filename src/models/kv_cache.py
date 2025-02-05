"""
KVCache: Adapted from https://github.com/pytorch/torchtune/blob/main/torchtune/modules/kv_cache.py
"""

from typing import Tuple

import torch
from torch import nn


class KVCache:
    """
    Args:
        batch_size (int): batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_kv_heads (int): number of key/value heads.
        head_dim (int): per-attention head embedding dimension
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        # super().__init__()
        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.cache_shape = cache_shape
        self.k_cache = [None for _ in range(n_layers)]
        self.v_cache = [None for _ in range(n_layers)]
        self.cache_pos = [torch.arange(0, max_seq_len) for _ in range(n_layers)]
        self.layer = 0

    @property
    def size(self) -> int:
        return self.cache_pos[self.layer][0].item()

    def set_layer(self, layer: int):
        self.layer = layer

    def setup_cache(self, k_val, v_val):
        if self.k_cache[self.layer] is not None:
            return

        self.k_cache[self.layer] = torch.zeros(self.cache_shape, dtype=k_val.dtype, device=k_val.device)
        self.v_cache[self.layer] = torch.zeros(self.cache_shape, dtype=v_val.dtype, device=v_val.device)

    def reset(self):
        self.layer = 0
        size = self.size
        for i, pos in enumerate(self.cache_pos):
            pos -= size
            # self.k_cache[i].zero_() # seems unnecessary
            # self.v_cache[i].zero_()

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache with the new ``k_val``, ``v_val`` and return the updated cache.

        Note:
            When updating the KV cache, it is assumed that subsequent updates should update key-value
            positions in consecutive sequence positions.

        Example:
            >>> cache = KVCache(batch_size=2, max_seq_len=16, num_kv_heads=4, head_dim=32, dtype=torch.bfloat16)
            >>> keys, values = torch.ones((2, 4, 8, 32)), torch.ones((2, 4, 8, 32))
            >>> cache.update(keys, values)
            >>> # now positions 0 through 7 are filled
            >>> cache.size
            >>> 8
            >>> keys, values = torch.ones((2, 4, 1, 32)), torch.ones((2, 4, 1, 32))
            >>> cache.update(keys, values)
            >>> # this will fill at position 8
            >>> cache.size
            >>> 9

        Args:
            k_val (torch.Tensor): Current key tensor with shape [B, H, S, D]
            v_val (torch.Tensor): Current value tensor with shape [B, H, S, D]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated key and value cache tensors, respectively.

        Raises:
            AssertionError: if the sequence length of ``k_val`` is longer than the maximum cache sequence length.
            ValueError: if the batch size of the new key (or value) tensor is greater than the batch size
                used during cache setup.
        """
        self.setup_cache(k_val, v_val)

        bsz, _, seq_len, _ = k_val.shape
        l = self.layer
        k_cache, v_cache = self.k_cache[l], self.v_cache[l]
        if bsz > k_cache.shape[0]:
            raise ValueError(
                f"The current cache has been setup with a batch size of {k_cache.shape[0]}"
                f", but found new key tensors with batch size {k_val.shape[0]}!"
            )

        assert (self.cache_pos[l][0] + seq_len) <= k_cache.shape[2]

        k_cache[:bsz, :, self.cache_pos[l][:seq_len]] = k_val
        v_cache[:bsz, :, self.cache_pos[l][:seq_len]] = v_val

        # forward cache_pos seq_len positions along
        # cache_pos starts at (0, 1, 2, 3, 4, 5, ...)
        # an update of seq_len = 5 tokens brings it to
        # (5, 6, 7, 8, 9, ...)
        # this allows us to track the current position in the cache
        # after the last update in a compile-friendly way without any dynamism
        # e.g. relying on an int size tracker, or re-creating cache_pos every time
        self.cache_pos[l] += seq_len

        k_out = k_cache[:bsz, :, : self.size]
        v_out = v_cache[:bsz, :, : self.size]

        return k_out, v_out
