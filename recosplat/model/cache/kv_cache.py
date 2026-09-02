"""Key/value cache management for streaming transformer inference."""

from typing import List, Literal, Tuple

import torch

KV_CACHE_STRATEGY = Literal["accumulate"]


class KVCacheManager:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
        strategy: KV_CACHE_STRATEGY = "accumulate",
        chunk_size: int | List[int] | None = None,
    ):
        """
        Initializes the cache by pre-allocating tensors.

        Args:
            batch_size: The batch size.
            num_heads: The number of attention heads.
            max_seq_len: The maximum sequence length for which to allocate memory.
            head_dim: The dimension of each attention head.
            dtype: The data type of the tensors.
            device: The device to store the tensors on.
            strategy: Dictates the behavior of get().
            chunk_size: Needed for certain kv cache strategies.
        """
        self.max_seq_len = max_seq_len
        self.pos = 0

        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.k_cache = torch.empty(cache_shape, dtype=dtype, device=device)
        self.v_cache = torch.empty(cache_shape, dtype=dtype, device=device)

        self.strategy = strategy
        self.chunk_size = chunk_size

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """
        Updates the cache with new key and value tensors.

        Args:
            new_k: The new key tensor of shape (B, H, S_chunk, D).
            new_v: The new value tensor of shape (B, H, S_chunk, D).
        """
        new_len = new_k.shape[2]
        if self.pos + new_len > self.max_seq_len:
            raise ValueError(
                f"Cannot update cache: sequence length exceeds max_seq_len ({self.max_seq_len})"
            )

        self.k_cache[:, :, self.pos : self.pos + new_len, :] = new_k
        self.v_cache[:, :, self.pos : self.pos + new_len, :] = new_v

        self.pos += new_len

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the filled portion of the cache.

        This is what you'll use for the attention computation.
        """
        if self.strategy == "accumulate":
            return self.k_cache[:, :, : self.pos, :], self.v_cache[:, :, : self.pos, :]
        else:
            raise ValueError

    def clear(self):
        """Resets the cache position for a new sequence."""
        self.pos = 0
