from typing import List, Tuple
import torch


class StaticKVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        dim_kv_head: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        max_length: int = 8192,
    ):
        self.max_length = max_length
        self.num_layers = num_layers

        self.kv_cache = torch.zeros(
            2,
            num_layers,
            batch_size,
            num_kv_heads,
            max_length,
            dim_kv_head,
            device=device,
            dtype=dtype,
        )
        self.current_length = 0

    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def step(self) -> int:
        if self.current_length >= self.max_length:
            raise ValueError("KV cache is full")

        ret = self.current_length
        self.current_length += 1
        return ret

    def fill_caches(self, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.current_length = kv_caches[0][0].size(2)
        self.kv_cache.zero_()
        for i in range(self.num_layers):
            # 检查是否需要重复键值头以匹配缓存大小
            kv_cache_k = kv_caches[i][0]  # [batch_size, num_kv_heads, seq_len, head_dim]
            kv_cache_v = kv_caches[i][1]  # [batch_size, num_kv_heads, seq_len, head_dim]
            
            # 如果缓存头数大于kv头数，需要重复
            if self.kv_cache.size(3) > kv_cache_k.size(1):
                repeat_factor = self.kv_cache.size(3) // kv_cache_k.size(1)
                kv_cache_k = kv_cache_k.repeat_interleave(repeat_factor, dim=1)
                kv_cache_v = kv_cache_v.repeat_interleave(repeat_factor, dim=1)
            
            self.kv_cache[0, i, :, :, : self.current_length, :] = kv_cache_k
            self.kv_cache[1, i, :, :, : self.current_length, :] = kv_cache_v
