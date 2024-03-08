from typing import List, Optional

import torch

from vllm._C import cache_ops
from flash_attn import flash_attn_func
from vllm.model_executor.input_metadata import InputMetadata

_PARTITION_SIZE = 512

class FlashAttentionImpl:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 128, 256]

    @staticmethod
    def reshape_and_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            input_metadata.slot_mapping.flatten(),
            input_metadata.kv_cache_dtype,
        )

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        max_num_partitions = (
            (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
            _PARTITION_SIZE)

        return flash_attn_func.flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            None,
            None,
            cache_seqlens=input_metadata.context_lens,
            block_table=input_metadata.block_tables,
            causal=True,
            alibi_slopes=alibi_slopes,
            num_splits=max_num_partitions,
        )

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        alibi_slopes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        max_num_partitions = (
            (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
            _PARTITION_SIZE)

        return flash_attn_func.flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            key,
            value,
            cache_seqlens=input_metadata.context_lens,
            block_table=input_metadata.block_tables,
            causal=True,
            alibi_slopes=alibi_slopes,
            num_splits=max_num_partitions,
        )
