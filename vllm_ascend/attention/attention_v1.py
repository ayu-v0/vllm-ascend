#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Literal, NamedTuple

import torch
import torch_npu
import vllm.envs as envs_vllm
import vllm_ascend.envs as envs_ascend
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (  # type: ignore
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.context_parallel.common_cp import AscendMetadataForDecode, AscendMetadataForPrefill
from vllm_ascend.attention.kvcomp_attn.attention_utils import (
    get_kvcomp_decode_params,
    is_enable_hamming_sparse,
    reshape_and_cache_kvcomp,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_cp,
    split_decodes_and_prefills,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.flashcomm2_oshard_manager import flashcomm2_oshard_manager
from vllm_ascend.utils import weak_ref_tensors
from vllm_ascend.worker.kvcomp_utils import KVCompMetaData

# default max value of sliding window size
SWA_INT_MAX = 2147483647
_ATTN_KEYS_BUFFER = None

# Ascend FIA TND currently supports these head dimensions on the vLLM-Ascend
# path. Larger heterogeneous-head models need a prefill fallback to avoid
# unsupported-kernel behavior.
FIA_TND_SUPPORTED_HEAD_SIZES = {64, 128, 192}

GraphParamKind = Literal["paged_attention", "fia"]
logger = init_logger(__name__)
_GEMMA4_MTP_DEBUG = envs_ascend.VLLM_ASCEND_GEMMA4_MTP_DEBUG
_GEMMA4_MTP_ORACLE = envs_ascend.VLLM_ASCEND_GEMMA4_MTP_ORACLE
_VALID_GEMMA4_PREFILL_ATTENTION_IMPLS = {
    "oracle",
    "reference",
    "windowed",
}
_GEMMA4_SPLITFUSE_CAUSAL_MASK_SHAPE = (2048, 2048)
_GEMMA4_PREFILL_ORACLE_ATOL = 2e-2
_GEMMA4_PREFILL_ORACLE_RTOL = 2e-2


def _normalize_gemma4_prefill_attention_impl(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _VALID_GEMMA4_PREFILL_ATTENTION_IMPLS:
        expected = ", ".join(
            sorted(_VALID_GEMMA4_PREFILL_ATTENTION_IMPLS)
        )
        raise ValueError(
            "Unsupported Gemma4 prefill attention impl "
            f"{value!r}; expected one of: {expected}"
        )
    return normalized


def _has_nonempty_mm_prefix_range(mm_prefix_range: object) -> bool:
    if mm_prefix_range is None:
        return False
    if not isinstance(mm_prefix_range, dict):
        return True
    for ranges in mm_prefix_range.values():
        if not isinstance(ranges, (list, tuple)):
            return True
        if ranges:
            return True
    return False


def _debug_shape(value: object) -> tuple[int, ...] | None:
    if torch.is_tensor(value):
        return tuple(value.shape)
    return None


def _debug_preview(value: object, max_items: int = 8) -> object:
    try:
        if torch.is_tensor(value):
            flat = value.detach().flatten()[:max_items].cpu().tolist()
            return flat
        if isinstance(value, (list, tuple)):
            return list(value[:max_items])
        return value
    except Exception as exc:
        return f"<preview failed: {type(exc).__name__}: {exc}>"


def _debug_finite_summary(value: object, max_rows: int = 2) -> object:
    try:
        if not torch.is_tensor(value):
            return None
        rows = value.detach()
        if rows.dim() > 0:
            rows = rows[:max_rows]
        if rows.numel() == 0:
            return {"shape": tuple(rows.shape), "finite": 0, "total": 0}
        finite = torch.isfinite(rows)
        finite_count = int(finite.sum().item())
        summary: dict[str, object] = {
            "shape": tuple(rows.shape),
            "finite": finite_count,
            "total": int(rows.numel()),
            "nan": int(torch.isnan(rows).sum().item()),
            "inf": int(torch.isinf(rows).sum().item()),
        }
        if finite_count:
            finite_rows = rows[finite].float()
            summary["min"] = float(finite_rows.min().item())
            summary["max"] = float(finite_rows.max().item())
        return summary
    except Exception as exc:
        return f"<finite failed: {type(exc).__name__}: {exc}>"


class AttentionGraphParam(NamedTuple):
    """Captured attention graph metadata.

    `kind` records which attention op was captured, and `layer_name` binds the
    captured params back to the real attention layer during graph replay. This
    avoids inferring op type from tuple length or relying on metadata dict order.
    """

    kind: GraphParamKind
    params: tuple
    layer_name: str | None


def _normalize_graph_param(param: AttentionGraphParam, fallback_layer_name: str) -> tuple[GraphParamKind, tuple, str]:
    if not isinstance(param, AttentionGraphParam):
        raise TypeError(f"Expected AttentionGraphParam, got {type(param).__name__}")
    return param.kind, param.params, param.layer_name or fallback_layer_name


def _get_graph_param_kind(param: AttentionGraphParam) -> GraphParamKind:
    kind, _, _ = _normalize_graph_param(param, "")
    return kind


def _uses_sliding_window_attention(vllm_config: VllmConfig) -> bool:
    return getattr(vllm_config.model_config.hf_text_config, "sliding_window", None) is not None


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl

            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPMetadataBuilder

            return AscendAttentionCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class CompactPagedKVMetadata:
    """Forward-scoped physical slots shared by one KV cache group."""

    physical_slots: torch.Tensor
    block_table_snapshot: torch.Tensor
    actual_seq_lengths_kv: list[int]
    seq_lens: tuple[int, ...]
    block_size: int
    cache_block_capacity: int
    source_block_table_data_ptr: int
    source_block_table_shape: tuple[int, ...]
    source_block_table_stride: tuple[int, ...]
    source_block_table_dtype: torch.dtype
    source_block_table_device: torch.device


@dataclass(frozen=True)
class SingleRequestSlidingKVView:
    """A zero-copy tail view of compact physical slots."""

    physical_slots: torch.Tensor
    actual_seq_lengths_kv: list[int]
    window_start: int
    full_kv_tokens: int
    windowed_kv_tokens: int
    kv_tokens_saved: int


def _select_single_request_sliding_kv_view(
    physical_slots: torch.Tensor,
    *,
    seq_len: int,
    query_len: int,
    sliding_window: int,
) -> SingleRequestSlidingKVView:
    if query_len <= 0:
        raise ValueError("query_len must be positive")
    if sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    if seq_len < query_len:
        raise ValueError("seq_len must be >= query_len")
    if physical_slots.numel() != seq_len:
        raise ValueError(
            "single-request compact slot count must equal seq_len: "
            f"slots={physical_slots.numel()} seq_len={seq_len}"
        )

    history_len = seq_len - query_len
    window_start = max(0, history_len - sliding_window)
    windowed_slots = physical_slots[window_start:seq_len]
    windowed_kv_tokens = seq_len - window_start
    return SingleRequestSlidingKVView(
        physical_slots=windowed_slots,
        actual_seq_lengths_kv=[windowed_kv_tokens],
        window_start=window_start,
        full_kv_tokens=seq_len,
        windowed_kv_tokens=windowed_kv_tokens,
        kv_tokens_saved=window_start,
    )


def _build_compact_paged_kv_metadata(
    block_table: torch.Tensor,
    seq_lens: list[int],
    *,
    block_size: int,
    cache_block_capacity: int,
) -> CompactPagedKVMetadata:
    """Build sequence-major physical slots without rectangular KV padding."""

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}")
    if (
        isinstance(cache_block_capacity, bool)
        or not isinstance(cache_block_capacity, int)
        or cache_block_capacity <= 0
    ):
        raise ValueError(
            "cache_block_capacity must be a positive integer, "
            f"got {cache_block_capacity!r}"
        )
    if block_table.ndim != 2:
        raise ValueError(
            "block_table must be 2-D, "
            f"got shape={tuple(block_table.shape)}"
        )
    if block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "block_table must use int32 or int64, "
            f"got dtype={block_table.dtype}"
        )
    if not isinstance(seq_lens, list):
        raise TypeError(
            "seq_lens must be a list of non-negative integers, "
            f"got type={type(seq_lens).__name__}"
        )
    if any(
        isinstance(seq_len, bool) or not isinstance(seq_len, int)
        for seq_len in seq_lens
    ):
        raise TypeError("seq_lens must contain non-negative integers")
    if any(seq_len < 0 for seq_len in seq_lens):
        raise ValueError("seq_lens must contain non-negative integers")

    batch_size = len(seq_lens)
    total_tokens = sum(seq_lens)
    max_seq_len = max(seq_lens, default=0)
    required_block_cols = cdiv(max_seq_len, block_size) if max_seq_len else 0
    if block_table.shape[0] < batch_size:
        raise ValueError(
            "block_table row count is smaller than the sequence batch: "
            f"rows={block_table.shape[0]} batch_size={batch_size}"
        )
    if block_table.shape[1] < required_block_cols:
        raise ValueError(
            "block_table has insufficient logical block columns: "
            f"cols={block_table.shape[1]} required={required_block_cols} "
            f"max_seq_len={max_seq_len} block_size={block_size}"
        )

    actual_seq_lengths_kv = list(itertools.accumulate(seq_lens))
    block_table_snapshot = block_table[
        :batch_size,
        :required_block_cols,
    ].long().clone().contiguous()
    if total_tokens == 0:
        physical_slots = block_table.new_empty((0,), dtype=torch.long)
    else:
        sequence_starts = [0, *itertools.accumulate(seq_lens[:-1])]
        seq_lens_tensor = torch.tensor(
            seq_lens,
            dtype=torch.long,
            device=block_table.device,
        )
        sequence_ids = torch.arange(
            batch_size,
            dtype=torch.long,
            device=block_table.device,
        ).repeat_interleave(
            seq_lens_tensor,
            output_size=total_tokens,
        )
        sequence_starts_tensor = torch.tensor(
            sequence_starts,
            dtype=torch.long,
            device=block_table.device,
        )
        token_offsets = (
            torch.arange(
                total_tokens,
                dtype=torch.long,
                device=block_table.device,
            )
            - sequence_starts_tensor.index_select(0, sequence_ids)
        )
        logical_block_indices = torch.div(
            token_offsets,
            block_size,
            rounding_mode="floor",
        )
        in_block_offsets = torch.remainder(token_offsets, block_size)
        flat_block_table_indices = (
            sequence_ids * required_block_cols + logical_block_indices
        )
        physical_block_ids = block_table_snapshot.reshape(-1).index_select(
            0,
            flat_block_table_indices,
        )
        physical_slots = physical_block_ids * block_size + in_block_offsets

    if physical_slots.numel() != total_tokens:
        raise RuntimeError(
            "Compact paged KV slot count does not match the logical token "
            f"count: slots={physical_slots.numel()} total_tokens={total_tokens}"
        )

    return CompactPagedKVMetadata(
        physical_slots=physical_slots,
        block_table_snapshot=block_table_snapshot,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        seq_lens=tuple(seq_lens),
        block_size=block_size,
        cache_block_capacity=cache_block_capacity,
        source_block_table_data_ptr=block_table.data_ptr(),
        source_block_table_shape=tuple(block_table.shape),
        source_block_table_stride=tuple(block_table.stride()),
        source_block_table_dtype=block_table.dtype,
        source_block_table_device=block_table.device,
    )


@dataclass
class AscendMetadata:
    """
    Per-layer attention metadata for Ascend FlashAttention backend.

    Contains attention masks, token counts, sequence lengths and KV cache
    related properties for attention computation.
    """

    # **************************** Basic Properties ************************** #
    attn_mask: torch.Tensor | None = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0
    num_decodes_flatten: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_list: list[int] = None  # type: ignore
    actual_seq_lengths_q: list[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: int | None = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    # pcp
    prefill: AscendMetadataForPrefill | None = None
    # dcp
    decode_meta: AscendMetadataForDecode | None = None

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None

    kvcomp_metadata: KVCompMetaData | None = None

    # Lazily initialized by Gemma4's single-card large-head fallback. Layers
    # in the same KV cache group share this AscendMetadata instance.
    compact_paged_kv: CompactPagedKVMetadata | None = None


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.

    Handles attention mask generation and metadata preparation for
    Ascend FlashAttention backend.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch, scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )

        block_table = common_attn_metadata.block_table_tensor
        # Prefer _seq_lens_cpu (always available, updated during draft
        # iterations) over seq_lens_cpu (None in async spec decode mode).
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        # this slot_mapping override doesn't work since vllm will override it again. We should fix it vllm.
        # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            seq_lens = common_attn_metadata.seq_lens
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)
        elif self.speculative_config and self.speculative_config.parallel_drafting:
            seq_lens = common_attn_metadata.seq_lens

        attn_state = common_attn_metadata.attn_state

        # Get attn_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(common_attn_metadata.causal, self.model_config)

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
            kvcomp_metadata=common_attn_metadata.kvcomp_metadata,
        )
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        ):
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and ChunkedPrefill state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        self.gemma4_prefill_attention_impl = (
            _normalize_gemma4_prefill_attention_impl(
                envs_ascend.VLLM_ASCEND_GEMMA4_PREFILL_ATTENTION_IMPL
            )
        )
        self._gemma4_prefill_attention_stats = {
            "reference_attention_calls": 0,
            "windowed_attention_calls": 0,
            "windowed_oracle_comparisons": 0,
            "full_kv_tokens": 0,
            "windowed_kv_tokens": 0,
            "kv_tokens_saved": 0,
        }
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32, device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        self.enable_c8_quant = self.vllm_config.quant_config is not None and getattr(
            self.vllm_config.quant_config, "enable_c8_quant", False
        )
        self.sinks = sinks
        self.layerIndex = 0
        self.enable_hamming_sparse = is_enable_hamming_sparse()
        self._layer_name: str | None = None

    def get_gemma4_prefill_attention_stats(self) -> dict[str, int | str]:
        return {
            "gemma4_prefill_attention_impl": (
                self.gemma4_prefill_attention_impl
            ),
            **self._gemma4_prefill_attention_stats,
        }

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        attn_params = graph_params.attn_params.get(num_tokens, [])
        uses_paged_attention_params = len(attn_params) > 0 and all(
            _get_graph_param_kind(param) == "paged_attention" for param in attn_params
        )

        if uses_paged_attention_params or (using_paged_attention(num_tokens, vllm_config) and len(attn_params) == 0):
            # Paged Attention update logic
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    forward_context.attn_metadata,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    _, param, layer_name = _normalize_graph_param(param, key)
                    (
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        num_heads,
                        scale,
                        block_table,
                        seq_lens,
                        output,
                    ) = param
                    metadata_key = layer_name if layer_name in forward_context.attn_metadata else key
                    current_attn_metadata = forward_context.attn_metadata[metadata_key]
                    block_table = current_attn_metadata.block_tables
                    seq_lens = current_attn_metadata.seq_lens

                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                        workspace=workspace,
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        elif _EXTRA_CTX.sinks:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                attn_keys = list(attn_metadata[0].keys())
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            if _EXTRA_CTX.is_draft_model:
                attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        num_kv_heads,
                        num_heads,
                        scale,
                        sliding_window,
                        sinks,
                        attn_output,
                        softmax_lse,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step = attn_count // num_layers
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        attn_count = attn_count + 1
                    else:
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score_v2.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_qlen=actual_seq_lengths_q,
                        actual_seq_kvlen=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_query_heads=num_heads,
                        sparse_mode=4 if sliding_window is not None else 3,
                        pre_tokens=sliding_window if sliding_window is not None else SWA_INT_MAX,
                        next_tokens=0,
                        softmax_scale=scale,
                        learnable_sink=sinks,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        else:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                if _EXTRA_CTX.is_draft_model_prefill:
                    graph_params = get_draft_graph_prefill_params()
                else:
                    graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                attn_keys = list(attn_metadata[0].keys())
            else:
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
                # In some speculative methods (such as DFlash), the order of attn_keys in the Target model
                # will be disrupted instead of increasing by layer index, so need regular expressions to
                # reorder the attn_keys and stor the results in _ATTN_KEYS_BUFFER.
                attn_keys_length = len(graph_params.attn_params[num_tokens])
                global _ATTN_KEYS_BUFFER
                if _ATTN_KEYS_BUFFER is None:
                    import regex as re

                    def extract_layer_index(key: str) -> int:
                        match = re.search(r"(\d+)", key)
                        return int(match.group(1)) if match else 0

                    attn_keys_tmp = attn_keys[:attn_keys_length]
                    attn_keys_tmp.sort(key=extract_layer_index)
                    _ATTN_KEYS_BUFFER = attn_keys_tmp
                attn_keys[:attn_keys_length] = _ATTN_KEYS_BUFFER
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            if _EXTRA_CTX.is_draft_model:
                attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    param_kind, param, layer_name = _normalize_graph_param(param, key)
                    if param_kind == "paged_attention":
                        (
                            query,
                            key_cache,
                            value_cache,
                            num_kv_heads,
                            num_heads,
                            scale,
                            block_table,
                            seq_lens,
                            output,
                        ) = param
                        if _EXTRA_CTX.is_draft_model:
                            draft_step = attn_count // num_layers
                            block_table = attn_metadata[draft_step][key].block_tables
                            seq_lens = attn_metadata[draft_step][key].seq_lens
                            attn_count = attn_count + 1
                        else:
                            metadata_key = layer_name if layer_name in attn_metadata else key
                            block_table = attn_metadata[metadata_key].block_tables
                            seq_lens = attn_metadata[metadata_key].seq_lens
                        workspace = torch_npu._npu_paged_attention_get_workspace(
                            query=query,
                            key_cache=key_cache,
                            value_cache=value_cache,
                            num_kv_heads=num_kv_heads,
                            num_heads=num_heads,
                            scale_value=scale,
                            block_table=block_table,
                            context_lens=seq_lens,
                            out=output,
                        )
                        torch.npu.graph_task_update_begin(update_stream, handle)
                        torch_npu._npu_paged_attention(
                            query=query,
                            key_cache=key_cache,
                            value_cache=value_cache,
                            num_kv_heads=num_kv_heads,
                            num_heads=num_heads,
                            scale_value=scale,
                            block_table=block_table,
                            context_lens=seq_lens,
                            out=output,
                            workspace=workspace,
                        )
                        torch.npu.graph_task_update_end(update_stream)
                        event.record(update_stream)
                        continue

                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        query_start_loc,
                        num_kv_heads,
                        num_heads,
                        scale,
                        attn_output,
                        softmax_lse,
                        sparse_mode,
                        pre_tokens,
                        next_tokens,
                        c8_k_aq_scale,
                        c8_k_aq_offset,
                        c8_v_aq_scale,
                        c8_v_aq_offset,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step = attn_count // num_layers
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        block_tables = attn_metadata[draft_step][key].block_tables
                        attn_count = attn_count + 1
                        if not attn_metadata[draft_step][key].causal:
                            sparse_mode = 0
                    else:
                        metadata_key = layer_name if layer_name in attn_metadata else key
                        seq_lens = attn_metadata[metadata_key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[metadata_key].actual_seq_lengths_q
                        # SWA full-graph replay keeps the captured block table
                        # tensor. Rebinding it from per-step metadata has been
                        # observed to corrupt SWA decode replay on NPU.
                        # Non-SWA models preserve the previous behavior and
                        # refresh block tables from current metadata.
                        if not _uses_sliding_window_attention(vllm_config):
                            block_tables = attn_metadata[metadata_key].block_tables

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    input_layout = "TND"
                    extra_args = {}
                    if c8_k_aq_scale is not None:
                        extra_args = {
                            "key_antiquant_scale": c8_k_aq_scale,
                            "key_antiquant_offset": c8_k_aq_offset,
                            "value_antiquant_scale": c8_v_aq_scale,
                            "value_antiquant_offset": c8_v_aq_offset,
                            "key_antiquant_mode": 0,
                            "value_antiquant_mode": 0,
                        }
                        input_layout = "BNSD"
                        sparse_mode = 0
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout=input_layout,
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=sparse_mode,
                        pre_tokens=pre_tokens,
                        next_tokens=next_tokens,
                        **extra_args,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)

                    event.record(update_stream)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        if flashcomm2_oshard_manager.flashcomm2_oshard_enable():
            flashcomm2_oshard_manager.post_process_after_loading()

    def full_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer=None,
    ) -> torch.Tensor:
        passed_key = key
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        if self.enable_hamming_sparse and attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            reshape_and_cache_kvcomp(attn_metadata.kvcomp_metadata, self.layerIndex, passed_key)
        elif self.enable_hamming_sparse:
            block_table, actual_seq_lengths_kv = get_kvcomp_decode_params(
                self.layerIndex, attn_metadata.kvcomp_metadata, query, passed_key, block_table, actual_seq_lengths_kv
            )

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        # Prepare tensors for attention output
        # TODO: Refactor this to step-level instead of layer-level

        # Get workspace from cache or calculate it if not present.
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        input_layout = "TND"
        attn_mask = attn_metadata.attn_mask
        sparse_mode = 4 if self.sliding_window else 3 if attn_metadata.causal else 0
        pre_tokens = self.sliding_window or SWA_INT_MAX
        next_tokens = 0 if self.sliding_window else SWA_INT_MAX

        extra_args = {}
        if self.enable_c8_quant:
            extra_args = {
                "key_antiquant_scale": layer._c8_k_aq_scale,
                "key_antiquant_offset": layer._c8_k_aq_offset,
                "value_antiquant_scale": layer._c8_v_aq_scale,
                "value_antiquant_offset": layer._c8_v_aq_offset,
                "key_antiquant_mode": 0,
                "value_antiquant_mode": 0,
            }
            # TODO: Convert kvcache to NZ, and change layerout from BNSD to TND.
            input_layout = "BNSD"
            query = query.unsqueeze(2)
            output = output.unsqueeze(2)
            attn_mask = None
            sparse_mode = 0
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout=input_layout,
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                scale=self.scale,
                **extra_args,
            )
            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        # Record the owning layer so graph replay can refresh metadata by the
        # real attention layer instead of relying on dict iteration order.
        layer_name = layer.layer_name if layer is not None else self._layer_name
        attn_params = (
            weak_ref_tensors(query),
            weak_ref_tensors(key),
            weak_ref_tensors(value),
            weak_ref_tensors(block_table),
            weak_ref_tensors(attn_mask) if attn_mask is not None else None,
            block_size,
            actual_seq_lengths_kv,
            actual_seq_lengths_q,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            weak_ref_tensors(output),
            weak_ref_tensors(softmax_lse),
            sparse_mode,
            pre_tokens,
            next_tokens,
        )
        if self.enable_c8_quant:
            attn_params = attn_params + (
                weak_ref_tensors(layer._c8_k_aq_scale),
                weak_ref_tensors(layer._c8_k_aq_offset),
                weak_ref_tensors(layer._c8_v_aq_scale),
                weak_ref_tensors(layer._c8_v_aq_offset),
            )  # type: ignore
        else:
            attn_params = attn_params + (None, None, None, None)  # type: ignore
        graph_params.attn_params[num_tokens].append(AttentionGraphParam("fia", attn_params, layer_name))

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_mask,
            block_table=block_table,
            input_layout=input_layout,
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            workspace=workspace,
            out=[output, softmax_lse],
            **extra_args,
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_fia_v2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        actual_seq_lengths_kv = attn_metadata.seq_lens
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()

        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_qlen=actual_seq_lengths_q,
                actual_seq_kvlen=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                num_query_heads=self.num_heads,
                sparse_mode=4 if self.sliding_window is not None else 3,
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                learnable_sink=self.sinks,
            )

            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (
                weak_ref_tensors(query),
                weak_ref_tensors(key),
                weak_ref_tensors(value),
                weak_ref_tensors(block_table),
                weak_ref_tensors(attn_metadata.attn_mask),
                block_size,
                actual_seq_lengths_kv,
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                self.sliding_window,
                self.sinks,
                weak_ref_tensors(output),
                weak_ref_tensors(softmax_lse),
            )
        )
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_qlen=actual_seq_lengths_q,
            actual_seq_kvlen=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            sparse_mode=4 if self.sliding_window is not None else 3,
            pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
            next_tokens=0,
            softmax_scale=self.scale,
            learnable_sink=self.sinks,
            workspace=workspace,
            out=[output, softmax_lse],
        )
        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ):
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        if _EXTRA_CTX.capturing:
            # Get workspace from cache or calculate it if not present.
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_paged_attention_get_workspace(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output,
                )
                update_graph_params_workspaces(num_tokens, workspace)

            # Handle graph capturing mode
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            graph_params.attn_params[num_tokens].append(
                AttentionGraphParam(
                    "paged_attention",
                    (
                        weak_ref_tensors(query),
                        weak_ref_tensors(self.key_cache),
                        weak_ref_tensors(self.value_cache),
                        self.num_kv_heads,
                        self.num_heads,
                        self.scale,
                        attn_metadata.block_tables,
                        attn_metadata.seq_lens,
                        weak_ref_tensors(output),
                    ),
                    self._layer_name,
                )
            )

            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output,
                workspace=workspace,
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            return output

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata, kv_cache=None):
        # PrefillNoCache doesn't need key_cache, but other modes do
        # Only initialize/require cache for modes that actually use it
        if attn_metadata.attn_state != AscendAttentionState.PrefillNoCache:
            # Initialize cache from kv_cache if not already set (for DecodeOnly mode)
            if self.key_cache is None and kv_cache is not None:
                if (
                    isinstance(kv_cache, torch.Tensor)
                    and kv_cache.dim() > 0
                    and kv_cache.shape[0] == 2
                    or isinstance(kv_cache, (list, tuple))
                    and len(kv_cache) >= 2
                ):
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            if self.key_cache is None:
                raise RuntimeError(
                    f"key_cache is None in _get_fia_params for mode {attn_metadata.attn_state}. kv_cache={kv_cache}"
                )

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill.
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if _EXTRA_CTX.capturing:
            if self.sinks is not None:
                attn_output, num_tokens = self.full_graph_fia_v2(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            else:
                attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
        passed_key = key
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )
        if self.enable_hamming_sparse and attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            reshape_and_cache_kvcomp(attn_metadata.kvcomp_metadata, self.layerIndex, passed_key)
        elif self.enable_hamming_sparse:
            block_table, actual_seq_lengths_kv = get_kvcomp_decode_params(
                self.layerIndex, attn_metadata.kvcomp_metadata, query, passed_key, block_table, actual_seq_lengths_kv
            )
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]
        # Get workspace from cache or calculate it if not present.
        if self.sinks is not None:
            actual_seq_qlen = attn_metadata.actual_seq_lengths_q
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                actual_seq_qlen = torch.tensor([1] * len(attn_metadata.seq_lens_list), dtype=torch.int32).cumsum(dim=0)
            if self.sliding_window is not None:
                sparse_mode = 4
            else:
                sparse_mode = 3
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                key,
                value,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                atten_mask=attn_metadata.attn_mask,
                sparse_mode=sparse_mode,
                softmax_scale=self.scale,
                block_table=block_table,
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_lengths_kv,
                learnable_sink=self.sinks,
            )
        else:
            if not attn_metadata.causal:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=0,
                )
            elif self.sliding_window is not None:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    pre_tokens=self.sliding_window,
                    next_tokens=0,
                    sparse_mode=4,
                )
            else:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=3,
                )

            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        _: torch.Tensor,
    ) -> torch.Tensor:
        # use default sparse_mode 0 in normal scenario, which means no mask works on it
        return torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.scale,
            actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
            actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
        )[0]

    def _forward_large_head_prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if self.gemma4_prefill_attention_impl == "reference":
            use_windowed = False
        else:
            use_windowed = self._can_use_single_request_windowed_prefill(
                attn_metadata
            )
            if not use_windowed and self.sliding_window is not None:
                self._log_windowed_prefill_fallback_once(attn_metadata)
        windowed_view = None
        if use_windowed:
            (
                selected_key,
                selected_value,
                actual_seq_lengths_kv,
                windowed_view,
            ) = self._get_single_request_windowed_prefill_kv(
                attn_metadata,
                num_tokens,
            )
        else:
            selected_key, selected_value, actual_seq_lengths_kv = (
                self._get_large_head_prefill_kv(
                    key,
                    value,
                    attn_metadata,
                    num_tokens,
                )
            )
        if _GEMMA4_MTP_DEBUG and self.kv_sharing_target_layer_name is not None:
            logger.warning(
                "Gemma4 MTP debug: shared_kv_prefill dense_kv "
                "layer=%s target=%s query_shape=%s key_shape=%s "
                "value_shape=%s actual_seq_q=%s actual_seq_kv=%s "
                "seq_lens=%s key_finite=%s value_finite=%s",
                getattr(self, "_layer_name", None),
                self.kv_sharing_target_layer_name,
                _debug_shape(query),
                _debug_shape(selected_key),
                _debug_shape(selected_value),
                attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv,
                _debug_preview(attn_metadata.seq_lens_list),
                _debug_finite_summary(selected_key),
                _debug_finite_summary(selected_value),
            )
        sparse_mode = 4 if self.sliding_window is not None else 3 if attn_metadata.causal else 0
        pre_tokens = self.sliding_window if self.sliding_window is not None else SWA_INT_MAX
        next_tokens = 0 if attn_metadata.causal or self.sliding_window is not None else SWA_INT_MAX
        attn_mask = attn_metadata.attn_mask
        if attn_mask is not None and attn_mask.dtype not in (torch.bool, torch.uint8):
            attn_mask = attn_mask.bool()
        if (
            use_windowed
            and self.gemma4_prefill_attention_impl == "oracle"
        ):
            reference_key, reference_value, reference_seq_lens = (
                self._get_large_head_prefill_kv(
                    key,
                    value,
                    attn_metadata,
                    num_tokens,
                )
            )
            reference_output = self._run_large_head_prefill_attention(
                query=query,
                key=reference_key,
                value=reference_value,
                attn_mask=attn_mask,
                actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=reference_seq_lens,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
            )
            attn_output = self._run_large_head_prefill_attention(
                query=query,
                key=selected_key,
                value=selected_value,
                attn_mask=attn_mask,
                actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
            )
            self._gemma4_prefill_attention_stats[
                "reference_attention_calls"
            ] += 1
            self._gemma4_prefill_attention_stats[
                "windowed_oracle_comparisons"
            ] += 1
            assert windowed_view is not None
            self._assert_windowed_prefill_oracle_equal(
                reference_output,
                attn_output,
                attn_metadata,
                windowed_view,
            )
        else:
            attn_output = self._run_large_head_prefill_attention(
                query=query,
                key=selected_key,
                value=selected_value,
                attn_mask=attn_mask,
                actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
            )
            if (
                not use_windowed
                and self.gemma4_prefill_attention_impl != "reference"
            ):
                self._gemma4_prefill_attention_stats[
                    "reference_attention_calls"
                ] += 1
        if _GEMMA4_MTP_DEBUG and self.kv_sharing_target_layer_name is not None:
            logger.warning(
                "Gemma4 MTP debug: shared_kv_prefill output "
                "layer=%s target=%s attn_output_shape=%s "
                "attn_output_finite=%s",
                getattr(self, "_layer_name", None),
                self.kv_sharing_target_layer_name,
                _debug_shape(attn_output),
                _debug_finite_summary(attn_output),
            )
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def _run_large_head_prefill_attention(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        actual_seq_lengths_q: list[int],
        actual_seq_lengths_kv: list[int],
        sparse_mode: int,
        pre_tokens: int,
        next_tokens: int,
    ) -> torch.Tensor:
        return torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            head_num=self.num_heads,
            input_layout="TND",
            atten_mask=attn_mask,
            scale=self.scale,
            pre_tockens=pre_tokens,
            next_tockens=next_tokens,
            actual_seq_qlen=actual_seq_lengths_q,
            actual_seq_kvlen=actual_seq_lengths_kv,
            sparse_mode=sparse_mode,
        )[0]

    def _can_use_single_request_windowed_prefill(
        self,
        attn_metadata: AscendMetadata,
    ) -> bool:
        attn_mask = attn_metadata.attn_mask
        model_config = getattr(self.vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", hf_config)
        model_type = getattr(text_config, "model_type", None)
        return (
            self.gemma4_prefill_attention_impl != "reference"
            and not _EXTRA_CTX.is_draft_model
            and model_type in {"gemma4", "gemma4_text"}
            and self._can_use_singlecard_compact_paged_kv()
            and not self.enable_c8_quant
            and self.sliding_window is not None
            and attn_metadata.attn_state
            == AscendAttentionState.ChunkedPrefill
            and attn_metadata.num_decodes == 0
            and attn_metadata.num_prefills == 1
            and attn_metadata.seq_lens_list is not None
            and len(attn_metadata.seq_lens_list) == 1
            and attn_metadata.actual_seq_lengths_q is not None
            and len(attn_metadata.actual_seq_lengths_q) == 1
            and attn_metadata.causal
            and attn_metadata.model_runner_type == "generate"
            and attn_mask is not None
            and attn_mask.dtype == torch.int8
            and tuple(attn_mask.shape)
            == _GEMMA4_SPLITFUSE_CAUSAL_MASK_SHAPE
            and not _has_nonempty_mm_prefix_range(
                getattr(attn_metadata, "mm_prefix_range", None)
            )
            and self.key_cache is not None
            and self.value_cache is not None
            and self.key_cache.dtype in (torch.bfloat16, torch.float16)
            and self.value_cache.dtype == self.key_cache.dtype
        )

    def _log_windowed_prefill_fallback_once(
        self,
        attn_metadata: AscendMetadata,
    ) -> None:
        attn_mask = attn_metadata.attn_mask
        model_config = getattr(self.vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", hf_config)
        logger.info_once(
            "Gemma4 windowed prefill attention fallback: "
            "attn_state=%s num_decodes=%s num_prefills=%s "
            "is_draft_model=%s model_type=%s singlecard=%s "
            "enable_c8_quant=%s sliding_window=%s "
            "seq_lens_count=%s actual_q_count=%s causal=%s "
            "model_runner_type=%s attn_mask_dtype=%s "
            "attn_mask_shape=%s mm_prefix_range=%s "
            "key_cache_dtype=%s value_cache_dtype=%s",
            attn_metadata.attn_state,
            attn_metadata.num_decodes,
            attn_metadata.num_prefills,
            _EXTRA_CTX.is_draft_model,
            getattr(text_config, "model_type", None),
            self._can_use_singlecard_compact_paged_kv(),
            self.enable_c8_quant,
            self.sliding_window,
            (
                None
                if attn_metadata.seq_lens_list is None
                else len(attn_metadata.seq_lens_list)
            ),
            (
                None
                if attn_metadata.actual_seq_lengths_q is None
                else len(attn_metadata.actual_seq_lengths_q)
            ),
            attn_metadata.causal,
            attn_metadata.model_runner_type,
            getattr(attn_mask, "dtype", None),
            getattr(attn_mask, "shape", None),
            repr(getattr(attn_metadata, "mm_prefix_range", None)),
            getattr(self.key_cache, "dtype", None),
            getattr(self.value_cache, "dtype", None),
        )

    def _log_windowed_prefill_routing_once(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        shared_kv_prefill: bool,
    ) -> None:
        model_config = getattr(self.vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", hf_config)
        model_type = getattr(text_config, "model_type", None)
        if model_type not in {"gemma4", "gemma4_text"}:
            return

        attn_mask = attn_metadata.attn_mask
        windowed_guard = self._can_use_single_request_windowed_prefill(
            attn_metadata
        )
        logger.warning_once(
            "Gemma4 windowed prefill routing: "
            "layer=%s impl=%s model_type=%s head_size=%s "
            "sliding_window=%s attn_state=%s windowed_guard=%s "
            "num_decodes=%s num_prefills=%s seq_lens_count=%s "
            "actual_q_count=%s causal=%s model_runner_type=%s "
            "capturing=%s "
            "large_head_fallback=%s shared_kv_prefill=%s "
            "kv_sharing_target=%s query_shape=%s key_shape=%s "
            "value_shape=%s key_cache_available=%s "
            "value_cache_available=%s key_cache_dtype=%s "
            "value_cache_dtype=%s attn_mask_dtype=%s "
            "attn_mask_shape=%s mm_prefix_range=%s",
            getattr(self, "_layer_name", None),
            self.gemma4_prefill_attention_impl,
            model_type,
            self.head_size,
            self.sliding_window,
            getattr(
                attn_metadata.attn_state,
                "name",
                attn_metadata.attn_state,
            ),
            windowed_guard,
            attn_metadata.num_decodes,
            attn_metadata.num_prefills,
            (
                None
                if attn_metadata.seq_lens_list is None
                else len(attn_metadata.seq_lens_list)
            ),
            (
                None
                if attn_metadata.actual_seq_lengths_q is None
                else len(attn_metadata.actual_seq_lengths_q)
            ),
            attn_metadata.causal,
            attn_metadata.model_runner_type,
            _EXTRA_CTX.capturing,
            self._should_use_large_head_attention_fallback(),
            shared_kv_prefill,
            self.kv_sharing_target_layer_name,
            tuple(query.shape),
            None if key is None else tuple(key.shape),
            None if value is None else tuple(value.shape),
            self.key_cache is not None,
            self.value_cache is not None,
            getattr(self.key_cache, "dtype", None),
            getattr(self.value_cache, "dtype", None),
            getattr(attn_mask, "dtype", None),
            None if attn_mask is None else tuple(attn_mask.shape),
            repr(getattr(attn_metadata, "mm_prefix_range", None)),
        )

    def _get_single_request_windowed_prefill_kv(
        self,
        attn_metadata: AscendMetadata,
        num_tokens: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int],
        SingleRequestSlidingKVView,
    ]:
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError(
                "Windowed prefill requires initialized Key/Value cache"
            )
        self._validate_compact_paged_kv_cache(
            self.key_cache,
            self.value_cache,
            attn_metadata,
        )
        compact = self._get_or_build_compact_paged_kv_metadata(
            attn_metadata,
            self.key_cache,
        )
        seq_len = attn_metadata.seq_lens_list[0]
        query_len = attn_metadata.actual_seq_lengths_q[0]
        if query_len != num_tokens:
            raise ValueError(
                "single-request query length must equal num_tokens: "
                f"query_len={query_len} num_tokens={num_tokens}"
            )
        assert self.sliding_window is not None
        view = _select_single_request_sliding_kv_view(
            compact.physical_slots,
            seq_len=seq_len,
            query_len=query_len,
            sliding_window=self.sliding_window,
        )
        flat_key_cache = self.key_cache.view(
            -1,
            self.num_kv_heads,
            self.head_size,
        )
        flat_value_cache = self.value_cache.view(
            -1,
            self.num_kv_heads,
            self.head_size,
        )
        dense_key = flat_key_cache.index_select(0, view.physical_slots)
        dense_value = flat_value_cache.index_select(
            0,
            view.physical_slots,
        )
        stats = self._gemma4_prefill_attention_stats
        stats["windowed_attention_calls"] += 1
        stats["full_kv_tokens"] += view.full_kv_tokens
        stats["windowed_kv_tokens"] += view.windowed_kv_tokens
        stats["kv_tokens_saved"] += view.kv_tokens_saved
        if stats["windowed_attention_calls"] == 1:
            log_enabled = (
                logger.warning
                if self.gemma4_prefill_attention_impl == "oracle"
                else logger.info
            )
            log_enabled(
                "Gemma4 windowed prefill attention enabled: "
                "layer=%s seq_len=%d query_len=%d sliding_window=%d "
                "full_kv_tokens=%d windowed_kv_tokens=%d "
                "kv_tokens_saved=%d impl=%s",
                self._layer_name,
                seq_len,
                query_len,
                self.sliding_window,
                view.full_kv_tokens,
                view.windowed_kv_tokens,
                view.kv_tokens_saved,
                self.gemma4_prefill_attention_impl,
            )
        return (
            dense_key,
            dense_value,
            view.actual_seq_lengths_kv,
            view,
        )

    def _assert_windowed_prefill_oracle_equal(
        self,
        reference: torch.Tensor,
        windowed: torch.Tensor,
        attn_metadata: AscendMetadata,
        view: SingleRequestSlidingKVView,
    ) -> None:
        context = (
            f"layer={self._layer_name} "
            f"sliding_window={self.sliding_window} "
            f"seq_len={attn_metadata.seq_lens_list[0]} "
            f"query_len={attn_metadata.actual_seq_lengths_q[0]} "
            f"window_start={view.window_start} "
            f"full_kv_tokens={view.full_kv_tokens} "
            f"windowed_kv_tokens={view.windowed_kv_tokens}"
        )
        if (
            reference.shape != windowed.shape
            or reference.dtype != windowed.dtype
            or reference.device != windowed.device
        ):
            raise RuntimeError(
                "Gemma4 windowed prefill oracle metadata mismatch: "
                f"{context} reference_shape={tuple(reference.shape)} "
                f"windowed_shape={tuple(windowed.shape)} "
                f"reference_dtype={reference.dtype} "
                f"windowed_dtype={windowed.dtype} "
                f"reference_device={reference.device} "
                f"windowed_device={windowed.device}"
            )
        finite = torch.isfinite(reference).all() & torch.isfinite(
            windowed
        ).all()
        close = torch.isclose(
            reference,
            windowed,
            atol=_GEMMA4_PREFILL_ORACLE_ATOL,
            rtol=_GEMMA4_PREFILL_ORACLE_RTOL,
            equal_nan=False,
        ).all()
        if bool((finite & close).item()):
            logger.warning(
                "Gemma4 windowed prefill oracle passed: %s",
                context,
            )
            return

        abs_error = torch.abs(reference - windowed)
        relative_error = abs_error / torch.clamp(
            torch.abs(reference),
            min=1e-12,
        )
        raise RuntimeError(
            "Gemma4 windowed prefill oracle mismatch: "
            f"{context} max_abs_error={abs_error.max().item()} "
            f"max_rel_error={relative_error.max().item()}"
        )

    def _get_large_head_prefill_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            or self.key_cache is None
            or self.value_cache is None
        ):
            return key[:num_tokens], value[:num_tokens], attn_metadata.actual_seq_lengths_q

        seq_lens = attn_metadata.seq_lens_list
        if not seq_lens:
            return key[:num_tokens], value[:num_tokens], attn_metadata.actual_seq_lengths_q

        # Chunked prefill can have historical KV already resident in the paged
        # cache. The large-head fallback uses dense TND attention, so gather the
        # paged KV blocks into sequence-major dense tensors first.
        return self._gather_paged_kv_to_dense(
            self.key_cache,
            self.value_cache,
            attn_metadata,
        )

    def _gather_paged_kv_to_dense(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if self._can_use_singlecard_compact_paged_kv():
            return self._gather_paged_kv_to_dense_compact(
                key_cache,
                value_cache,
                attn_metadata,
            )
        return self._gather_paged_kv_to_dense_legacy(
            key_cache,
            value_cache,
            attn_metadata,
        )

    def _can_use_singlecard_compact_paged_kv(self) -> bool:
        parallel_config = self.vllm_config.parallel_config
        return (
            parallel_config.tensor_parallel_size == 1
            and parallel_config.pipeline_parallel_size == 1
            and parallel_config.data_parallel_size == 1
            and parallel_config.prefill_context_parallel_size == 1
            and parallel_config.decode_context_parallel_size == 1
        )

    def _validate_compact_paged_kv_cache(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
    ) -> None:
        layer_type = "sliding" if self.sliding_window is not None else "full"
        context = (
            f"layer={self._layer_name} layer_type={layer_type} "
            f"target={self.kv_sharing_target_layer_name}"
        )
        if self.enable_c8_quant:
            raise TypeError(
                "Compact paged KV gather does not support C8/INT8 cache: "
                f"{context}"
            )
        if key_cache.ndim != 4 or value_cache.ndim != 4:
            raise ValueError(
                "Compact paged KV cache must be 4-D: "
                f"{context} key_shape={tuple(key_cache.shape)} "
                f"value_shape={tuple(value_cache.shape)}"
            )
        if key_cache.shape != value_cache.shape:
            raise ValueError(
                "Compact paged Key/Value cache shapes differ: "
                f"{context} key_shape={tuple(key_cache.shape)} "
                f"value_shape={tuple(value_cache.shape)}"
            )
        if key_cache.shape[0] <= 0 or key_cache.shape[1] <= 0:
            raise ValueError(
                "Compact paged KV cache capacity and block size must be "
                f"positive: {context} cache_shape={tuple(key_cache.shape)}"
            )
        if key_cache.shape[2:] != (self.num_kv_heads, self.head_size):
            raise ValueError(
                "Compact paged KV cache layout does not match the attention "
                f"layer: {context} cache_shape={tuple(key_cache.shape)} "
                f"expected_heads={self.num_kv_heads} "
                f"expected_head_size={self.head_size}"
            )
        if key_cache.device != value_cache.device:
            raise ValueError(
                "Compact paged Key/Value cache devices differ: "
                f"{context} key_device={key_cache.device} "
                f"value_device={value_cache.device}"
            )
        if key_cache.dtype != value_cache.dtype:
            raise TypeError(
                "Compact paged Key/Value cache dtypes differ: "
                f"{context} key_dtype={key_cache.dtype} "
                f"value_dtype={value_cache.dtype}"
            )
        if key_cache.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                "Compact paged KV gather supports only BF16/FP16 cache: "
                f"{context} cache_dtype={key_cache.dtype}"
            )
        if not key_cache.is_contiguous() or not value_cache.is_contiguous():
            raise ValueError(
                "Compact paged KV cache must be contiguous: "
                f"{context} key_contiguous={key_cache.is_contiguous()} "
                f"value_contiguous={value_cache.is_contiguous()}"
            )
        block_table = attn_metadata.block_tables
        if not torch.is_tensor(block_table):
            raise TypeError(
                "Compact paged KV metadata requires a block table tensor: "
                f"{context} block_table_type={type(block_table).__name__}"
            )
        if block_table.device != key_cache.device:
            raise ValueError(
                "Compact paged block table and KV cache devices differ: "
                f"{context} block_table_device={block_table.device} "
                f"cache_device={key_cache.device}"
            )
        if attn_metadata.seq_lens_list is None:
            raise ValueError(
                "Compact paged KV metadata requires seq_lens_list: "
                f"{context}"
            )

    @staticmethod
    def _compact_paged_kv_signature(
        block_table: torch.Tensor,
        seq_lens: list[int],
        key_cache: torch.Tensor,
    ) -> tuple[object, ...]:
        return (
            tuple(seq_lens),
            key_cache.shape[1],
            key_cache.shape[0],
            block_table.data_ptr(),
            tuple(block_table.shape),
            tuple(block_table.stride()),
            block_table.dtype,
            block_table.device,
        )

    @staticmethod
    def _cached_compact_paged_kv_signature(
        compact: CompactPagedKVMetadata,
    ) -> tuple[object, ...]:
        return (
            compact.seq_lens,
            compact.block_size,
            compact.cache_block_capacity,
            compact.source_block_table_data_ptr,
            compact.source_block_table_shape,
            compact.source_block_table_stride,
            compact.source_block_table_dtype,
            compact.source_block_table_device,
        )

    def _get_or_build_compact_paged_kv_metadata(
        self,
        attn_metadata: AscendMetadata,
        key_cache: torch.Tensor,
    ) -> CompactPagedKVMetadata:
        block_table = attn_metadata.block_tables
        seq_lens = attn_metadata.seq_lens_list
        signature = self._compact_paged_kv_signature(
            block_table,
            seq_lens,
            key_cache,
        )
        compact = attn_metadata.compact_paged_kv
        if compact is None:
            compact = _build_compact_paged_kv_metadata(
                block_table,
                seq_lens,
                block_size=key_cache.shape[1],
                cache_block_capacity=key_cache.shape[0],
            )
            attn_metadata.compact_paged_kv = compact
            return compact

        cached_signature = self._cached_compact_paged_kv_signature(compact)
        if signature != cached_signature:
            layer_type = "sliding" if self.sliding_window is not None else "full"
            raise RuntimeError(
                "Stale compact paged KV metadata cannot be reused: "
                f"layer={self._layer_name} layer_type={layer_type} "
                f"target={self.kv_sharing_target_layer_name} "
                f"cached_signature={cached_signature} "
                f"current_signature={signature}"
            )
        return compact

    def _log_paged_kv_gather_bounds(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_block_table: torch.Tensor,
        selected_block_table: torch.Tensor,
        selected_block_ids: torch.Tensor,
        seq_lens: list[int],
    ) -> None:
        if not _GEMMA4_MTP_DEBUG or selected_block_ids.numel() == 0:
            return
        selected_block_ids_cpu = selected_block_ids.detach().cpu()
        flat_block_id_min = int(selected_block_ids_cpu.min().item())
        flat_block_id_max = int(selected_block_ids_cpu.max().item())
        cache_block_capacity = key_cache.shape[0]
        logger.warning(
            "Gemma4 MTP debug: paged_kv_gather "
            "layer=%s target=%s key_cache_shape=%s value_cache_shape=%s "
            "block_table_shape=%s selected_block_table_shape=%s "
            "block_size=%s num_blocks=%s seq_lens=%s "
            "flat_block_id_min=%s flat_block_id_max=%s "
            "cache_block_capacity=%s",
            self._layer_name,
            self.kv_sharing_target_layer_name,
            _debug_shape(key_cache),
            _debug_shape(value_cache),
            _debug_shape(input_block_table),
            _debug_shape(selected_block_table),
            key_cache.shape[1],
            selected_block_table.shape[1],
            _debug_preview(seq_lens),
            flat_block_id_min,
            flat_block_id_max,
            cache_block_capacity,
        )
        if flat_block_id_min < 0 or flat_block_id_max >= cache_block_capacity:
            raise RuntimeError(
                "Gemma4 MTP paged KV block id is out of range: "
                f"layer={self._layer_name} "
                f"target={self.kv_sharing_target_layer_name} "
                f"min={flat_block_id_min} max={flat_block_id_max} "
                f"cache_block_capacity={cache_block_capacity} "
                f"key_cache_shape={tuple(key_cache.shape)} "
                f"block_table_shape={tuple(input_block_table.shape)}"
            )

    def _gather_paged_kv_snapshot_to_dense_reference(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table_snapshot: torch.Tensor,
        seq_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = key_cache.shape[1]
        seq_lens_tensor = torch.tensor(
            seq_lens,
            dtype=torch.long,
            device=key_cache.device,
        )
        num_blocks = block_table_snapshot.shape[1]
        max_tokens_padded = num_blocks * block_size
        dense_shape = (
            len(seq_lens),
            max_tokens_padded,
            self.num_kv_heads,
            self.head_size,
        )
        flat_block_ids = block_table_snapshot.reshape(-1)
        gathered_key = key_cache.index_select(0, flat_block_ids).reshape(
            dense_shape
        )
        gathered_value = value_cache.index_select(0, flat_block_ids).reshape(
            dense_shape
        )
        positions = torch.arange(
            max_tokens_padded,
            dtype=torch.long,
            device=key_cache.device,
        )
        valid_mask = positions.unsqueeze(0) < seq_lens_tensor.unsqueeze(1)
        return (
            gathered_key[valid_mask].contiguous(),
            gathered_value[valid_mask].contiguous(),
        )

    def _assert_compact_kv_oracle_equal(
        self,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
        old_key: torch.Tensor,
        old_value: torch.Tensor,
        compact: CompactPagedKVMetadata,
        attn_metadata: AscendMetadata,
    ) -> None:
        layer_type = "sliding" if self.sliding_window is not None else "full"
        context = (
            f"layer={self._layer_name} layer_type={layer_type} "
            f"target={self.kv_sharing_target_layer_name} "
            f"seq_lens={attn_metadata.seq_lens_list} "
            f"block_table_snapshot_shape={tuple(compact.block_table_snapshot.shape)} "
            f"physical_slots_shape={tuple(compact.physical_slots.shape)}"
        )
        for name, new, old in (
            ("key", new_key, old_key),
            ("value", new_value, old_value),
        ):
            if new.shape != old.shape or new.dtype != old.dtype or new.device != old.device:
                raise RuntimeError(
                    "Gemma4 compact paged KV oracle metadata mismatch: "
                    f"tensor={name} {context} new_shape={tuple(new.shape)} "
                    f"old_shape={tuple(old.shape)} new_dtype={new.dtype} "
                    f"old_dtype={old.dtype} new_device={new.device} "
                    f"old_device={old.device}"
                )
            if torch.equal(new, old):
                continue
            mismatch = torch.ne(new, old).reshape(-1)
            first_mismatch_flat_index = int(
                torch.nonzero(mismatch, as_tuple=False)[0].item()
            )
            new_scalar = new.reshape(-1)[first_mismatch_flat_index].item()
            old_scalar = old.reshape(-1)[first_mismatch_flat_index].item()
            raise RuntimeError(
                "Gemma4 compact paged KV oracle mismatch: "
                f"tensor={name} {context} "
                f"first_mismatch_flat_index={first_mismatch_flat_index} "
                f"old_value={old_scalar} new_value={new_scalar}"
            )
        logger.warning(
            "Gemma4 compact paged KV oracle passed: %s",
            context,
        )

    def _gather_paged_kv_to_dense_compact(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        self._validate_compact_paged_kv_cache(
            key_cache,
            value_cache,
            attn_metadata,
        )
        compact = self._get_or_build_compact_paged_kv_metadata(
            attn_metadata,
            key_cache,
        )
        if _GEMMA4_MTP_DEBUG and compact.physical_slots.numel():
            selected_block_ids = torch.div(
                compact.physical_slots,
                compact.block_size,
                rounding_mode="floor",
            )
            self._log_paged_kv_gather_bounds(
                key_cache,
                value_cache,
                attn_metadata.block_tables,
                compact.block_table_snapshot,
                selected_block_ids,
                attn_metadata.seq_lens_list,
            )

        flat_key_cache = key_cache.view(
            -1,
            self.num_kv_heads,
            self.head_size,
        )
        flat_value_cache = value_cache.view(
            -1,
            self.num_kv_heads,
            self.head_size,
        )
        dense_key = flat_key_cache.index_select(0, compact.physical_slots)
        dense_value = flat_value_cache.index_select(0, compact.physical_slots)

        if _GEMMA4_MTP_ORACLE:
            old_key, old_value = (
                self._gather_paged_kv_snapshot_to_dense_reference(
                    key_cache,
                    value_cache,
                    compact.block_table_snapshot,
                    attn_metadata.seq_lens_list,
                )
            )
            self._assert_compact_kv_oracle_equal(
                dense_key,
                dense_value,
                old_key,
                old_value,
                compact,
                attn_metadata,
            )

        return dense_key, dense_value, compact.actual_seq_lengths_kv

    def _gather_paged_kv_to_dense_legacy(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        block_table = attn_metadata.block_tables
        seq_lens = attn_metadata.seq_lens_list
        block_size = key_cache.shape[1]
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.long, device=key_cache.device)
        max_seq_len = int(seq_lens_tensor.max().item())
        num_blocks = cdiv(max_seq_len, block_size)
        # The runner reuses the source block-table buffer on subsequent
        # scheduler steps. Keep an NPU-owned snapshot for the asynchronous
        # gather so its indices cannot be mutated after this forward is queued.
        block_table_snapshot = block_table[
            : len(seq_lens),
            :num_blocks,
        ].long().clone().contiguous()
        flat_block_ids = block_table_snapshot.reshape(-1)
        self._log_paged_kv_gather_bounds(
            key_cache,
            value_cache,
            block_table,
            block_table_snapshot,
            flat_block_ids,
            seq_lens,
        )
        dense_key, dense_value = self._gather_paged_kv_snapshot_to_dense_reference(
            key_cache,
            value_cache,
            block_table_snapshot,
            seq_lens,
        )
        return dense_key, dense_value, list(itertools.accumulate(seq_lens))

    def _should_use_large_head_attention_fallback(self) -> bool:
        unsupported_fia_tnd_head_size = self.head_size not in FIA_TND_SUPPORTED_HEAD_SIZES
        return self.sinks is None and unsupported_fia_tnd_head_size

    def _get_current_token_shared_kv(
        self,
        attn_metadata: AscendMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Gather current-token KV from the producer layer's shared cache."""

        if self.key_cache is None or self.value_cache is None:
            return None, None
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if attn_metadata.slot_mapping is None or attn_metadata.slot_mapping.numel() < num_tokens:
            return None, None
        slots = attn_metadata.slot_mapping[:num_tokens].long()
        key = self.key_cache.reshape(-1, self.num_kv_heads, self.head_size).index_select(0, slots)
        value = self.value_cache.reshape(-1, self.num_kv_heads, self.head_size).index_select(0, slots)
        return key, value

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY):
            return

        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            slot_mapping=slot_mapping,
        )

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            if self.kv_sharing_target_layer_name is not None:
                # KV-sharing target layers, used by Gemma4 local/global layer
                # pairs, consume the producer layer's cache. Re-caching here
                # would overwrite the shared KV slots before attention reads it.
                if self.is_kv_producer:
                    attn_metadata.reshape_cache_event.record()
                return query, key, value, output
            slots = attn_metadata.slot_mapping
            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
            DeviceOperator.reshape_and_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                # quick fix to make sure slots is int32 for cross attention case.
                # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots.to(torch.int32),
            )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
        return query, key, value, output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        num_tokens = query.shape[0]
        shared_kv_prefill = (
            self.kv_sharing_target_layer_name is not None
            and key is not None
            and value is not None
            and query.shape[0] == key.shape[0]
            and attn_metadata.attn_state in (AscendAttentionState.PrefillNoCache, AscendAttentionState.ChunkedPrefill)
        )
        if (
            self.gemma4_prefill_attention_impl != "reference"
            and self.sliding_window is not None
            and attn_metadata.attn_state
            in (
                AscendAttentionState.PrefillNoCache,
                AscendAttentionState.ChunkedPrefill,
            )
        ):
            self._log_windowed_prefill_routing_once(
                query,
                key,
                value,
                attn_metadata,
                shared_kv_prefill,
            )
        if shared_kv_prefill:
            shared_cache_available = (
                self.key_cache is not None and self.value_cache is not None
            )
            if _GEMMA4_MTP_DEBUG:
                logger.warning(
                    "Gemma4 MTP debug: shared_kv_prefill enter "
                    "layer=%s target=%s attn_state=%s query_shape=%s "
                    "key_shape=%s value_shape=%s shared_cache_available=%s "
                    "slot_mapping_shape=%s slot_mapping_preview=%s "
                    "actual_seq_q=%s seq_lens=%s block_tables_shape=%s",
                    getattr(self, "_layer_name", None),
                    self.kv_sharing_target_layer_name,
                    getattr(attn_metadata.attn_state, "name", attn_metadata.attn_state),
                    _debug_shape(query),
                    _debug_shape(key),
                    _debug_shape(value),
                    shared_cache_available,
                    _debug_shape(attn_metadata.slot_mapping),
                    _debug_preview(attn_metadata.slot_mapping),
                    attn_metadata.actual_seq_lengths_q,
                    _debug_preview(attn_metadata.seq_lens_list),
                    _debug_shape(attn_metadata.block_tables),
                )
            shared_key, shared_value = self._get_current_token_shared_kv(attn_metadata)
            if _GEMMA4_MTP_DEBUG:
                logger.warning(
                    "Gemma4 MTP debug: shared_kv_prefill current_token_kv "
                    "layer=%s target=%s current_key_shape=%s "
                    "current_value_shape=%s current_key_finite=%s "
                    "current_value_finite=%s",
                    getattr(self, "_layer_name", None),
                    self.kv_sharing_target_layer_name,
                    _debug_shape(shared_key),
                    _debug_shape(shared_value),
                    _debug_finite_summary(shared_key),
                    _debug_finite_summary(shared_value),
                )
            if (
                attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill
                and shared_cache_available
            ):
                return self._forward_large_head_prefill_attention(
                    query,
                    shared_key if shared_key is not None else key,
                    shared_value if shared_value is not None else value,
                    attn_metadata,
                    output,
                )
            if shared_key is not None and shared_value is not None:
                return self._forward_large_head_prefill_attention(
                    query,
                    shared_key,
                    shared_value,
                    attn_metadata,
                    output,
                )

        if (
            self.sliding_window is None
            and (
                (
                    attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                    and (
                        using_paged_attention(num_tokens, self.vllm_config)
                        or self._should_use_large_head_attention_fallback()
                    )
                )
                or (
                    attn_metadata.attn_state == AscendAttentionState.SpecDecoding
                    and self._should_use_large_head_attention_fallback()
                )
            )
        ):
            output = self.forward_paged_attention(query, attn_metadata, output)
        elif (
            not _EXTRA_CTX.capturing
            and self._should_use_large_head_attention_fallback()
            and self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
            and query.shape[0] == key.shape[0]
            and attn_metadata.attn_state in (AscendAttentionState.PrefillNoCache, AscendAttentionState.ChunkedPrefill)
        ):
            output = self._forward_large_head_prefill_attention(query, key, value, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if self.enable_hamming_sparse:
            self.layerIndex = int(layer.layer_name.split(".")[2])
        self._layer_name = layer.layer_name

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        # Initialize key_cache and value_cache from kv_cache if not already set.
        # This is needed for DecodeOnly mode where key/value are None but we still
        # need access to the cache for attention computation.
        if self.key_cache is None and kv_cache is not None:
            if (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.dim() > 0
                and kv_cache.shape[0] == 2
                or isinstance(kv_cache, (list, tuple))
                and len(kv_cache) >= 2
            ):
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        output_padded = None
        if key is not None and value is not None:
            output_padded = output
            query, key, value, output_padded = self.reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )
        # pooling model branch
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if output_padded is not None:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
        else:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
        output[:num_tokens] = attn_output[:num_tokens]
        return output


class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention backend implementation for INT8 KV cache (C8/QuaRot) models.

    This subclass handles static per-channel INT8 KV cache quantization.
    It is activated via class surgery in AscendC8KVCacheAttentionMethod.create_weights
    (vllm_ascend/quantization/methods/kv_c8.py)
    so that C8 attention layers automatically use this forward path.
    """

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendC8AttentionBackendImpl")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        self._prepare_c8_scales(layer, query.device)
        float_key, float_value = None, None
        if self.vllm_config.kv_transfer_config is None:
            if key is not None and value is not None:
                if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                    float_key, float_value = key, value
                key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                query, key, value, _ = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
            # pooling model branch
            if attn_metadata.model_runner_type == "pooling":
                attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                return self._forward_c8_decode(query, attn_metadata, output, layer)
            elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
                return self._forward_c8_chunked_prefill(query, float_key, float_value, attn_metadata, output, layer)
            else:
                return self._forward_c8_fused_infer_attention(
                    query,
                    float_key if float_key is not None else key,
                    float_value if float_value is not None else value,
                    attn_metadata,
                    output,
                    layer,
                )
        else:
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly and self.is_kv_producer:
                output_padded = None
                if key is not None and value is not None:
                    output_padded = output
                    query, key, value, output_padded = self.reshape_and_cache(
                        query, key, value, kv_cache, attn_metadata, output
                    )
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if output_padded is not None:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
                else:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            elif not self.is_kv_producer:
                if key is not None and value is not None:
                    key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                    query, key, value, _ = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                    return self._forward_c8_decode(query, attn_metadata, output, layer)

    def _prepare_c8_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C8 scales/offsets to this TP rank and pre-compute
        BF16 BNSD antiquant tensors for FIA V1 decode fast path.
        """
        if hasattr(layer, "_c8_scales_prepared"):
            return

        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
            if raw.numel() == 1:
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)

        layer._c8_k_scale = _shard_and_reshape(layer.k_cache_scale.data)
        layer._c8_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        layer._c8_v_scale = _shard_and_reshape(layer.v_cache_scale.data)
        layer._c8_v_offset = _shard_and_reshape(layer.v_cache_offset.data)

        bnsd = (1, self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale = layer._c8_k_scale.view(bnsd).contiguous()
        layer._c8_k_aq_offset = layer._c8_k_offset.view(bnsd).contiguous()
        layer._c8_v_aq_scale = layer._c8_v_scale.view(bnsd).contiguous()
        layer._c8_v_aq_offset = layer._c8_v_offset.view(bnsd).contiguous()

        layer._c8_k_inv_scale = 1.0 / layer._c8_k_scale
        layer._c8_v_inv_scale = 1.0 / layer._c8_v_scale

        layer._c8_scales_prepared = True

    def _dequant_paged_kv_to_dense(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list,
        target_dtype: torch.dtype,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged INT8 KV blocks and dequantize."""
        batch_size = block_table.shape[0]
        block_size = key.shape[1]
        H = key.shape[2]
        max_blocks_per_seq = block_table.shape[1]
        max_tokens_padded = max_blocks_per_seq * block_size

        flat_ids = block_table.reshape(-1)
        gathered_k = key[flat_ids].view(batch_size, max_tokens_padded, H)
        gathered_v = value[flat_ids].view(batch_size, max_tokens_padded, H)

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, H)[valid_mask]
        dense_v = gathered_v.view(-1, H)[valid_mask]

        dense_k = dense_k.view(-1, self.num_kv_heads, self.head_size)
        dense_v = dense_v.view(-1, self.num_kv_heads, self.head_size)
        dense_k = (dense_k.to(target_dtype) - layer._c8_k_offset) * layer._c8_k_scale
        dense_v = (dense_v.to(target_dtype) - layer._c8_v_offset) * layer._c8_v_scale
        return dense_k, dense_v

    def _quantize_kv_to_int8(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K/V from float to INT8 using static per-channel C8 scales."""
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        k_int8 = torch.clamp(
            torch.round(actual_key * layer._c8_k_inv_scale + layer._c8_k_offset),
            -128,
            127,
        ).to(torch.int8)
        v_int8 = torch.clamp(
            torch.round(actual_value * layer._c8_v_inv_scale + layer._c8_v_offset),
            -128,
            127,
        ).to(torch.int8)
        return k_int8, v_int8

    def _forward_c8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 decode via FIA V1 BNSD with native paged INT8 KV + perchannel antiquant."""
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
        assert block_size % 32 == 0, f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
        key = self.key_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
        value = self.value_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
        batch_size = len(attn_metadata.seq_lens_list)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._c8_k_aq_scale,
            key_antiquant_offset=layer._c8_k_aq_offset,
            value_antiquant_scale=layer._c8_v_aq_scale,
            value_antiquant_offset=layer._c8_v_aq_offset,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            sparse_mode=0,
        )
        attn_output = attn_output.squeeze(2)
        output[:batch_size] = attn_output
        return output

    def _forward_c8_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 ChunkedPrefill: decode via FIA V1 BNSD paged INT8 (zero gather),
        prefill via FIA V1 TND with float KV (new) or gather+dequant (continuing).
        """
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]

        if num_decode_tokens > 0:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
            assert block_size % 32 == 0, (
                f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
            )
            kv_k = self.key_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
            kv_v = self.value_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]

            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode_tokens].unsqueeze(2),
                kv_k,
                kv_v,
                key_antiquant_scale=layer._c8_k_aq_scale,
                key_antiquant_offset=layer._c8_k_aq_offset,
                value_antiquant_scale=layer._c8_v_aq_scale,
                value_antiquant_offset=layer._c8_v_aq_offset,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                block_size=block_size,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
                sparse_mode=0,
            )
            output[:num_decode_tokens] = attn_out.squeeze(2)

        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode_tokens:num_tokens]

            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]

            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                prefill_k = float_key[num_decode_tokens:num_tokens]
                prefill_v = float_value[num_decode_tokens:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                num_block, blk_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
                paged_k = self.key_cache.view(num_block, blk_size, -1)  # type: ignore[attr-defined]
                paged_v = self.value_cache.view(num_block, blk_size, -1)  # type: ignore[attr-defined]
                prefill_bt = attn_metadata.block_tables[num_decodes:]
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_paged_kv_to_dense(
                    paged_k, paged_v, prefill_bt, prefill_sl, query.dtype, layer
                )
                prefill_seq_kvlen = torch.tensor(prefill_sl, dtype=torch.int32).cumsum(dim=0)

            # block_table is None for prefill; FIA ignores block_size in this case.
            # Use cache block_size for consistency rather than a magic number.
            cache_block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q,
                key=prefill_k,
                value=prefill_v,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND",
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
            n_prefill = num_tokens - num_decode_tokens
            attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
            output[num_decode_tokens:num_tokens] = attn_out[:n_prefill]

        return output

    def _forward_c8_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ):
        """C8 FIA V1 TND for prefill states (PrefillNoCache uses float KV directly,
        PrefillCacheHit gathers + dequants paged INT8 KV).
        """
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv if isinstance(actual_seq_lengths_kv, list) else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_paged_kv_to_dense(key, value, block_table, seq_lens, query.dtype, layer)
                block_table = None
                # block_table is None after dequant; FIA ignores block_size.
                # Use cache block_size for consistency rather than a magic number.
                block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
                actual_seq_lengths_kv = torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)
            else:
                key = (key.to(query.dtype) - layer._c8_k_offset) * layer._c8_k_scale
                value = (value.to(query.dtype) - layer._c8_v_offset) * layer._c8_v_scale

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output
