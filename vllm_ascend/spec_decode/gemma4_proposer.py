# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

logger = init_logger(__name__)


class AscendGemma4Proposer(Gemma4Proposer, AscendSpecDecodeBaseProposer):
    """Ascend adaptation of Gemma4 assistant MTP.

    Gemma4 needs the upstream Gemma4 proposer semantics (tuple hidden states,
    own lm_head, multi-group KV sharing) plus Ascend's NPU proposer runtime.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        AscendSpecDecodeBaseProposer.__init__(
            self,
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        # Gemma4 MTP has mixed local/global attention heads and a 512-dim
        # global head. Full ACL graph padding makes small online batches run at
        # graph capture sizes, which is too expensive for this drafter path.
        self.use_cuda_graph = False
        self._runnable = self._run_merged_draft
        self.constant_draft_positions = True
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        self._centroids_sizes: list[int] = []
        self._centroids_graphs: dict[int, object] = {}
        self._centroids_inputs: dict[int, torch.Tensor] = {}
        self._centroids_outputs: dict[int, torch.Tensor] = {}

    def _setup_gemma4_kv_sharing(
        self,
        target_attn_layer_names: set[str],
    ) -> None:
        """Propagate post-construction KV sharing mappings to Ascend backends."""
        super()._setup_gemma4_kv_sharing(target_attn_layer_names)

        if not (hasattr(self.model, "model") and hasattr(self.model.model, "layers")):
            return

        # Gemma4Proposer wires the high-level Attention modules after model
        # construction. AscendAttentionBackendImpl receives its own copy at
        # construction, so it must be updated explicitly as well.
        for draft_idx, layer in enumerate(self.model.model.layers):
            attn = getattr(getattr(layer, "self_attn", None), "attn", None)
            target_layer_name = getattr(attn, "kv_sharing_target_layer_name", None)
            impl = getattr(attn, "impl", None)
            if target_layer_name is None or impl is None:
                continue
            impl.kv_sharing_target_layer_name = target_layer_name
            logger.info(
                "Gemma4 MTP: synced draft layer %d backend KV target -> %s",
                draft_idx,
                target_layer_name,
            )

    def build_constant_position_multi_step_metadata(
        self,
        common_attn_metadata,
        initial_per_layer_attn_metadata: dict[str, object],
        batch_size: int,
        num_input_tokens: int,
        used_update_positions: torch.Tensor,
        aclgraph_runtime_mode,
        ori_seq_len=None,
        slot_indices=None,
        mtp_slot_mapping=None,
    ) -> list[dict[str, object]]:
        """Build Gemma4 draft metadata without advancing target state.

        Gemma4 assistant steps reuse the last target position. Each attention
        group owns a distinct KV block table, so its metadata must be built
        independently and retained for only that group's layers.
        """
        multi_steps_attn_metadata: list[dict[str, object]] = []
        step_slot_indices = slot_indices

        for draft_step in range(1, self.num_speculative_tokens):
            per_layer_attn_metadata: dict[str, object] = {}
            for attn_group in self.draft_attn_groups:
                group_common_attn_metadata = self.shallow_copy_metadata(common_attn_metadata)
                group_id = attn_group.kv_cache_group_id
                group_block_table = self._per_group_block_tables.get(group_id)
                if group_block_table is not None:
                    group_common_attn_metadata.block_table_tensor = group_block_table[:batch_size]

                first_layer_name = attn_group.layer_names[0]
                _, attn_metadata = self.attn_update_stack_num_spec_norm(
                    draft_step,
                    initial_per_layer_attn_metadata[first_layer_name],
                    group_common_attn_metadata,
                    batch_size,
                    num_input_tokens,
                    used_update_positions.clone(),
                    aclgraph_runtime_mode,
                    ori_seq_len,
                    None if step_slot_indices is None else step_slot_indices.clone(),
                    mtp_slot_mapping,
                    attn_group=attn_group,
                    keep_positions_and_seq_lens=True,
                )
                # The Ascend builder retains a view into slot_mapping_group.
                # Preserve this group's mapping before the next group updates it.
                attn_metadata.slot_mapping = attn_metadata.slot_mapping.clone()
                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata

            multi_steps_attn_metadata.append(per_layer_attn_metadata)
            if step_slot_indices is not None:
                step_slot_indices = step_slot_indices + self.pcp_size

        return multi_steps_attn_metadata

    def _setup_centroids_cuda_graphs(self) -> None:
        """Skip CUDA graph capture on Ascend.

        The inherited Gemma4 fast path captures CUDA graphs for sparse centroid
        argmax. Ascend should first use the regular NPU/eager logits path; NPU
        graph optimization can be added later behind a separate validation step.
        """
        logger.info("Gemma4 MTP: skip CUDA centroid graph capture on Ascend.")

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if getattr(self.model, "masked_embedding", None) is not None:
            return self.model.get_top_tokens(hidden_states)
        return super()._greedy_sample(hidden_states)
