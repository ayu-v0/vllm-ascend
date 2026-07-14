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
        self._centroids_sizes: list[int] = []
        self._centroids_graphs: dict[int, object] = {}
        self._centroids_inputs: dict[int, torch.Tensor] = {}
        self._centroids_outputs: dict[int, torch.Tensor] = {}

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
