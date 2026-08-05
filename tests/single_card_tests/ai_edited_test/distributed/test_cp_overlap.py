# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Tests for the overlapped FlashMask context parallel path.

Covers:
  - the "_overlap"/"_nonoverlap" suffix of cp_balance_mode, and that configs
    written without a suffix keep meaning non-overlap;
  - that the localized mask is bit-identical to the validated FM-4 reference
    pipeline (preprocess_index_dual_chunks + rearrange_blocks + roll), and to its
    hierarchical variant (+ hier_map_chunk permutation) under the kernel's
    FLASHMASK_USE_HIERARCHICAL switch;
  - flashmask_attention_cp dispatch, and that overlap=False never reaches the
    overlap module;
  - rejection of the features the overlapped kernel does not implement.
"""

import os
import unittest
from unittest import mock

import paddle

from paddlefleet import (
    context_parallel_utils as cpu,
    overlap_context_parallel as ocp,
)
from paddlefleet.transformer.transformer_config import TransformerConfig


def _config(mode, **kwargs):
    return TransformerConfig(
        hidden_size=8, num_attention_heads=1, cp_balance_mode=mode, **kwargs
    )


class _Group:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size


def _hier_map_chunk(logical_pos, my_pe, total_n_pes, gpus_per_node):
    """Rank mapping of dist_flashmask_dev/overlap_flashmask_fm4.py."""
    my_pe_node = my_pe % gpus_per_node
    my_node_id = my_pe // gpus_per_node
    num_nodes = total_n_pes // gpus_per_node
    if logical_pos < num_nodes:
        return (
            my_pe_node
            + ((my_node_id + logical_pos) % num_nodes) * gpus_per_node
        )
    adj_pos = logical_pos - num_nodes
    slot = adj_pos // num_nodes + 1
    sub = adj_pos % num_nodes
    base = (my_pe_node + slot) % gpus_per_node
    return base + ((my_node_id + sub) % num_nodes) * gpus_per_node


def _reference_masks(
    startend_row_indices, rank, cp_size, seqlen_local, gpus_per_node=0
):
    """Mask pipeline of dist_flashmask_dev/overlap_flashmask_fm4.py."""
    seq_blocksize = seqlen_local // 2
    mask = cpu.preprocess_index_dual_chunks(
        startend_row_indices,
        chunk_id_first=rank,
        chunk_id_second=2 * cp_size - rank - 1,
        seq_blocksize=seq_blocksize,
        max_seqlen_q=seq_blocksize,
    )
    # rearrange_blocks
    batch_size, _, seqlen, _ = mask.shape
    n_blocks = cp_size * 2
    order = []
    for i in range(cp_size):
        order += [i, n_blocks - 1 - i]
    blocks = mask.reshape([batch_size, -1, n_blocks, seqlen // n_blocks, 2])
    mask = blocks.index_select(
        index=paddle.to_tensor(order, dtype="int64"), axis=2
    ).reshape([batch_size, -1, seqlen, 2])

    if not gpus_per_node:
        return (
            paddle._C_ops.roll(mask, shifts=-seqlen_local * (rank + 1), axis=2),
            paddle._C_ops.roll(mask, shifts=-seqlen_local * rank, axis=2),
        )

    def _permute(positions):
        perm = paddle.to_tensor(
            [
                _hier_map_chunk(pos, rank, cp_size, gpus_per_node)
                for pos in positions
            ],
            dtype="int64",
        )
        return (
            mask.reshape([batch_size, -1, cp_size, seqlen_local, 2])
            .index_select(perm, axis=2)
            .reshape([batch_size, -1, seqlen, 2])
        )

    return (
        _permute(range(cp_size - 1, -1, -1)),
        _permute(range(cp_size)),
    )


def _localized_masks(startend_row_indices, rank, cp_size, seqlen_local):
    group = _Group(rank, cp_size)
    mask = ocp.localize_mask(
        startend_row_indices, seqlen_local, group, "dualchunk_allgather"
    )
    return (
        ocp.gathered_kv_order(mask, group, False),
        ocp.gathered_kv_order(mask, group, True),
    )


class TestCpOverlapConfig(unittest.TestCase):
    def test_suffix_normalized_into_cp_overlap(self):
        for mode, expected in (
            ("dualchunk_allgather", False),
            ("dualchunk_allgather_nonoverlap", False),
            ("dualchunk_allgather_overlap", True),
        ):
            config = _config(mode)
            self.assertEqual(config.cp_balance_mode, "dualchunk_allgather")
            self.assertEqual(config.cp_overlap, expected)

    def test_suffixless_modes_untouched(self):
        for mode in ("contiguous_allgather", "contiguous_a2a"):
            config = _config(mode)
            self.assertEqual(config.cp_balance_mode, mode)
            self.assertFalse(config.cp_overlap)

    def test_overlap_requires_dualchunk(self):
        with self.assertRaises(ValueError):
            _config("contiguous_allgather_overlap")
        with self.assertRaises(ValueError):
            _config("contiguous_a2a", cp_overlap=True)

    def test_unrecognized_suffix_still_rejected(self):
        # Only the exact "_overlap"/"_nonoverlap" suffixes are recognized; a
        # near miss must fall through to the layout whitelist.
        with self.assertRaises(ValueError):
            _config("dualchunk_allgather_overlapping")


class TestCpOverlapMask(unittest.TestCase):
    def setUp(self):
        # The traversal cache samples the environment on the first miss, like the
        # kernel does; tests that switch the environment have to invalidate it.
        ocp.BLOCK_ORDER_CACHE.clear()
        self.addCleanup(ocp.BLOCK_ORDER_CACHE.clear)

    def _check(self, cp_size, gpus_per_node=0):
        seqlen_total = 32 * cp_size
        seqlen_local = seqlen_total // cp_size
        startend_row_indices = paddle.randint(
            0, seqlen_total, [2, 1, seqlen_total, 2], dtype="int32"
        )
        for rank in range(cp_size):
            got = _localized_masks(
                startend_row_indices, rank, cp_size, seqlen_local
            )
            expected = _reference_masks(
                startend_row_indices,
                rank,
                cp_size,
                seqlen_local,
                gpus_per_node,
            )
            for actual, ref in zip(got, expected):
                self.assertTrue(bool((actual == ref).all()))

    def test_matches_reference_pipeline(self):
        for cp_size in (1, 2, 4, 8):
            self._check(cp_size)

    def test_matches_hierarchical_reference_pipeline(self):
        for gpus_per_node, cp_size in ((2, 4), (2, 8), (4, 8), (8, 16)):
            ocp.BLOCK_ORDER_CACHE.clear()
            with mock.patch.dict(
                os.environ,
                {
                    "FLASHMASK_USE_HIERARCHICAL": "true",
                    "HIERARCHICAL_GPUS_PER_NODE": str(gpus_per_node),
                },
            ):
                self._check(cp_size, gpus_per_node)

    def test_hierarchical_switch_follows_kernel_semantics(self):
        # Unset, unrecognized and false-ish values leave the traversal circular;
        # a single node does too, because the kernel falls back to it as well.
        for value, cp_size, expected in (
            (None, 8, 0),
            ("", 8, 0),
            ("0", 8, 0),
            ("off", 8, 0),
            ("maybe", 8, 0),
            ("TRUE", 8, 0),
            ("TRUE", 16, 8),
            ("yes", 16, 8),
        ):
            environ = (
                {} if value is None else {"FLASHMASK_USE_HIERARCHICAL": value}
            )
            with mock.patch.dict(os.environ, environ, clear=True):
                self.assertEqual(
                    ocp.hierarchical_gpus_per_node(cp_size), expected
                )

    def test_unsupported_layout_rejected(self):
        with self.assertRaises(ValueError):
            ocp.localize_mask(
                paddle.zeros([1, 1, 8, 2], dtype="int32"),
                4,
                _Group(0, 2),
                "contiguous_allgather",
            )


class TestCpOverlapDispatch(unittest.TestCase):
    def _query(self, seqlen=8):
        return paddle.zeros([1, seqlen, 1, 8], dtype="bfloat16")

    def test_overlap_mode_routes_to_overlap_layer(self):
        with mock.patch.object(
            ocp, "overlap_flashmask_attention_cp", return_value="overlapped"
        ) as patched:
            output = cpu.flashmask_attention_cp(
                self._query(),
                self._query(),
                self._query(),
                paddle.zeros([1, 1, 8, 2], dtype="int32"),
                mode="dualchunk_allgather_overlap",
            )
        self.assertEqual(output, "overlapped")
        patched.assert_called_once()
        # The suffix is stripped before it reaches the overlap layer.
        self.assertEqual(
            patched.call_args[0][-1],
            "dualchunk_allgather",
        )

    def test_default_does_not_reach_overlap_layer(self):
        with (
            mock.patch.object(ocp, "overlap_flashmask_attention_cp") as patched,
            self.assertRaises(ValueError),
        ):
            cpu.flashmask_attention_cp(
                self._query(),
                self._query(),
                self._query(),
                paddle.zeros([1, 1, 8, 2], dtype="int32"),
                mode="unsupported_mode",
            )
        patched.assert_not_called()


class TestCpOverlapUnsupported(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(ocp, "OVERLAP_SUPPORTED", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _call(self, seqlen=8, **kwargs):
        query = paddle.zeros([1, seqlen, 1, 8], dtype="bfloat16")
        return ocp.overlap_flashmask_attention_cp(
            query,
            query,
            query,
            paddle.zeros([1, 1, seqlen, 2], dtype="int32"),
            **kwargs,
        )

    def test_rejects_unimplemented_features(self):
        for kwargs in (
            {"dropout": 0.1},
            {"causal": True},
            {"fixed_seed_offset": paddle.zeros([1], dtype="int64")},
            {"learnable_sink": paddle.zeros([1], dtype="bfloat16")},
        ):
            with self.assertRaises(NotImplementedError):
                self._call(**kwargs)

    def test_rejects_odd_local_sequence_length(self):
        with self.assertRaises(AssertionError):
            self._call(seqlen=7)

    def test_requires_capable_build(self):
        with (
            mock.patch.object(ocp, "OVERLAP_SUPPORTED", False),
            self.assertRaises(AssertionError),
        ):
            self._call()


if __name__ == "__main__":
    unittest.main()
