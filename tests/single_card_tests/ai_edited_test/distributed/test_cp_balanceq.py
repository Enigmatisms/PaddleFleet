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

"""
Tests for the FlashMask balanceq context-parallel layout in
context_parallel_utils.py, its config validation, and the AttentionMetaInfo
transport used to carry the chunk assignment through the model.

Covers (single-card, mocked distributed groups):
  - is_cp_balanceq_mode (exact mode-name match)
  - require_cp_balanceq_buckets (missing-buckets guard)
  - buckets_to_tensor (Python buckets -> int32 chunk-id tensor)
  - reorder_chunked_tensor (chunk gather + axis restore, divisibility guard)
  - scatter_balanceq / all_gather_balanceq value correctness + nranks==1 clone
  - scatter <-> gather round trip restores global natural order (property test)
  - balanceq branches of ContextParallelScatterOp / GatherOp / AllGatherOp
    (forward and backward dispatch + buckets threading)
  - expand_cp_startend_row_indices_to_d2 (D=1 -> D=2 canonicalization)
  - AttentionMetaInfo / unpack_attention_meta type dispatch
  - TransformerConfig cp_balance_mode validation
"""

import os
import sys

# Insert local src/ before site-packages so we test the dev version.
_project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
sys.path.insert(0, os.path.join(_project_root, "src"))

import unittest
from unittest import mock

import paddle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_group(nranks=4, rank=1):
    """Create a mock context-parallel process group."""
    group = mock.MagicMock()
    group.nranks = nranks
    group.rank = rank
    return group


def _make_mock_hcg(nranks=4, rank=1):
    """Create a mock hybrid communicate group exposing a CP group."""
    hcg = mock.MagicMock()
    group = _make_mock_group(nranks, rank)
    hcg.get_context_parallel_group.return_value = group
    hcg.get_context_parallel_world_size.return_value = nranks
    return hcg, group


def _patch_fleet(hcg):
    """Patch fleet.get_hybrid_communicate_group to return the given hcg."""
    return mock.patch(
        "paddle.distributed.fleet.get_hybrid_communicate_group",
        return_value=hcg,
    )


# ===========================================================================
# Group 1: is_cp_balanceq_mode (pure function, prefix matching)
# ===========================================================================


class TestIsCpBalanceqMode(unittest.TestCase):
    """is_cp_balanceq_mode matches the single balanceq layout name exactly."""

    def test_balanceq_allgather_true(self):
        from paddlefleet.context_parallel_utils import is_cp_balanceq_mode

        self.assertTrue(is_cp_balanceq_mode("balanceq_allgather"))

    def test_non_balanceq_false(self):
        from paddlefleet.context_parallel_utils import is_cp_balanceq_mode

        self.assertFalse(is_cp_balanceq_mode("dualchunk_allgather"))
        self.assertFalse(is_cp_balanceq_mode("contiguous_allgather"))

    def test_unknown_suffix_false(self):
        """Exact match: unknown variants must not silently take the balanceq path."""
        from paddlefleet.context_parallel_utils import is_cp_balanceq_mode

        self.assertFalse(is_cp_balanceq_mode("balanceq_allgather_overlap"))


# ===========================================================================
# Group 2: require_cp_balanceq_buckets (missing-buckets guard)
# ===========================================================================


class TestRequireCpBalanceqBuckets(unittest.TestCase):
    """require_cp_balanceq_buckets fails fast when balanceq lacks buckets."""

    def test_without_buckets_raises(self):
        from paddlefleet.context_parallel_utils import (
            require_cp_balanceq_buckets,
        )

        with self.assertRaises(ValueError):
            require_cp_balanceq_buckets(None)

    def test_with_buckets_returns_them(self):
        from paddlefleet.context_parallel_utils import (
            require_cp_balanceq_buckets,
        )

        buckets = paddle.to_tensor([[0, 1], [2, 3]], dtype="int32")
        self.assertIs(require_cp_balanceq_buckets(buckets), buckets)


# ===========================================================================
# Group 3: buckets_to_tensor (Python buckets -> int32 chunk-id tensor)
# ===========================================================================


class TestBucketsToTensor(unittest.TestCase):
    """buckets_to_tensor extracts item[-1] into an int32 [cp, chunks] tensor."""

    def test_none_returns_none(self):
        from paddlefleet.context_parallel_utils import buckets_to_tensor

        self.assertIsNone(buckets_to_tensor(None))

    def test_list_of_tuples_extracts_last_field(self):
        from paddlefleet.context_parallel_utils import buckets_to_tensor

        # Each item is (workload, chunk_id); only the chunk id is kept.
        buckets = [[(0, 3), (0, 1)], [(0, 0), (0, 2)]]
        out = buckets_to_tensor(buckets)
        self.assertEqual(out.dtype, paddle.int32)
        self.assertEqual(list(out.shape), [2, 2])
        expected = paddle.to_tensor([[3, 1], [0, 2]], dtype="int32")
        self.assertTrue(paddle.equal_all(out, expected))

    def test_single_element_items_use_last(self):
        """item[-1] also works for length-1 items (bare chunk id)."""
        from paddlefleet.context_parallel_utils import buckets_to_tensor

        buckets = [[(5,), (4,)], [(1,), (0,)]]
        out = buckets_to_tensor(buckets)
        expected = paddle.to_tensor([[5, 4], [1, 0]], dtype="int32")
        self.assertTrue(paddle.equal_all(out, expected))


# ===========================================================================
# Group 4: reorder_chunked_tensor (chunk gather + axis restore)
# ===========================================================================


class TestReorderChunkedTensor(unittest.TestCase):
    """reorder_chunked_tensor gathers fixed-size chunks then restores layout."""

    def test_axis_1_reorders_chunks(self):
        from paddlefleet.context_parallel_utils import reorder_chunked_tensor

        # [1, 8, 2], 4 chunks of size 2 along axis 1, reordered to [3,1,0,2].
        x = paddle.arange(16).reshape([1, 8, 2]).cast("float32")
        chunk_ids = paddle.to_tensor([3, 1, 0, 2], dtype="int64")
        out = reorder_chunked_tensor(
            x, axis=1, chunk_ids=chunk_ids, total_chunks=4
        )
        expected = paddle.concat(
            [x[:, 6:8, :], x[:, 2:4, :], x[:, 0:2, :], x[:, 4:6, :]], axis=1
        )
        self.assertTrue(paddle.equal_all(out, expected))
        self.assertEqual(list(out.shape), [1, 8, 2])

    def test_axis_0_reorders_chunks(self):
        from paddlefleet.context_parallel_utils import reorder_chunked_tensor

        x = paddle.arange(8).reshape([4, 2]).cast("float32")
        chunk_ids = paddle.to_tensor([1, 0, 3, 2], dtype="int64")
        out = reorder_chunked_tensor(
            x, axis=0, chunk_ids=chunk_ids, total_chunks=4
        )
        expected = paddle.concat([x[1:2], x[0:1], x[3:4], x[2:3]], axis=0)
        self.assertTrue(paddle.equal_all(out, expected))

    def test_negative_axis_reorders_chunks(self):
        from paddlefleet.context_parallel_utils import reorder_chunked_tensor

        x = paddle.arange(8).reshape([2, 4]).cast("float32")
        chunk_ids = paddle.to_tensor([1, 0], dtype="int64")
        out = reorder_chunked_tensor(
            x, axis=-1, chunk_ids=chunk_ids, total_chunks=2
        )
        expected = paddle.concat([x[:, 2:4], x[:, 0:2]], axis=-1)
        self.assertTrue(paddle.equal_all(out, expected))

    def test_subset_of_chunks(self):
        """chunk_ids may select fewer chunks than total (the scatter case)."""
        from paddlefleet.context_parallel_utils import reorder_chunked_tensor

        x = paddle.arange(8).reshape([1, 8]).cast("float32")
        # total 4 chunks of size 2, pick only chunks [3, 0] -> shape [1, 4]
        chunk_ids = paddle.to_tensor([3, 0], dtype="int64")
        out = reorder_chunked_tensor(
            x, axis=1, chunk_ids=chunk_ids, total_chunks=4
        )
        expected = paddle.concat([x[:, 6:8], x[:, 0:2]], axis=1)
        self.assertTrue(paddle.equal_all(out, expected))
        self.assertEqual(list(out.shape), [1, 4])

    def test_non_divisible_raises(self):
        from paddlefleet.context_parallel_utils import reorder_chunked_tensor

        x = paddle.arange(10).reshape([1, 10]).cast("float32")
        chunk_ids = paddle.to_tensor([0, 1, 2], dtype="int64")
        with self.assertRaises(AssertionError):
            reorder_chunked_tensor(
                x, axis=1, chunk_ids=chunk_ids, total_chunks=3
            )


# ===========================================================================
# Group 5: scatter_balanceq / all_gather_balanceq value correctness
# ===========================================================================


class TestScatterBalanceq(unittest.TestCase):
    """scatter_balanceq selects this rank's chunks per the buckets table."""

    def test_nranks_1_returns_clone(self):
        from paddlefleet.context_parallel_utils import scatter_balanceq

        group = _make_mock_group(nranks=1, rank=0)
        x = paddle.arange(8).reshape([1, 8]).cast("float32")
        out = scatter_balanceq(x, group=group, axis=1, buckets=None)
        self.assertTrue(paddle.equal_all(out, x))
        out[0, 0] = -1.0
        self.assertNotEqual(x[0, 0].item(), -1.0)  # clone, not a view

    def test_selects_rank_chunks(self):
        from paddlefleet.context_parallel_utils import scatter_balanceq

        # 4 chunks of size 2 along axis 1, distributed to 2 ranks.
        x = paddle.arange(8).reshape([1, 8]).cast("float32")
        buckets = paddle.to_tensor([[3, 1], [0, 2]], dtype="int32")
        out0 = scatter_balanceq(
            x, group=_make_mock_group(2, 0), axis=1, buckets=buckets
        )
        self.assertTrue(
            paddle.equal_all(
                out0, paddle.concat([x[:, 6:8], x[:, 2:4]], axis=1)
            )
        )
        out1 = scatter_balanceq(
            x, group=_make_mock_group(2, 1), axis=1, buckets=buckets
        )
        self.assertTrue(
            paddle.equal_all(
                out1, paddle.concat([x[:, 0:2], x[:, 4:6]], axis=1)
            )
        )

    def test_group_none_uses_fleet(self):
        from paddlefleet.context_parallel_utils import scatter_balanceq

        hcg, _ = _make_mock_hcg(nranks=1, rank=0)
        x = paddle.arange(8).reshape([1, 8]).cast("float32")
        with _patch_fleet(hcg):
            out = scatter_balanceq(x, group=None, axis=1, buckets=None)
        self.assertTrue(paddle.equal_all(out, x))


class TestAllGatherBalanceq(unittest.TestCase):
    """all_gather_balanceq restores global natural order from rank-order gather."""

    def test_nranks_1_returns_clone(self):
        from paddlefleet.context_parallel_utils import all_gather_balanceq

        group = _make_mock_group(nranks=1, rank=0)
        x = paddle.arange(8).reshape([1, 8]).cast("float32")
        out = all_gather_balanceq(x, group=group, axis=1, buckets=None)
        self.assertTrue(paddle.equal_all(out, x))

    def test_restores_natural_order(self):
        from paddlefleet.context_parallel_utils import all_gather_balanceq

        buckets = paddle.to_tensor([[3, 1], [0, 2]], dtype="int32")
        natural = paddle.arange(8).reshape([1, 8]).cast("float32")
        c = [natural[:, i * 2 : (i + 1) * 2] for i in range(4)]
        # all_gather_contiguous yields rank-order concat: rank0 [3,1], rank1 [0,2].
        gathered = paddle.concat([c[3], c[1], c[0], c[2]], axis=1)
        group = _make_mock_group(nranks=2, rank=0)
        with mock.patch(
            "paddlefleet.context_parallel_utils.all_gather_contiguous",
            return_value=gathered,
        ):
            out = all_gather_balanceq(
                gathered[:, :4], group=group, axis=1, buckets=buckets
            )
        self.assertTrue(paddle.equal_all(out, natural))


# ===========================================================================
# Group 6: scatter <-> gather round trip (property test)
# ===========================================================================


class TestScatterGatherRoundTrip(unittest.TestCase):
    """scatter_balanceq then all_gather_balanceq must recover the global order.

    This exercises the argsort/reshape inverse logic that is invisible to
    static reading: any mismatch between scatter's buckets[rank] indexing
    and gather's argsort(buckets.reshape(-1)) immediately breaks the identity.
    """

    def _round_trip(self, buckets_list, cp_size, chunk_size, hidden):
        from paddlefleet.context_parallel_utils import (
            all_gather_balanceq,
            scatter_balanceq,
        )

        buckets = paddle.to_tensor(buckets_list, dtype="int32")
        total_chunks = cp_size * len(buckets_list[0])
        seq_len = total_chunks * chunk_size
        x = (
            paddle.arange(seq_len * hidden)
            .reshape([1, seq_len, hidden])
            .cast("float32")
        )
        # Scatter each rank, then rank-order concat == all_gather_contiguous output.
        local_shards = [
            scatter_balanceq(
                x, group=_make_mock_group(cp_size, r), axis=1, buckets=buckets
            )
            for r in range(cp_size)
        ]
        gathered = paddle.concat(local_shards, axis=1)
        group = _make_mock_group(nranks=cp_size, rank=0)
        with mock.patch(
            "paddlefleet.context_parallel_utils.all_gather_contiguous",
            return_value=gathered,
        ):
            restored = all_gather_balanceq(
                local_shards[0], group=group, axis=1, buckets=buckets
            )
        self.assertTrue(paddle.equal_all(restored, x))

    def test_2_rank_shuffled(self):
        self._round_trip([[3, 1], [0, 2]], cp_size=2, chunk_size=2, hidden=3)

    def test_4_rank_shuffled(self):
        self._round_trip(
            [[6, 2], [5, 0], [7, 3], [4, 1]],
            cp_size=4,
            chunk_size=2,
            hidden=2,
        )

    def test_identity_buckets(self):
        self._round_trip([[0, 1], [2, 3]], cp_size=2, chunk_size=4, hidden=1)


# ===========================================================================
# Group 7: PyLayer op balanceq branches (forward/backward dispatch + buckets)
# ===========================================================================

_BUCKETS = [[3, 1], [0, 2]]


def _buckets_tensor():
    return paddle.to_tensor(_BUCKETS, dtype="int32")


class TestScatterOpBalanceq(unittest.TestCase):
    """ContextParallelScatterOp balanceq forward/backward dispatch."""

    def test_forward_calls_scatter_balanceq(self):
        from paddlefleet.context_parallel_utils import ContextParallelScatterOp

        hcg, _ = _make_mock_hcg(nranks=2, rank=0)
        x = paddle.randn([1, 8])
        buckets = _buckets_tensor()
        with (
            _patch_fleet(hcg),
            mock.patch(
                "paddlefleet.context_parallel_utils.scatter_balanceq",
                return_value=x[:, :4],
            ) as mock_fn,
        ):
            ctx = mock.MagicMock()
            ContextParallelScatterOp.forward(
                ctx, x, axis=1, mode="balanceq_allgather", buckets=buckets
            )
        mock_fn.assert_called_once()
        # buckets must reach the primitive and be cached on ctx for backward.
        self.assertIs(mock_fn.call_args.kwargs["buckets"], buckets)
        self.assertIs(ctx.buckets, buckets)
        self.assertEqual(ctx.mode, "balanceq_allgather")

    def test_backward_calls_all_gather_balanceq(self):
        from paddlefleet.context_parallel_utils import ContextParallelScatterOp

        ctx = mock.MagicMock()
        ctx.mode = "balanceq_allgather"
        ctx.axis = 1
        ctx.group = _make_mock_group(nranks=2, rank=0)
        ctx.buckets = _buckets_tensor()
        grad = paddle.randn([1, 4])
        with mock.patch(
            "paddlefleet.context_parallel_utils.all_gather_balanceq",
            return_value=paddle.randn([1, 8]),
        ) as mock_fn:
            ContextParallelScatterOp.backward(ctx, grad)
        mock_fn.assert_called_once()
        self.assertIs(mock_fn.call_args.kwargs["buckets"], ctx.buckets)


class TestGatherOpBalanceq(unittest.TestCase):
    """ContextParallelGatherOp balanceq forward/backward dispatch."""

    def test_forward_calls_all_gather_balanceq(self):
        from paddlefleet.context_parallel_utils import ContextParallelGatherOp

        hcg, _ = _make_mock_hcg(nranks=2, rank=0)
        x = paddle.randn([1, 4])
        buckets = _buckets_tensor()
        with (
            _patch_fleet(hcg),
            mock.patch(
                "paddlefleet.context_parallel_utils.all_gather_balanceq",
                return_value=paddle.randn([1, 8]),
            ) as mock_fn,
        ):
            ctx = mock.MagicMock()
            ContextParallelGatherOp.forward(
                ctx, x, axis=1, mode="balanceq_allgather", buckets=buckets
            )
        mock_fn.assert_called_once()
        self.assertIs(mock_fn.call_args.kwargs["buckets"], buckets)
        self.assertEqual(ctx.mode, "balanceq_allgather")

    def test_backward_calls_scatter_balanceq(self):
        from paddlefleet.context_parallel_utils import ContextParallelGatherOp

        ctx = mock.MagicMock()
        ctx.mode = "balanceq_allgather"
        ctx.axis = 1
        ctx.group = _make_mock_group(nranks=2, rank=0)
        ctx.buckets = _buckets_tensor()
        grad = paddle.randn([1, 8])
        with mock.patch(
            "paddlefleet.context_parallel_utils.scatter_balanceq",
            return_value=paddle.randn([1, 4]),
        ) as mock_fn:
            ContextParallelGatherOp.backward(ctx, grad)
        mock_fn.assert_called_once()
        self.assertIs(mock_fn.call_args.kwargs["buckets"], ctx.buckets)


class TestAllGatherOpBalanceq(unittest.TestCase):
    """ContextParallelAllGatherOp balanceq forward/backward dispatch.

    AllGatherOp intentionally keeps rank order (no buckets reorder), so
    forward uses all_gather_contiguous and backward uses reduce_scatter_contiguous.
    """

    def test_forward_calls_all_gather_contiguous(self):
        from paddlefleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        hcg, _ = _make_mock_hcg(nranks=2, rank=0)
        x = paddle.randn([1, 4])
        with (
            _patch_fleet(hcg),
            mock.patch(
                "paddlefleet.context_parallel_utils.all_gather_contiguous",
                return_value=paddle.randn([1, 8]),
            ) as mock_fn,
        ):
            ctx = mock.MagicMock()
            ContextParallelAllGatherOp.forward(
                ctx, x, axis=1, mode="balanceq_allgather"
            )
        mock_fn.assert_called_once()
        self.assertEqual(ctx.mode, "balanceq_allgather")

    def test_forward_rejects_buckets_kwarg(self):
        """No buckets parameter by design: rank order is intentional here."""
        from paddlefleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        with self.assertRaises(TypeError):
            ContextParallelAllGatherOp.forward(
                mock.MagicMock(),
                paddle.randn([1, 4]),
                axis=1,
                mode="balanceq_allgather",
                buckets=_buckets_tensor(),
            )

    def test_backward_calls_reduce_scatter_contiguous(self):
        from paddlefleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        ctx = mock.MagicMock()
        ctx.mode = "balanceq_allgather"
        ctx.axis = 1
        ctx.group = _make_mock_group(nranks=2, rank=0)
        ctx.buckets = _buckets_tensor()
        grad = paddle.randn([1, 8])
        with mock.patch(
            "paddlefleet.context_parallel_utils.reduce_scatter_contiguous",
            return_value=paddle.randn([1, 4]),
        ) as mock_fn:
            ContextParallelAllGatherOp.backward(ctx, grad)
        mock_fn.assert_called_once()


# ===========================================================================
# Group 8: TransformerConfig balanceq validation
# ===========================================================================


class TestTransformerConfigBalanceqValidation(unittest.TestCase):
    """__post_init__ validation around cp_balance_mode='balanceq_allgather'."""

    @staticmethod
    def _config(**kwargs):
        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        return TransformerConfig(**kwargs)

    def test_balanceq_with_dataflow_constructs(self):
        cfg = self._config(
            cp_balance_mode="balanceq_allgather", experimental_dataflow=True
        )
        self.assertEqual(cfg.cp_balance_mode, "balanceq_allgather")

    def test_unknown_mode_rejected(self):
        """A typo'd mode must not silently fall back to dualchunk."""
        with self.assertRaises(ValueError):
            self._config(cp_balance_mode="bogus_mode")
        with self.assertRaises(ValueError):
            self._config(
                cp_balance_mode="balanceq_allgather_overlap",
                experimental_dataflow=True,
            )

    def test_balanceq_requires_dataflow(self):
        """Buckets are built at the dataflow entry, so balanceq needs it on."""
        with self.assertRaises(AssertionError):
            self._config(cp_balance_mode="balanceq_allgather")

    def test_default_mode_unaffected(self):
        cfg = self._config()
        self.assertEqual(cfg.cp_balance_mode, "dualchunk_allgather")


# ===========================================================================
# Group 9: expand_cp_startend_row_indices_to_d2
# ===========================================================================


class TestExpandCpStartendRowIndicesToD2(unittest.TestCase):
    """CP kernels need the D=2 mask layout; D=1 gains an arange row column."""

    def test_d1_appends_arange(self):
        from paddlefleet.context_parallel_utils import (
            expand_cp_startend_row_indices_to_d2,
        )

        seq_len = 6
        mask = paddle.full([1, 1, seq_len, 1], seq_len, dtype="int32").cuda()
        out = expand_cp_startend_row_indices_to_d2(mask, seq_len)
        self.assertEqual(list(out.shape), [1, 1, seq_len, 2])
        self.assertTrue(paddle.equal_all(out[..., 0], mask[..., 0]))
        expected = (
            paddle.arange(seq_len, dtype="int32").cuda().reshape([1, 1, -1])
        )
        self.assertTrue(paddle.equal_all(out[..., 1], expected))

    def test_d2_passthrough_is_identity(self):
        from paddlefleet.context_parallel_utils import (
            expand_cp_startend_row_indices_to_d2,
        )

        mask = paddle.zeros([1, 1, 4, 2], dtype="int32")
        self.assertIs(expand_cp_startend_row_indices_to_d2(mask, 4), mask)

    def test_other_last_dim_raises(self):
        from paddlefleet.context_parallel_utils import (
            expand_cp_startend_row_indices_to_d2,
        )

        mask = paddle.zeros([1, 1, 4, 3], dtype="int32")
        with self.assertRaises(ValueError):
            expand_cp_startend_row_indices_to_d2(mask, 4)


# ===========================================================================
# Group 10: AttentionMetaInfo transport
# ===========================================================================


class TestUnpackAttentionMeta(unittest.TestCase):
    """The mask slot carries either a bare Tensor/None or an AttentionMetaInfo."""

    def test_none_passthrough(self):
        from paddlefleet.attention_meta_info import unpack_attention_meta

        self.assertEqual(unpack_attention_meta(None), (None, None))

    def test_tensor_passthrough_keeps_identity(self):
        from paddlefleet.attention_meta_info import unpack_attention_meta

        mask = paddle.zeros([1, 1, 4, 2], dtype="int32")
        out_mask, buckets = unpack_attention_meta(mask)
        self.assertIs(out_mask, mask)
        self.assertIsNone(buckets)

    def test_meta_info_unpacks_both_fields(self):
        from paddlefleet.attention_meta_info import (
            AttentionMetaInfo,
            unpack_attention_meta,
        )

        mask = paddle.zeros([1, 1, 4, 2], dtype="int32")
        buckets = _buckets_tensor()
        out_mask, out_buckets = unpack_attention_meta(
            AttentionMetaInfo(
                startend_row_indices=mask, cp_balance_buckets=buckets
            )
        )
        self.assertIs(out_mask, mask)
        self.assertIs(out_buckets, buckets)

    def test_empty_meta_info_defaults_to_none(self):
        from paddlefleet.attention_meta_info import (
            AttentionMetaInfo,
            unpack_attention_meta,
        )

        self.assertEqual(
            unpack_attention_meta(AttentionMetaInfo()), (None, None)
        )


if __name__ == "__main__":
    unittest.main()
