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

"""Paddle wrapper around the cuDNN-frontend fused CSA compressor.

Replaces the gather / +ape / softmax / weighted-sum region of
``Compressor.forward`` with the fused kernels in
``paddlefleet_ops.cudnn.csa.compressor``, which address blocks through two THD
prefix sums instead of a gather index table. HCA only: ratio 128, own-block
window (``coff == 1``), bf16 projections and fp32 ``ape``.
"""

from __future__ import annotations

import math
import os
import sys

import paddle
from paddle.autograd import PyLayer

# cuDNN snapshots one launch and replays it by writing new values into each
# argument's storage, while paddlefleet_ops memoizes CUTLASS scalar storage keyed
# by VALUE (paddlefleet_ops/__init__.py:31-81), so two scalars that were equal
# when the snapshot was taken share one box. The backward's ``n_seq`` then takes
# ``rows_per_cta``'s value on every later call: missing gradients when the stale
# value is below the real segment count, CUDA 700 when above. The flag is read on
# every launch (compressor_sm100.py:123), so setting it here is enough.
os.environ["CUDNNFE_CSA_COMPRESSOR_FAST_LAUNCH"] = "0"

if not hasattr(paddle.device.Stream, "cuda_stream"):
    # compressor_sm100_r128.py:567 reads it off what it thinks is torch; same
    # bridge as cudnn_ops/attn/csa_sparse_attn_bwd_cudnn.py:122.
    paddle.device.Stream.cuda_stream = property(
        lambda self: self.stream_base.cuda_stream
    )

if not hasattr(paddle, "are_deterministic_algorithms_enabled"):
    # api.py:142 refuses the backward under deterministic mode, dAPE being an
    # fp32 atomic accumulator. Answer with paddle's own flag.
    paddle.are_deterministic_algorithms_enabled = lambda: bool(
        paddle.get_flags("FLAGS_cudnn_deterministic")["FLAGS_cudnn_deterministic"]
    )


def _int_numel(*tensors):
    """Give these tensors a torch-style int ``numel()``.

    The launchers hand ``cu_seqlens.numel()`` and ``kv.numel()`` straight to
    ``cutlass.Int32`` (compressor_sm100_r128.py:625,1233), which rejects paddle's
    Tensor, and ``api.py:426,523`` branch on ``out.numel()`` / ``grad_out.numel()``,
    which would sync. Only these five tensors are read that way, so ``ape`` and the
    gradient buffers keep paddle's semantics.
    """
    for t in tensors:
        t.numel = lambda n=math.prod(t.shape): n


_APIS: dict = {}


def _api(kind, ratio, kv, score, ape, cu_seqlens, cu_seqlens_comp, out):
    """One compiled API per (kind, shape); the kernel JIT itself is cached in cuDNN."""
    from paddlefleet_ops import cudnn

    # CSA symbols are lazy attributes (cudnn/__init__.py:317-321) imported after the
    # loader dropped the top-level ``cudnn`` alias (paddlefleet_ops/utils.py:87-102),
    # so their package-internal ``from cudnn.api_base import ...`` needs it back.
    sys.modules.setdefault("cudnn", cudnn)
    key = (kind, ratio, kv.shape[0], kv.shape[1], out.shape[0], cu_seqlens.shape[0])
    api = _APIS.get(key)
    if api is None:
        cls = (
            cudnn.CSACompressorForward
            if kind == "fwd"
            else cudnn.CSACompressorBackward
        )
        api = cls(
            sample_kv=kv,
            sample_score=score,
            sample_ape=ape,
            sample_cu_seqlens=cu_seqlens,
            sample_cu_seqlens_comp=cu_seqlens_comp,
            sample_out=out,
            ratio=ratio,
            coff=1,
        )
        api.check_support()
        api.compile()
        _APIS[key] = api
    return api


class _FusedCompressor(PyLayer):
    @staticmethod
    def forward(ctx, kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp):
        ratio = ape.shape[0]
        if total_comp == 0:
            out = paddle.zeros([0, kv.shape[1]], dtype=kv.dtype)
        else:
            out = paddle.empty([total_comp, kv.shape[1]], dtype=kv.dtype)
            _int_numel(kv, cu_seqlens, cu_seqlens_comp, out)
            _api(
                "fwd", ratio, kv, score, ape, cu_seqlens, cu_seqlens_comp, out
            ).execute(kv, score, ape, cu_seqlens, cu_seqlens_comp, out)
        ctx.save_for_backward(kv, score, ape, cu_seqlens, cu_seqlens_comp)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        kv, score, ape, cu_seqlens, cu_seqlens_comp = ctx.saved_tensor()
        grad_kv = paddle.zeros_like(kv)
        grad_score = paddle.zeros_like(score)
        grad_ape = paddle.zeros_like(ape)  # fp32 atomic accumulator
        if grad_out.shape[0] != 0:
            grad_out = grad_out.contiguous()
            _int_numel(kv, cu_seqlens, cu_seqlens_comp, grad_out)
            _api(
                "bwd",
                ape.shape[0],
                kv,
                score,
                ape,
                cu_seqlens,
                cu_seqlens_comp,
                grad_out,
            ).execute(
                kv,
                score,
                ape,
                cu_seqlens,
                cu_seqlens_comp,
                grad_out,
                grad_kv,
                grad_score,
                grad_ape,
            )
        return grad_kv, grad_score, grad_ape


def csa_compressor_cudnn(kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp):
    """Gated pooling of ``ratio``-token blocks, addressed by two THD prefix sums.

    ``kv`` / ``score`` are ``[total_tokens, head_dim]`` bf16, ``ape`` is
    ``[ratio, head_dim]`` fp32, both prefix sums int32. Returns
    ``[total_comp, head_dim]`` bf16 before RMSNorm; rows past
    ``cu_seqlens_comp[-1]`` are static capacity and hold a recomputed copy of the
    first block by design (api.py:371-374).
    """
    return _FusedCompressor.apply(
        kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp
    )
