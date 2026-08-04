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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddle import Tensor


@dataclass
class AttentionMetaInfo:
    """Attention metadata travelling in the ``attn_mask_startend_row_indices``
    slot, so extra attention inputs need no new parameter on every forward.

    Only constructed under CP balanceq; other layouts keep the slot a bare
    ``Tensor`` / ``None``. Consumers dispatch on type via
    :func:`unpack_attention_meta`.
    """

    startend_row_indices: Tensor | None = None
    cp_balance_buckets: Tensor | None = None


def unpack_attention_meta(startend_row_indices):
    """Unpack the mask slot into ``(mask_tensor, cp_balance_buckets)``."""
    if isinstance(startend_row_indices, AttentionMetaInfo):
        return (
            startend_row_indices.startend_row_indices,
            startend_row_indices.cp_balance_buckets,
        )
    return startend_row_indices, None
