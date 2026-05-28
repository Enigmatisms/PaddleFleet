import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

__all__ = [
    "csa_attn_target_reducesum",
    "csa_indexer_bwd",
    "csa_indexer_topk_fwd",
]


def __getattr__(name):
    if name in {
        "csa_attn_target_reducesum",
        "csa_indexer_bwd",
        "csa_indexer_topk_fwd",
    }:
        from .indexer.csa_indexer import (
            csa_attn_target_reducesum,
            csa_indexer_bwd,
            csa_indexer_topk_fwd,
        )

        exports = {
            "csa_attn_target_reducesum": csa_attn_target_reducesum,
            "csa_indexer_bwd": csa_indexer_bwd,
            "csa_indexer_topk_fwd": csa_indexer_topk_fwd,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
