#!/usr/bin/env python3
"""
Standalone Drop-In Fused Triton Bregman-Taylor KL Divergence Kernel.

Accompanying artifact for:
  Marcio Diaz & Space AI (2026),
  "Silent Cancellation and Backward Gradient Corruption in Low-Precision KL Divergence:
   Theory, Fused Triton Kernels, and LLM Post-Training" (Version 3.0)
  Zenodo DOI: https://doi.org/10.5281/zenodo.24513063

Usage:
    from bregman_kl_triton import bregman_kl_div

    # Forward KL D_KL(p_ref || q_policy) for logit distillation / SFT:
    loss = bregman_kl_div(ref_logits, policy_logits, reverse=False, reduction="mean")

    # Reverse KL D_KL(q_policy || p_ref) for RLHF / GRPO:
    loss = bregman_kl_div(ref_logits, policy_logits, reverse=True, reduction="mean")
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_bregman_kl_fwd_bwd_kernel(
    ZP_ptr,
    ZQ_ptr,
    OUT_KL_ptr,
    DZQ_ptr,
    stride_zp_row,
    stride_zq_row,
    stride_dzq_row,
    V,
    THRESH: tl.constexpr,
    IS_REVERSE: tl.constexpr,
    WRITE_GRAD: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Fused 3-pass tiled SRAM row kernel (zero (N, V) FP32 HBM temporaries):
      Pass 1: Streaming online FP32 logsumexp(z_p) and logsumexp(z_q).
      Pass 2: Overflow-free, strictly nonnegative 6th-order Bregman-Taylor KL in FP32.
      Pass 3 (when WRITE_GRAD=True): Cancellation-free analytic logit gradient dz_q -> HBM.
    """
    row = tl.program_id(0)
    zp_row = ZP_ptr + row * stride_zp_row
    zq_row = ZQ_ptr + row * stride_zq_row

    m_p = -float("inf")
    s_p = 0.0
    m_q = -float("inf")
    s_q = 0.0

    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        zp = tl.load(zp_row + cols, mask=mask, other=-float("inf")).to(tl.float32)
        zq = tl.load(zq_row + cols, mask=mask, other=-float("inf")).to(tl.float32)

        blk_mp = tl.max(zp, axis=0)
        new_mp = tl.maximum(m_p, blk_mp)
        s_p = s_p * tl.exp(m_p - new_mp) + tl.sum(tl.where(mask, tl.exp(zp - new_mp), 0.0), axis=0)
        m_p = new_mp

        blk_mq = tl.max(zq, axis=0)
        new_mq = tl.maximum(m_q, blk_mq)
        s_q = s_q * tl.exp(m_q - new_mq) + tl.sum(tl.where(mask, tl.exp(zq - new_mq), 0.0), axis=0)
        m_q = new_mq

    lse_p = m_p + tl.log(s_p)
    lse_q = m_q + tl.log(s_q)

    kl_acc = 0.0
    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        zp = tl.load(zp_row + cols, mask=mask, other=0.0).to(tl.float32)
        zq = tl.load(zq_row + cols, mask=mask, other=0.0).to(tl.float32)

        lp = zp - lse_p
        lq = zq - lse_q
        p = tl.exp(lp)
        q = tl.exp(lq)

        if IS_REVERSE:
            d = lp - lq
            w = q
            other_prob = p
        else:
            d = lq - lp
            w = p
            other_prob = q

        taylor = d * d * (
            0.5
            + d
            * (
                0.1666666716337204
                + d
                * (
                    0.0416666679084301
                    + d * (0.008333333767950535 + d * 0.0013888889225199819)
                )
            )
        )
        exact_term = tl.maximum(other_prob - w - w * d, 0.0)
        small = tl.abs(d) < THRESH
        term = tl.where(small, w * tl.maximum(taylor, 0.0), exact_term)
        kl_acc += tl.sum(tl.where(mask, term, 0.0), axis=0)

    kl_val = tl.maximum(kl_acc, 0.0)
    tl.store(OUT_KL_ptr + row, kl_val)

    if WRITE_GRAD:
        dzq_row = DZQ_ptr + row * stride_dzq_row
        for off in range(0, V, BLOCK_V):
            cols = off + tl.arange(0, BLOCK_V)
            mask = cols < V
            zp = tl.load(zp_row + cols, mask=mask, other=0.0).to(tl.float32)
            zq = tl.load(zq_row + cols, mask=mask, other=0.0).to(tl.float32)

            lp = zp - lse_p
            lq = zq - lse_q
            p = tl.exp(lp)
            q = tl.exp(lq)

            if IS_REVERSE:
                g = q * ((lq - lp) - kl_val)
            else:
                d = lq - lp
                expm1_t = d * (
                    1.0
                    + d
                    * (
                        0.5
                        + d
                        * (
                            0.1666666716337204
                            + d * (0.0416666679084301 + d * 0.008333333767950535)
                        )
                    )
                )
                small = tl.abs(d) < THRESH
                g = tl.where(small, p * expm1_t, q - p)

            tl.store(dzq_row + cols, g, mask=mask)


class _FusedBregmanKLFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_p: torch.Tensor, z_q: torch.Tensor, reverse: bool = False, thresh: float = 0.05):
        orig_shape = z_q.shape
        V = orig_shape[-1]
        z_p_2d = z_p.reshape(-1, V).contiguous()
        z_q_2d = z_q.reshape(-1, V).contiguous()
        N = z_p_2d.shape[0]

        out_kl = torch.empty((N,), dtype=torch.float32, device=z_p.device)
        need_grad = z_q.requires_grad
        dz_q = torch.empty_like(z_q_2d) if need_grad else torch.empty((1,), dtype=z_q_2d.dtype, device=z_p.device)

        block_v = 4096 if V >= 4096 else triton.next_power_of_2(max(V, 128))
        _fused_bregman_kl_fwd_bwd_kernel[(N,)](
            z_p_2d,
            z_q_2d,
            out_kl,
            dz_q,
            z_p_2d.stride(0),
            z_q_2d.stride(0),
            dz_q.stride(0) if need_grad else 0,
            V,
            THRESH=thresh,
            IS_REVERSE=reverse,
            WRITE_GRAD=need_grad,
            BLOCK_V=block_v,
            num_warps=8 if block_v >= 2048 else 4,
        )
        ctx.save_for_backward(dz_q)
        ctx.need_grad = need_grad
        ctx.orig_shape = orig_shape
        return out_kl.reshape(orig_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (dz_q,) = ctx.saved_tensors
        if not ctx.need_grad:
            return None, None, None, None
        g = dz_q * grad_out.reshape(-1, 1).to(dz_q.dtype)
        return None, g.reshape(ctx.orig_shape), None, None


def bregman_kl_div(
    z_ref: torch.Tensor,
    z_policy: torch.Tensor,
    reverse: bool = False,
    reduction: str = "mean",
    thresh: float = 0.05,
) -> torch.Tensor:
    """
    Cancellation-free, overflow-free fused Triton KL divergence from raw logits.

    Args:
        z_ref: Reference/teacher logits of shape (*, V) on CUDA (BF16/FP16/FP32).
        z_policy: Policy/student logits of shape (*, V) on CUDA (supports autograd).
        reverse: If False, computes forward KL D_KL(p_ref || q_policy).
                 If True, computes reverse KL D_KL(q_policy || p_ref) for RLHF/GRPO.
        reduction: 'mean' | 'sum' | 'none'.
        thresh: Taylor switching threshold tau (default 0.05; Theorem 1 error <= 1.33e-10).
    """
    kl = _FusedBregmanKLFunc.apply(z_ref, z_policy, reverse, thresh)
    if reduction == "none":
        return kl
    if reduction == "sum":
        return kl.sum()
    if reduction == "mean":
        return kl.mean()
    raise ValueError(f"Unsupported reduction: {reduction!r}")
