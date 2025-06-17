import jax 
import jax.numpy as jnp

import jax.experimental.pallas as pl 
import jax.experimental.pallas.triton as plgpu

import math
from functools import partial

"""
Dimension Key:

L: Sequence Length (Q sequence length)
M: Memory Length (KV sequence length)
H: Number of Heads
K: Head Dimension
P: Head Dimension Padded
R: Dimension of Q blocks 
C: Dimension of KV blocks 
"""


def block_mask_mod(start_q, start_k, R, C, mask_fn):
    q_indices_R = start_q + jnp.arange(R)[:, None]
    k_indices_C = start_k + jnp.arange(C)[None, :]
    return mask_fn(q_indices_R, k_indices_C)

def block_scores_mod(QK_RC, start_q, start_k, R, C, scores_fn):
    q_indices_R = start_q + jnp.arange(R)[:, None]
    k_indices_C = start_k + jnp.arange(C)[None, :]
    return scores_fn(QK_RC, q_indices_R, k_indices_C)

def sparsify_block_mask(mask_fn, L, M, R, C):
    n_q_blocks = pl.cdiv(L, R)
    n_k_blocks = pl.cdiv(M, C)
    
    def check_block(i, j):
      block_mask = mask_fn(i, j)
      all_true = jnp.all(block_mask)
      any_true = jnp.any(block_mask)
      return jnp.where(all_true, 2, jnp.where(any_true, 1, 0)) 
    
    q_block_starts = jnp.arange(n_q_blocks) * R
    k_block_starts = jnp.arange(n_k_blocks) * C

    check_block_vmap = jax.vmap(jax.vmap(check_block, in_axes=(None, 0)), in_axes=(0, None))
    block_types = check_block_vmap(q_block_starts, k_block_starts).astype(jnp.int8)

    return block_types

def flash_attention_kernel(
        q_ref,
        k_ref,
        v_ref,
        block_mask_ref,
        out_ref,
        scale,
        R,
        C,
        K,
        scores_fn=None, 
        mask_fn=None
):
    M = k_ref.shape[0]
    start_q = pl.program_id(0)
    # head_idx = pl.program_id(1)
    P = q_ref.shape[-1]

    m_R = jnp.zeros((R,), dtype=jnp.float32) - float("inf") # Max buffer
    l_R = jnp.zeros((R,), dtype=jnp.float32) # Norm buffer

    o_RP = jnp.zeros((R, P), dtype=jnp.float32) # Output buffer

    head_mask = (jnp.arange(P) < K)[None, :]
    q_RP = pl.load(q_ref, (slice(None), slice(None)), mask=head_mask, other=0.0)

    qk_scale = math.log2(math.e)
    if scale != 1.0:
        qk_scale *= scale

    def _compute(start_kv, m_prev_R, l_prev_R, o_prev_RP, should_mask):
        kv_slice = pl.dslice(start_kv * C, C)

        # Compute QK
        k_CP = pl.load(k_ref, (kv_slice, slice(None)), mask=head_mask, other=0.0)
        qk_RC = pl.dot(q_RP, k_CP.T)
        qk_RC *= qk_scale
        
        if scores_fn is not None:
          qk_RC = scores_fn(qk_RC, start_q, start_kv)

        if mask_fn is not None:
          qk_RC = jax.lax.cond(
              should_mask,
              lambda: jnp.where(mask_fn(start_q, start_kv), qk_RC, float("-inf")),
              lambda: qk_RC
          )

        # Update running max and norm
        m_curr_R = jnp.max(qk_RC, axis=-1)
        m_next_R = jnp.maximum(m_prev_R, m_curr_R)
        correction_R = jnp.exp2(m_prev_R - m_next_R)

        s_curr_RC = jnp.exp2((qk_RC - m_next_R[:, None]))

        l_next_R = l_prev_R * correction_R + s_curr_RC.sum(axis=-1)
        # Update output
        v_CP = pl.load(v_ref, (kv_slice, slice(None)), mask=head_mask, other=0.0)
        o_next_RP = o_prev_RP * correction_R[:, None] + pl.dot(s_curr_RC.astype(v_CP.dtype), v_CP)

        return m_next_R, l_next_R, o_next_RP

    def body(start_kv, carry):
        m_prev_R, l_prev_R, o_prev_RP = carry
        should_compute = block_mask_ref[start_q, start_kv] != 0
        should_mask = block_mask_ref[start_q, start_kv] == 1
        return jax.lax.cond(should_compute, lambda: _compute(start_kv, m_prev_R, l_prev_R, o_prev_RP, should_mask), lambda: carry)


    m_R, l_R, o_RP = jax.lax.fori_loop(0, pl.cdiv(M, C), body, (m_R, l_R, o_RP))

    l_R = jnp.where(l_R == 0.0, 1, l_R)
    o_RP /= l_R[:, None]
    pl.store(out_ref, (slice(None), slice(o_RP.shape[-1])), o_RP.astype(out_ref.dtype), mask=head_mask)


@partial(jax.jit, static_argnames=['scale', 'mask_mod', 'scores_mod', 'block_size_q', 'block_size_kv', 'num_warps', 'num_stages'])
def flash_attention(
        q_LHK,
        k_MHK,
        v_MHK,
        scale=1.0,
        mask_mod=None,
        scores_mod=None,
        block_size_q=16,
        block_size_kv=16,
        num_warps=None,
        num_stages=2
):
    L, H, K = q_LHK.shape
    M = k_MHK.shape[0]
    R = min(L, block_size_q)
    C = min(M, block_size_kv)
    P = pl.next_power_of_2(K)

    QB = pl.cdiv(L, R)
    KB = pl.cdiv(M, C)

    grid_ = (QB, H) # (Q Blocks, Heads)

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if H < 64 else 8

    _mask_mod = mask_mod
    if _mask_mod is not None:
        _mask_mod = partial(block_mask_mod, R=R, C=C, mask_fn=mask_mod)

    _scores_mod = scores_mod
    if _scores_mod is not None:
        _scores_mod = partial(block_scores_mod, R=R, C=C, scores_fn=scores_mod)
  
    kernel = partial(flash_attention_kernel, scale=scale, R=R, C=C, K=K, mask_fn=_mask_mod, scores_fn=_scores_mod)

    in_specs = [
        pl.BlockSpec((R, None, P), lambda i, j: (i, j, 0)),
        pl.BlockSpec((M, None, P), lambda _, j: (0, j, 0)),
        pl.BlockSpec((M, None, P), lambda _, j: (0, j, 0)),
        pl.BlockSpec((QB, KB), lambda i, j: (0, 0)),
    ]

    out_specs = [
        pl.BlockSpec((R, None, P), lambda i, j: (i, j, 0))]

    out_shape = [jax.ShapeDtypeStruct(q_LHK.shape, q_LHK.dtype)]

    block_mask = jnp.ones((pl.cdiv(L, R), pl.cdiv(M, C)), dtype=jnp.int8)
    if _mask_mod is not None:
      block_mask = sparsify_block_mask(_mask_mod, L, M, R, C)

    out_LHK = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=plgpu.TritonCompilerParams(
            num_warps=num_warps_,
            num_stages=num_stages,
        ),
    )(q_LHK, k_MHK, v_MHK, block_mask)
    return out_LHK[0]