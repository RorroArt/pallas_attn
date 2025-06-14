import jax 
import jax.numpy as jnp

import jax.experimental.pallas as pl 
import jax.experimental.pallas.triton as plgpu

import math
from functools import partial

"""
Dimension Key:

L: Sequence Length
H: Number of Heads
K: Head Dimension
P: Head Dimension Padded
R: Dimension of Q blocks 
C: Dimension of KV blocks 
"""


def flash_attention_kernel(
        q_ref,
        k_ref,
        v_ref,
        out_ref,
        scale,
        R,
        C,
        K
):
    M = k_ref.shape[0]
    start_q = pl.program_id(0)
    P = q_ref.shape[-1]

    m_R = jnp.zeros((R,), dtype=jnp.float32) - float("inf") # Max buffer
    l_R = jnp.zeros((R,), dtype=jnp.float32) # Norm buffer

    o_RP = jnp.zeros((R, P), dtype=jnp.float32) # Output buffer

    head_mask = (jnp.arange(P) < K)[None, :]
    q_RP = pl.load(q_ref, (slice(None), slice(None)), mask=head_mask, other=0.0)

    qk_scale = math.log2(math.e)
    if scale != 1.0:
        qk_scale *= scale

    def body(start_kv, carry):
        m_prev_R, l_prev_R, o_prev_RP = carry
        kv_slice = pl.dslice(start_kv * C, C)

        # Compute QK
        k_CP = pl.load(k_ref, (kv_slice, slice(None)), mask=head_mask, other=0.0)
        qk_RC = pl.dot(q_RP, k_CP.T)
        qk_RC *= qk_scale

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

    m_R, l_R, o_RP = jax.lax.fori_loop(0, pl.cdiv(M, C), body, (m_R, l_R, o_RP))

    o_RP /= l_R[:, None]
    pl.store(out_ref, (slice(None), slice(o_RP.shape[-1])), o_RP.astype(out_ref.dtype), mask=head_mask)


@partial(jax.jit, static_argnames=['scale', 'block_size_q', 'block_size_kv', 'grid', 'num_warps', 'num_stages'])
def flash_attention(
        q_LHK,
        k_MHK,
        v_MHK,
        scale=1.0,
        block_size_q=16,
        block_size_kv=16,
        grid=None,
        num_warps=None,
        num_stages=2
):
    L, H, K = q_LHK.shape
    M = k_MHK.shape[0]
    R = min(L, block_size_q)
    C = min(M, block_size_kv)
    P = pl.next_power_of_2(K)

    grid_ = (pl.cdiv(L, R), H) # (Q Blocks, Heads)

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if H < 64 else 8

    kernel = partial(flash_attention_kernel, scale=scale, R=R, C=C, K=K)

    in_specs = [
        pl.BlockSpec(
            (R, None, P),
            lambda i, j: (i, j, 0), # Grid handles the Q block's loop and heads.
        ),
        pl.BlockSpec(
            (M, None, P),
            lambda _, j: (0, j, 0), # Grid handles heads
        ),
        pl.BlockSpec(
            (M, None, P),
            lambda _, j: (0, j, 0) # Grid handles heads
        ),
    ]

    out_specs = [
        pl.BlockSpec((R, None, P), lambda i, j: (i, j, 0))]

    out_shape = [jax.ShapeDtypeStruct(q_LHK.shape, q_LHK.dtype)]

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
        # debug=True
    )(q_LHK, k_MHK, v_MHK)
    return out_LHK[0]