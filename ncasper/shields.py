"""Shields on the policy's output distribution, for top-k sampling (JAX).

A shield maps the raw distribution p = softmax(logits) and the allowed set S (bool mask) to an
output distribution z and a top-k support T. The agent samples from q = z restricted to T,
renormalised. Every shield below guarantees supp q ⊆ S.

  mask      z = p * 1[S] / P_S,   T = the k largest entries of p inside S
  mask_pen  as mask; the training loss adds lambda * sum_{j not in S} p_j  (see ppo.py)
  casper    z = CASPER-k(p, S),   T = the k largest entries of p inside S;
            the training loss adds lambda_r / 2 * ||z - p||^2            (see ppo.py)

CASPER-k (closed-form surrogate; a line-by-line port of the project's `casper_k`):
  |S| >= k:  with m_T = min_T p, M_T = max_{j not in T} p and e = min(eps, m_T),
               L = {t in T : p_t <= M_T + e},   H = {j not in T : p_j >= m_T - e},
               u = (sum_{L u H} p + |H| e) / (|L| + |H|),
               z_L = u,  z_H = u - e,  every other entry unchanged.
             Then z_t >= z_j + e for all t in T, j not in T, so top_k(z) = T. If L is empty,
             p is already k-compliant and z = p.
  |S| <  k:  face projection: z = p on S plus the mass outside S spread uniformly over S, T = S.
  gamma > 0  (only used in the tests): gamma-tilted margins that keep a gamma-fraction of the
             differences inside T.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

SHIELDS = ("mask", "mask_pen", "casper")


def topk_mask(scores: jax.Array, k: int) -> jax.Array:
    """Boolean mask of the k largest entries along the last axis (any batch shape)."""
    flat = scores.reshape(-1, scores.shape[-1])
    _, idx = jax.lax.top_k(flat, k)
    mask = jax.vmap(lambda i: jnp.zeros(flat.shape[-1], bool).at[i].set(True))(idx)
    return mask.reshape(scores.shape)


def mask_topk(p: jax.Array, allowed: jax.Array, k: int):
    """Zero the forbidden actions, renormalise, take the top k inside S (all of S if |S| < k)."""
    z = jnp.where(allowed, p, 0.0)
    z = z / z.sum(-1, keepdims=True)
    T = topk_mask(jnp.where(allowed, p, -1.0), k) & allowed
    return z, T


def casper_k(p: jax.Array, allowed: jax.Array, k: int, eps: float = 1e-3, gamma: float = 0.0):
    """CASPER-k projection. p: (..., A) distribution, allowed: (..., A) bool. Returns (z, T)."""
    n_allowed = allowed.sum(-1, keepdims=True)
    # |S| >= k
    T = topk_mask(jnp.where(allowed, p, -1.0), k)
    m_t0 = jnp.where(T, p, 2.0).min(-1, keepdims=True)
    tilt = jnp.where(T, gamma * (p - m_t0), 0.0)
    y = p - tilt
    m_t = jnp.where(T, y, 2.0).min(-1, keepdims=True)  # weakest member of T
    m_out = jnp.where(T, -1.0, y).max(-1, keepdims=True)  # strongest non-member
    e = jnp.minimum(jax.lax.stop_gradient(m_t), eps)
    L = T & (y <= m_out + e)
    H = (~T) & (y >= m_t - e)
    n_l, n_h = L.sum(-1, keepdims=True), H.sum(-1, keepdims=True)
    u = ((y * (L | H)).sum(-1, keepdims=True) + n_h * e) / jnp.maximum(n_l + n_h, 1)
    z_big = jnp.where(L, u, jnp.where(H, u - e, y))
    z_big = jnp.where(n_l > 0, z_big, y) + tilt
    # |S| < k: face projection
    forbidden_mass = jnp.where(allowed, 0.0, p).sum(-1, keepdims=True)
    z_small = jnp.where(allowed, p + forbidden_mass / jnp.maximum(n_allowed, 1), 0.0)
    big = n_allowed >= k
    return jnp.where(big, z_big, z_small), jnp.where(big, T, allowed)


def apply_shield(logits: jax.Array, allowed: jax.Array, shield: str, k: int, eps: float = 1e-3):
    """Returns (p, z, T, q): raw distribution, shielded distribution, top-k support, sampler."""
    p = jax.nn.softmax(logits, axis=-1)
    if shield in ("mask", "mask_pen"):
        z, T = mask_topk(p, allowed, k)
    elif shield == "casper":
        z, T = casper_k(p, allowed, k, eps)
    else:
        raise ValueError(f"unknown shield {shield!r}; choose from {SHIELDS}")
    q = jnp.where(T, jnp.maximum(z, 0.0), 0.0)
    q = q / jnp.maximum(q.sum(-1, keepdims=True), 1e-12)
    return p, z, T, q
