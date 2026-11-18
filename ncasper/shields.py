"""Shields on the policy's output distribution, for top-k sampling (JAX).

Every shield maps the raw distribution p = softmax(logits) and the allowed set
S (bool mask) to an output distribution z and a top-k support T; the agent then
samples from q_k(z) = z restricted to T and renormalised.

  none    z = p,                         T = top_k(p)            (may contain forbidden actions)
  mask    z = p * 1[S] / P_S,            T = top_k(z) within S   (compile-and-mask baseline)
  casper  z = CASPER-k(p, S),            T = top_k of p within S (CASPER-k, this project)

CASPER-k (surrogate, Thm 2 / Prop 3 of the project's sec3 notes; a line-by-line
port of `casper_k` in casper_k/casperk_learning_study.py):
  |S| >= k:  T = the k largest entries of p inside S. With m_T = min_T p and
             M_T = max_{j not in T} p, eps' = min(eps, m_T):
               L = {t in T : p_t <= M_T + eps'},  H = {j not in T : p_j >= m_T - eps'}
               u = (sum_{L u H} p + |H| eps') / (|L| + |H|)
               z_L = u,  z_H = u - eps',  everything else unchanged.
             Then z_t >= z_j + eps' for t in T, j not in T, so top_k(z) = T.
             If L is empty, p is already k-compliant and z = p.
  |S| <  k:  face projection, z = p on S plus the mass outside S spread
             uniformly over S, T = S.
  gamma > 0: gamma-tilted margins (NOTES.md), which keep a gamma-fraction of the
             differences inside T. gamma = 0 is plain CASPER-k; k = 1 is CASPER argmax.

Guarantee for mask and casper: supp q_k(z) is a subset of S, for every eps > 0.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

SHIELDS = ("none", "mask", "casper")


def topk_mask(scores: jax.Array, k: int) -> jax.Array:
    """Boolean mask of the k largest entries along the last axis (any batch shape)."""
    flat = scores.reshape(-1, scores.shape[-1])
    _, idx = jax.lax.top_k(flat, k)
    m = jax.vmap(lambda i: jnp.zeros(flat.shape[-1], bool).at[i].set(True))(idx)
    return m.reshape(scores.shape)


def casper_k(p: jax.Array, allowed: jax.Array, k: int, eps: float = 1e-3, gamma: float = 0.0):
    """p: (..., A) distribution, allowed: (..., A) bool. Returns (z, T)."""
    n_s = allowed.sum(-1, keepdims=True)
    big = n_s >= k
    # ---- |S| >= k
    T = topk_mask(jnp.where(allowed, p, -1.0), k)
    m_t0 = jnp.where(T, p, 2.0).min(-1, keepdims=True)
    off = jnp.where(T, gamma * (p - m_t0), 0.0)
    yp = p - off
    m_t = jnp.where(T, yp, 2.0).min(-1, keepdims=True)
    big_m = jnp.where(T, -1.0, yp).max(-1, keepdims=True)
    e = jnp.minimum(jax.lax.stop_gradient(m_t), eps)
    L = T & (yp <= big_m + e)
    H = (~T) & (yp >= m_t - e)
    B = L | H
    n_l, n_h = L.sum(-1, keepdims=True), H.sum(-1, keepdims=True)
    u = ((yp * B).sum(-1, keepdims=True) + n_h * e) / jnp.maximum(n_l + n_h, 1)
    z_big = jnp.where(L, u, jnp.where(H, u - e, yp))
    z_big = jnp.where(n_l > 0, z_big, yp) + off
    # ---- |S| < k: face projection
    out_mass = jnp.where(allowed, 0.0, p).sum(-1, keepdims=True)
    z_small = jnp.where(allowed, p + out_mass / jnp.maximum(n_s, 1), 0.0)
    z = jnp.where(big, z_big, z_small)
    return z, jnp.where(big, T, allowed)


def mask_topk(p: jax.Array, allowed: jax.Array, k: int):
    """Zero the forbidden actions, renormalise, take the top k inside S (all of S if |S| < k)."""
    z = jnp.where(allowed, p, 0.0)
    z = z / z.sum(-1, keepdims=True)
    T = topk_mask(jnp.where(allowed, p, -1.0), k) & allowed
    return z, T


def apply_shield(logits: jax.Array, allowed: jax.Array, shield: str, k: int,
                 eps: float = 1e-3, gamma: float = 0.0):
    """Returns (p, z, T, q) where q = q_k(z) is the sampling distribution."""
    p = jax.nn.softmax(logits, axis=-1)
    if shield == "none":
        z, T = p, topk_mask(p, k)
    elif shield == "mask":
        z, T = mask_topk(p, allowed, k)
    elif shield == "casper":
        z, T = casper_k(p, allowed, k, eps, gamma)
    else:
        raise ValueError(f"unknown shield {shield!r}; choose from {SHIELDS}")
    q = jnp.where(T, jnp.maximum(z, 0.0), 0.0)
    q = q / jnp.maximum(q.sum(-1, keepdims=True), 1e-12)
    return p, z, T, q
