"""Shield tests, including a check against the project's PyTorch CASPER-k reference."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from ncasper.shields import apply_shield, casper_k, mask_topk, topk_mask


def _batch(n=2048, a=7, p_allowed=0.5, seed=0, scale=2.0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, a)) * scale
    allowed = rng.random((n, a)) < p_allowed
    allowed[np.arange(n), rng.integers(a, size=n)] = True
    return jnp.asarray(logits, jnp.float32), jnp.asarray(allowed)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("shield", ["mask", "casper"])
def test_support_is_allowed(shield, k):
    logits, S = _batch()
    p, z, T, q = apply_shield(logits, S, shield, k)
    assert bool(jnp.all(jnp.where(q > 0, S, True)))  # never sample a forbidden action
    np.testing.assert_allclose(np.asarray(q.sum(-1)), 1.0, atol=1e-5)
    # support size = min(k, |S|)
    np.testing.assert_array_equal(np.asarray(T.sum(-1)), np.minimum(k, np.asarray(S.sum(-1))))


@pytest.mark.parametrize("gamma", [0.0, 1.0])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_casper_margin_and_topk(k, gamma):
    logits, S = _batch(seed=1)
    p = jax.nn.softmax(logits, -1)
    z, T = casper_k(p, S, k, eps=1e-3, gamma=gamma)
    big = np.asarray(S.sum(-1) >= k)
    z, T = np.asarray(z), np.asarray(T)
    # top-k of z is exactly T (rows with |S| >= k)
    zk = np.asarray(topk_mask(jnp.asarray(z), k))
    assert (zk[big] == T[big]).all()
    np.testing.assert_allclose(z.sum(-1), 1.0, atol=1e-5)
    # compliant rows are untouched
    gap = np.where(T, z, 9).min(-1) - np.where(T, -9, z).max(-1)
    assert (gap[big] >= -1e-6).all()


def test_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    from tests.casper_k_reference import casper_k as ref  # copy of the project's torch code

    for k in (1, 2, 3):
        for gamma in (0.0, 1.0):
            logits, S = _batch(seed=2 + k)
            p = jax.nn.softmax(logits, -1)
            z, T = casper_k(p, S, k, eps=1e-3, gamma=gamma)
            zr, Tr = ref(torch.tensor(np.asarray(p)), torch.tensor(np.asarray(S)), k=k, eps0=1e-3, gamma=gamma)
            np.testing.assert_allclose(np.asarray(z), zr.numpy(), atol=1e-6)
            assert (np.asarray(T) == Tr.numpy()).all()


def test_gradients_flow_to_forbidden_logits():
    # forbidden action 2 is the raw argmax; restoration-style loss must push it down
    logits = jnp.array([1.0, 0.5, 3.0])
    S = jnp.array([True, True, False])

    def rest(lg):
        p, z, T, q = apply_shield(lg, S, "casper", 2)
        return 0.5 * jnp.sum((z - p) ** 2)

    g = jax.grad(rest)(logits)
    assert g[2] > 0  # gradient descent lowers the forbidden logit


def test_mask_topk_basic():
    p = jnp.array([[0.5, 0.3, 0.2]])
    z, T = mask_topk(p, jnp.array([[False, True, True]]), 2)
    np.testing.assert_allclose(np.asarray(z), [[0.0, 0.6, 0.4]], atol=1e-6)
    assert np.asarray(T).tolist() == [[False, True, True]]


def test_mask_pen_is_mask_at_execution_and_penalty_lowers_forbidden():
    logits, S = _batch(seed=7)
    a = apply_shield(logits, S, "mask", 2)
    b = apply_shield(logits, S, "mask_pen", 2)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    lg = jnp.array([1.0, 0.5, 3.0])
    allowed = jnp.array([True, True, False])
    g = jax.grad(lambda l: jnp.where(allowed, 0.0, jax.nn.softmax(l)).sum())(lg)
    assert g[2] > 0 and g[0] < 0 and g[1] < 0  # descent lowers the forbidden logit

