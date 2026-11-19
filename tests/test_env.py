import jax
import jax.numpy as jnp

from ncasper.env import FORWARD, allowed_actions, lava_ahead, make_env, reset, step_autoreset


def test_lava_ahead_matches_forward_move():
    env = make_env("Navix-LavaGapS7-v0")
    # find a reset where the agent starts facing lava (lava column right next to it)
    for seed in range(50):
        ts = reset(env, jax.random.PRNGKey(seed))
        if bool(lava_ahead(ts.state)):
            break
    else:
        raise AssertionError("no start facing lava in 50 seeds")
    S = allowed_actions(ts.state, 3)
    assert not bool(S[FORWARD]) and bool(S[0]) and bool(S[1])
    # stepping forward into lava terminates with no reward
    nts = env.step(ts, jnp.asarray(FORWARD))
    assert int(nts.step_type) == 2 and float(nts.reward) == 0.0


def test_autoreset_returns_fresh_episode():
    env = make_env("Navix-LavaGapS7-v0")
    for seed in range(50):
        ts = reset(env, jax.random.PRNGKey(seed))
        if bool(lava_ahead(ts.state)):
            break
    nts, reward, done, success, ep_ret = step_autoreset(env, ts, jnp.asarray(FORWARD), jax.random.PRNGKey(1))
    assert bool(done) and not bool(success)
    assert int(nts.t) == 0  # already the first timestep of the next episode


def test_full_action_set():
    env = make_env("Navix-LavaGapS7-v0", full_action_set=True)
    assert int(env.action_space.maximum) + 1 == 7


# ---- Dynamic-Obstacles: moving balls, collision only via the forbidden action ----
DO = "Navix-Dynamic-Obstacles-6x6-v0"


def _rollout(env, policy, key, n=400):
    """Run n steps with autoreset; returns per-step (ball_ahead, allowed, action, reward, done)."""
    from ncasper.env import ball_ahead

    def body(carry, k):
        ts = carry
        k1, k2 = jax.random.split(k)
        S = allowed_actions(ts.state, 3)
        a = policy(k1, S)
        nts, r, d, _, _ = step_autoreset(env, ts, a, k2)
        return nts, (ball_ahead(ts.state), S, a, r, d)

    ts = reset(env, key)
    _, out = jax.lax.scan(body, ts, jax.random.split(key, n))
    return out


def test_obstacle_rule_and_collision():
    env = make_env(DO)
    forward_when_possible = lambda k, S: jnp.where(jax.random.uniform(k) < 0.7, FORWARD, jax.random.randint(k, (), 0, 2))
    ahead, S, a, r, d = _rollout(env, forward_when_possible, jax.random.PRNGKey(0), 2000)
    assert bool(ahead.any()), "never faced an obstacle"
    # S forbids exactly forward when a ball is ahead
    assert bool(jnp.all(S[:, FORWARD] == ~ahead)) and bool(jnp.all(S[:, :2]))
    bump = ahead & (a == FORWARD)
    assert bool(bump.any())
    # every bump ends the episode with -1, and -1 happens only on a bump
    assert bool(jnp.all(d[bump])) and bool(jnp.all(r[bump] == -1.0))
    assert bool(jnp.all((r == -1.0) == bump))


def test_obstacle_shielded_policy_never_collides():
    env = make_env(DO)
    # uniform over the allowed set: the balls still move, but can't move onto the agent
    def shielded(k, S):
        return jax.random.choice(k, 3, p=S / S.sum())
    ahead, S, a, r, d = _rollout(env, shielded, jax.random.PRNGKey(1), 3000)
    assert bool(ahead.any())
    assert not bool((r < 0).any())
    assert not bool((ahead & (a == FORWARD)).any())


def test_obstacle_env_rejects_full_action_set():
    import pytest
    with pytest.raises(ValueError):
        make_env(DO, full_action_set=True)
