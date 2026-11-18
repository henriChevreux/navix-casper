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
