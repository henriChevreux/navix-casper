"""NAVIX environments with the lava rule.

Rule: never move forward into lava. The allowed set S(s) is "every action
except `forward` when the cell in front of the agent is lava", read directly
from the state. There is a single categorical action and an observed condition,
so no constraint compilation is needed.

Default action set: MiniGrid's three movement actions (0 turn left, 1 turn right,
2 forward). `full_action_set=True` uses all seven MiniGrid actions; `forward` is
index 2 in both.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import navix as nx
from navix.actions import MINIGRID_ACTION_SET, forward, rotate_ccw, rotate_cw
from navix.grid import translate

FORWARD = 2
LAVA_ENVS = (
    "Navix-LavaGapS5-v0", "Navix-LavaGapS6-v0", "Navix-LavaGapS7-v0",
    "Navix-LavaCrossingS9N1-v0", "Navix-LavaCrossingS9N2-v0", "Navix-LavaCrossingS9N3-v0",
    "Navix-LavaCrossingS11N5-v0", "Navix-DistShift1-v0", "Navix-DistShift2-v0",
)


def make_env(env_id: str = "Navix-LavaGapS7-v0", full_action_set: bool = False):
    action_set = MINIGRID_ACTION_SET if full_action_set else (rotate_ccw, rotate_cw, forward)
    return nx.make(env_id, action_set=action_set)


def lava_ahead(state) -> jax.Array:
    """True if the cell the agent would enter with `forward` holds lava."""
    player = state.get_player(0)
    front = translate(player.position, player.direction)
    if "lava" not in state.entities:
        return jnp.asarray(False)
    lava = state.entities["lava"].position
    return jnp.any(jnp.all(lava == front[None], axis=-1))


def allowed_actions(state, n_actions: int) -> jax.Array:
    """S(s) as a bool mask (n_actions,)."""
    forbid = (jnp.arange(n_actions) == FORWARD) & lava_ahead(state)
    return ~forbid


def _with_return(ts):
    info = dict(ts.info)
    info.setdefault("return", jnp.asarray(0.0))
    return ts.replace(info=info)


def reset(env, key):
    return _with_return(env.reset(key))


def step_autoreset(env, ts, action, key):
    """Step a live (never terminal) timestep; reset in the same call if it ends.

    Returns (next_ts, reward, done, success, episode_return). `next_ts` is
    already the first timestep of a new episode when `done`.
    """
    nts = _with_return(env.step(ts, action))
    done = nts.is_done()
    reward = nts.reward
    ep_return = nts.info["return"]
    fresh = reset(env, key)
    next_ts = jax.tree.map(lambda a, b: jnp.where(done, a, b), fresh, nts)
    return next_ts, reward, done, done & (reward > 0), ep_return
