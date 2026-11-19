"""NAVIX environments with the lava / obstacle rule.

Rule: never move forward into a hazard. The allowed set S(s) is "every action
except `forward` when the cell in front of the agent holds lava or a moving
obstacle", read directly from the state. There is a single categorical action
and an observed condition, so no constraint compilation is needed.

Dynamic-Obstacles (`Navix-Dynamic-Obstacles-*`): balls take one random step
after every agent action, so the rule switches on and off from step to step.
NAVIX versions disagree on what a collision does (0.7.7: walking into a ball is
just blocked, and only a ball moving onto the agent ends the episode; 4.0: both
end it). We fix MiniGrid's semantics on every version:
  * moving forward into a ball ends the episode with reward -1;
  * a ball never moves onto the agent (it stays put), and that is not terminal;
  * reaching the goal gives +1.
So the only way to collide is the forbidden action, exactly as with terminal lava.
Only the 3-action set is supported there (pickup/toggle on a ball differ by version).

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


def is_obstacle_env(env_id: str) -> bool:
    return "Dynamic-Obstacles" in env_id


def make_env(env_id: str = "Navix-LavaGapS7-v0", full_action_set: bool = False):
    action_set = MINIGRID_ACTION_SET if full_action_set else (rotate_ccw, rotate_cw, forward)
    if not is_obstacle_env(env_id):
        return nx.make(env_id, action_set=action_set)
    if full_action_set:
        raise ValueError(f"{env_id}: only the 3-action set is supported (see ncasper/env.py)")
    return nx.make(env_id, action_set=action_set,
                   reward_fn=_obstacle_reward, termination_fn=_obstacle_termination)


def _entity_ahead(state, name: str) -> jax.Array:
    """True if the cell the agent would enter with `forward` holds an entity of type `name`."""
    if name not in state.entities:
        return jnp.asarray(False)
    player = state.get_player(0)
    front = translate(player.position, player.direction)
    pos = state.entities[name].position
    return jnp.any(jnp.all(pos == front[None], axis=-1))


def lava_ahead(state) -> jax.Array:
    """True if the cell the agent would enter with `forward` holds lava."""
    return _entity_ahead(state, "lava")


def ball_ahead(state) -> jax.Array:
    """True if the cell the agent would enter with `forward` holds a (moving) obstacle."""
    return _entity_ahead(state, "ball")


def hazard_ahead(state) -> jax.Array:
    return lava_ahead(state) | ball_ahead(state)


def allowed_actions(state, n_actions: int) -> jax.Array:
    """S(s) as a bool mask (n_actions,)."""
    forbid = (jnp.arange(n_actions) == FORWARD) & hazard_ahead(state)
    return ~forbid


def _bumped(prev_state, action) -> jax.Array:
    """The agent moved forward into a ball (checked before the balls move)."""
    return (action == FORWARD) & ball_ahead(prev_state)


def _obstacle_termination(prev_state, action, state) -> jax.Array:
    return jnp.asarray(nx.terminations.on_goal_reached(prev_state, action, state) | _bumped(prev_state, action),
                       dtype=jnp.bool_)


def _obstacle_reward(prev_state, action, state) -> jax.Array:
    goal = nx.rewards.on_goal_reached(prev_state, action, state)
    return jnp.asarray(goal - 1.0 * _bumped(prev_state, action), dtype=jnp.float32)


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
