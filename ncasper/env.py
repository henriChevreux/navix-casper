"""NAVIX tasks with a per-step safety rule.

Rule: never move forward into a hazard. The allowed set S(s) is "every action except `forward`
when the cell in front of the agent holds lava or an obstacle", read directly from the state:
one categorical action and an observed condition, so no constraint compilation is needed.

Actions are MiniGrid's three movement actions: 0 turn left, 1 turn right, 2 forward.

Tasks used in the experiment:
  Navix-LavaGapS7-v0              cross a lava wall through its gap; stepping into lava ends the episode
  Navix-Dynamic-Obstacles-6x6-v0  reach the goal past balls that take a random step after every action
  Navix-DistShift1-v0             walk past a lava strip (lava is not terminal in NAVIX)

Dynamic-Obstacles: NAVIX versions disagree on what a collision does, so we fix MiniGrid's
semantics on every version: moving forward into a ball ends the episode with reward -1; a ball
never moves onto the agent (it stays put, not terminal); reaching the goal gives +1. The only way
to collide is then the forbidden action, exactly as with lava.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import navix as nx
from navix.actions import forward, rotate_ccw, rotate_cw
from navix.grid import translate

ACTIONS = (rotate_ccw, rotate_cw, forward)
NUM_ACTIONS = len(ACTIONS)
FORWARD = 2


def make_env(env_id: str):
    if "Dynamic-Obstacles" in env_id:
        return nx.make(env_id, action_set=ACTIONS, reward_fn=_obstacle_reward, termination_fn=_obstacle_termination)
    return nx.make(env_id, action_set=ACTIONS)


# ---------------------------------------------------------------- the rule
def _entity_ahead(state, name: str) -> jax.Array:
    """True if the cell the agent would enter with `forward` holds an entity of type `name`."""
    if name not in state.entities:
        return jnp.asarray(False)
    player = state.get_player(0)
    front = translate(player.position, player.direction)
    return jnp.any(jnp.all(state.entities[name].position == front[None], axis=-1))


def lava_ahead(state) -> jax.Array:
    return _entity_ahead(state, "lava")


def ball_ahead(state) -> jax.Array:
    return _entity_ahead(state, "ball")


def allowed_actions(state) -> jax.Array:
    """S(s) as a bool mask of shape (NUM_ACTIONS,)."""
    hazard = lava_ahead(state) | ball_ahead(state)
    return ~((jnp.arange(NUM_ACTIONS) == FORWARD) & hazard)


# ---------------------------------------------------------------- Dynamic-Obstacles semantics
def _bumped(prev_state, action) -> jax.Array:
    """The agent moved forward into a ball (checked before the balls move)."""
    return (action == FORWARD) & ball_ahead(prev_state)


def _obstacle_termination(prev_state, action, state) -> jax.Array:
    goal = nx.terminations.on_goal_reached(prev_state, action, state)
    return jnp.asarray(goal | _bumped(prev_state, action), dtype=jnp.bool_)


def _obstacle_reward(prev_state, action, state) -> jax.Array:
    goal = nx.rewards.on_goal_reached(prev_state, action, state)
    return jnp.asarray(goal - 1.0 * _bumped(prev_state, action), dtype=jnp.float32)


# ---------------------------------------------------------------- stepping
def _with_return(ts):
    info = dict(ts.info)
    info.setdefault("return", jnp.asarray(0.0))
    return ts.replace(info=info)


def reset(env, key):
    return _with_return(env.reset(key))


def step_autoreset(env, ts, action, key):
    """Step a live timestep; if the episode ends, reset in the same call.

    Returns (next_ts, reward, done, success, episode_return); next_ts is already the first
    timestep of a new episode when `done`.
    """
    nts = _with_return(env.step(ts, action))
    done = nts.is_done()
    next_ts = jax.tree.map(lambda a, b: jnp.where(done, a, b), reset(env, key), nts)
    return next_ts, nts.reward, done, done & (nts.reward > 0), nts.info["return"]
