"""PPO with top-k sampling through a shield (PureJaxRL-style).

A whole run (rollouts, PPO updates, final evaluation) is one jitted function of a PRNG key, so many
seeds train in parallel with `jax.vmap`. The same seed gives the same initial network and the same
environment resets for every shield, so methods are paired seed by seed.

Policy: logits -> p = softmax -> shield -> (z, T) -> q = z restricted to the top-k set T. The
behaviour policy mixes in a little uniform exploration over the allowed actions,

    pi = (1 - explore_eps) * q + explore_eps * Uniform(S),

because top-k sampling alone never tries the action ranked k+1: with 3 actions and k = 2 an agent
could lock `forward` out for good. The mixture stays inside S, so the shield's guarantee holds.
PPO's ratio and entropy use pi; evaluation acts with q (pure top-k sampling).

Extra training losses, on the raw distribution p:
    mask_pen   penalty_lambda * sum_{j not in S} p_j          (forbidden probability mass)
    casper     casper_lambda / 2 * ||z - p||^2                 (CASPER-k restoration)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from .env import FORWARD, NUM_ACTIONS, allowed_actions, make_env, reset, step_autoreset
from .shields import apply_shield


@dataclass(frozen=True)
class Config:
    env_id: str = "Navix-LavaGapS7-v0"
    shield: str = "casper"  # mask | mask_pen | casper
    k: int = 2  # top-k sampling
    explore_eps: float = 0.05  # uniform exploration over S mixed into the behaviour policy
    casper_eps: float = 1e-3  # CASPER-k margin
    casper_lambda: float = 1.0  # casper: restoration weight
    penalty_lambda: float = 1.0  # mask_pen: penalty weight
    budget: int = 5_000_000  # env steps per seed
    num_envs: int = 64
    num_steps: int = 128
    num_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    eval_envs: int = 256
    eval_steps: int = 0  # 0 -> 2 x env.max_steps

    def to_dict(self):
        return asdict(self)


class ActorCritic(nn.Module):
    hidden: int = 128

    @nn.compact
    def __call__(self, obs):
        # symbolic observation (H, W, 3): one-hot entity tag (channel 0) and entity state
        # (channel 2, e.g. the player's direction); colour (channel 1) is dropped
        x = jnp.concatenate([jax.nn.one_hot(obs[..., 0], 11), jax.nn.one_hot(obs[..., 2], 4)], -1)
        x = x.reshape(x.shape[:-3] + (-1,))
        dense = lambda n, scale: nn.Dense(n, kernel_init=nn.initializers.orthogonal(scale))  # noqa: E731
        a = nn.tanh(dense(self.hidden, jnp.sqrt(2))(x))
        a = nn.tanh(dense(self.hidden, jnp.sqrt(2))(a))
        logits = dense(NUM_ACTIONS, 0.01)(a)
        c = nn.tanh(dense(self.hidden, jnp.sqrt(2))(x))
        c = nn.tanh(dense(self.hidden, jnp.sqrt(2))(c))
        return logits, dense(1, 1.0)(c)[..., 0]


def _log_prob(q, action):
    return jnp.log(jnp.maximum(jnp.take_along_axis(q, action[..., None], -1)[..., 0], 1e-8))


def _sample(key, q):
    return jax.random.categorical(key, jnp.where(q > 0, jnp.log(jnp.maximum(q, 1e-30)), -jnp.inf))


def make_train(cfg: Config):
    """Returns (train, num_updates); train(key) -> (curves, final_eval)."""
    env = make_env(cfg.env_id)
    net = ActorCritic(cfg.hidden)
    num_updates = cfg.budget // (cfg.num_envs * cfg.num_steps)
    eval_steps = cfg.eval_steps or 2 * env.max_steps
    v_allowed = jax.vmap(allowed_actions)
    v_reset = jax.vmap(lambda k: reset(env, k))
    v_step = jax.vmap(lambda ts, a, k: step_autoreset(env, ts, a, k))

    def shield(logits, allowed):
        return apply_shield(logits, allowed, cfg.shield, cfg.k, cfg.casper_eps)

    def behaviour(q, allowed):
        return (1.0 - cfg.explore_eps) * q + cfg.explore_eps * allowed / allowed.sum(-1, keepdims=True)

    def env_step(params, carry, mode):
        """One step of every env. mode: 'train' (behaviour policy), 'shield' (top-k sampling from q),
        'raw' (shield removed: greedy action of the raw distribution p)."""
        ts, key = carry
        key, k_act, k_env = jax.random.split(key, 3)
        allowed = v_allowed(ts.state)
        logits, value = net.apply(params, ts.observation)
        p, z, T, q = shield(logits, allowed)
        if mode == "train":
            q = behaviour(q, allowed)
        action = jnp.argmax(p, -1) if mode == "raw" else _sample(k_act, q)
        rule_active = ~allowed.all(-1)
        nts, reward, done, success, ep_ret = v_step(ts, action, jax.random.split(k_env, ts.t.shape[0]))
        info = dict(obs=ts.observation, allowed=allowed, action=action, logq=_log_prob(q, action),
                    value=value, reward=reward, done=done, success=success, ep_ret=ep_ret,
                    rule_active=rule_active,
                    exec_viol=rule_active & ~jnp.take_along_axis(allowed, action[:, None], -1)[:, 0],
                    raw_viol=rule_active & (jnp.argmax(p, -1) == FORWARD))
        return (nts, key), info

    def loss_fn(params, batch, adv, targets):
        logits, value = net.apply(params, batch["obs"])
        p, z, T, q = shield(logits, batch["allowed"])
        pi = behaviour(q, batch["allowed"])
        logp = _log_prob(pi, batch["action"])
        ratio = jnp.exp(logp - batch["logq"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        actor = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
        v_clip = batch["value"] + jnp.clip(value - batch["value"], -cfg.clip_eps, cfg.clip_eps)
        critic = 0.5 * jnp.maximum((value - targets) ** 2, (v_clip - targets) ** 2).mean()
        entropy = -jnp.where(pi > 0, pi * jnp.log(jnp.maximum(pi, 1e-30)), 0.0).sum(-1).mean()
        restoration = 0.5 * ((z - p) ** 2).sum(-1).mean()
        forbidden_mass = jnp.where(batch["allowed"], 0.0, p).sum(-1).mean()
        extra = {"casper": cfg.casper_lambda * restoration,
                 "mask_pen": cfg.penalty_lambda * forbidden_mass}.get(cfg.shield, 0.0)
        total = actor + cfg.vf_coef * critic - cfg.ent_coef * entropy + extra
        return total, dict(actor=actor, critic=critic, entropy=entropy, restoration=restoration,
                           forbidden_mass=forbidden_mass,
                           approx_kl=((ratio - 1) - (logp - batch["logq"])).mean())

    def rates(traj):
        n_ep, n_rule = traj["done"].sum(), traj["rule_active"].sum()
        return dict(episodes=n_ep, n_rule_active=n_rule,
                    success_rate=traj["success"].sum() / jnp.maximum(n_ep, 1),
                    raw_viol=traj["raw_viol"].sum() / jnp.maximum(n_rule, 1),
                    exec_viol=traj["exec_viol"].sum() / jnp.maximum(n_rule, 1))

    def train(rng):
        rng, k_init, k_reset = jax.random.split(rng, 3)
        ts = v_reset(jax.random.split(k_reset, cfg.num_envs))
        params = net.init(k_init, ts.observation)
        steps_per_update = cfg.num_minibatches * cfg.num_epochs
        lr = (lambda c: cfg.lr * (1.0 - (c // steps_per_update) / num_updates)) if cfg.anneal_lr else cfg.lr
        tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr, eps=1e-5))
        opt_state = tx.init(params)

        def update(state, _):
            params, opt_state, ts, key = state
            (ts, key), traj = jax.lax.scan(lambda c, _: env_step(params, c, "train"), (ts, key), None,
                                           cfg.num_steps)
            _, last_value = net.apply(params, ts.observation)

            def gae_step(carry, t):
                gae, next_value = carry
                delta = t["reward"] + cfg.gamma * next_value * (1 - t["done"]) - t["value"]
                gae = delta + cfg.gamma * cfg.gae_lambda * (1 - t["done"]) * gae
                return (gae, t["value"]), gae

            _, adv = jax.lax.scan(gae_step, (jnp.zeros_like(last_value), last_value),
                                  {k: traj[k] for k in ("reward", "done", "value")}, reverse=True)
            data = {k: traj[k] for k in ("obs", "allowed", "action", "logq", "value")}
            flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), (data, adv, adv + traj["value"]))
            n = cfg.num_envs * cfg.num_steps

            def epoch(carry, k):
                perm = jax.random.permutation(k, n)
                minibatches = jax.tree.map(lambda x: x[perm].reshape((cfg.num_minibatches, -1) + x.shape[1:]), flat)

                def minibatch_step(carry, mb):
                    params, opt_state = carry
                    (_, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, *mb)
                    updates, opt_state = tx.update(grads, opt_state, params)
                    return (optax.apply_updates(params, updates), opt_state), logs

                return jax.lax.scan(minibatch_step, carry, minibatches)

            key, k_ep = jax.random.split(key)
            (params, opt_state), logs = jax.lax.scan(epoch, (params, opt_state),
                                                     jax.random.split(k_ep, cfg.num_epochs))
            n_ep = traj["done"].sum()
            metrics = dict(rates(traj),
                           episode_return=(traj["ep_ret"] * traj["done"]).sum() / jnp.maximum(n_ep, 1),
                           rule_active_frac=traj["rule_active"].mean(),
                           **jax.tree.map(jnp.mean, logs))
            return (params, opt_state, ts, key), metrics

        (params, _, _, rng), curves = jax.lax.scan(update, (params, opt_state, ts, rng), None, num_updates)

        def evaluate(key, mode):
            k_reset, k_steps = jax.random.split(key)
            ts = v_reset(jax.random.split(k_reset, cfg.eval_envs))
            _, traj = jax.lax.scan(lambda c, _: env_step(params, c, mode), (ts, k_steps), None, eval_steps)
            return rates(traj)

        k_shield, k_raw = jax.random.split(rng)
        return curves, {"eval_shield": evaluate(k_shield, "shield"), "eval_raw": evaluate(k_raw, "raw")}

    return train, num_updates
