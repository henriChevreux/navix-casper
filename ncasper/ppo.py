"""PPO with top-k sampling through a shield (PureJaxRL-style).

The whole run (rollouts, PPO updates, final evaluation) is one jitted function
of a PRNG key, so many seeds are trained in parallel with `jax.vmap`. Seeds are
paired across shields: the same seed gives the same initial network and the
same environment resets for every shield.

Policy: logits -> p = softmax -> shield -> (z, T) -> q = z restricted to the
top-k set T, renormalised. The behaviour policy mixes in a little uniform
exploration over the actions the shield allows (all actions for `none`):

    pi = (1 - explore_eps) * q + explore_eps * Uniform(S)

Without it, top-k sampling never samples the action ranked k+1. With 3 actions
and k = 2 an agent can then lock out `forward` for good, and PPO's ratio blows
up whenever the sampled action leaves the top-k. The mixture keeps support
inside S, so the shields' guarantee holds. PPO's ratio and entropy use pi.
Evaluation acts with q itself (pure top-k sampling). CASPER adds restoration
lambda/2 ||z - p||^2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from .env import FORWARD, allowed_actions, make_env, reset, step_autoreset
from .shields import apply_shield, topk_mask


@dataclass(frozen=True)
class Config:
    env_id: str = "Navix-LavaGapS7-v0"
    full_action_set: bool = False
    shield: str = "casper"  # none | mask | casper
    k: int = 2  # top-k sampling (k=1 = greedy)
    explore_eps: float = 0.05  # uniform exploration over S mixed into the behaviour policy
    casper_eps: float = 1e-3
    casper_gamma: float = 0.0  # 0 = plain CASPER-k; 1 = gamma-tilted margins
    casper_lambda: float = 1.0  # restoration weight (casper only)
    budget: int = 1_000_000  # env steps per seed
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
    n_actions: int
    hidden: int = 128

    @nn.compact
    def __call__(self, obs):
        # symbolic obs (H, W, 3): one-hot entity tag (channel 0, ids < 11) and
        # entity state (channel 2: the player's direction); colour is dropped
        tag = jax.nn.one_hot(obs[..., 0], 11)
        state = jax.nn.one_hot(obs[..., 2], 4)
        x = jnp.concatenate([tag, state], -1)
        x = x.reshape(x.shape[:-3] + (-1,))
        init = nn.initializers.orthogonal
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=init(jnp.sqrt(2)))(x))
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=init(jnp.sqrt(2)))(a))
        logits = nn.Dense(self.n_actions, kernel_init=init(0.01))(a)
        c = nn.tanh(nn.Dense(self.hidden, kernel_init=init(jnp.sqrt(2)))(x))
        c = nn.tanh(nn.Dense(self.hidden, kernel_init=init(jnp.sqrt(2)))(c))
        value = nn.Dense(1, kernel_init=init(1.0))(c)[..., 0]
        return logits, value


def _log(q, a):
    return jnp.log(jnp.maximum(jnp.take_along_axis(q, a[..., None], -1)[..., 0], 1e-8))


def _sample(key, q):
    return jax.random.categorical(key, jnp.where(q > 0, jnp.log(jnp.maximum(q, 1e-30)), -jnp.inf))


def make_train(cfg: Config):
    env = make_env(cfg.env_id, cfg.full_action_set)
    n_actions = int(env.action_space.maximum) + 1
    net = ActorCritic(n_actions, cfg.hidden)
    num_updates = cfg.budget // (cfg.num_envs * cfg.num_steps)
    eval_steps = cfg.eval_steps or 2 * env.max_steps
    v_allowed = jax.vmap(lambda s: allowed_actions(s, n_actions))
    v_reset = jax.vmap(lambda k: reset(env, k))
    v_step = jax.vmap(lambda ts, a, k: step_autoreset(env, ts, a, k))

    def shield(logits, allowed):
        return apply_shield(logits, allowed, cfg.shield, cfg.k, cfg.casper_eps, cfg.casper_gamma)

    def behaviour(q, allowed):
        support = allowed if cfg.shield != "none" else jnp.ones_like(allowed)
        uni = support / support.sum(-1, keepdims=True)
        return (1.0 - cfg.explore_eps) * q + cfg.explore_eps * uni

    def rollout_step(params, carry, mode):
        """mode: 'shield' (training/eval policy), 'raw_topk', 'raw_greedy' (layer removed)."""
        ts, key = carry
        key, k_act, k_env = jax.random.split(key, 3)
        allowed = v_allowed(ts.state)
        logits, value = net.apply(params, ts.observation)
        p, z, T, q = shield(logits, allowed)
        if mode == "raw_topk":
            q_exec = jnp.where(topk_mask(p, cfg.k), p, 0.0)
            action = _sample(k_act, q_exec / q_exec.sum(-1, keepdims=True))
        elif mode == "raw_greedy":
            action = jnp.argmax(p, -1)
        elif mode == "shield":  # evaluation: pure top-k sampling from q
            action = _sample(k_act, q)
        else:  # training: behaviour policy
            q = behaviour(q, allowed)
            action = _sample(k_act, q)
        constrained = ~allowed.all(-1)
        exec_viol = constrained & ~jnp.take_along_axis(allowed, action[:, None], -1)[:, 0]
        nts, reward, done, success, ep_ret = v_step(ts, action, jax.random.split(k_env, ts.t.shape[0]))
        info = dict(
            obs=ts.observation, allowed=allowed, action=action, logq=_log(q, action), value=value,
            reward=reward, done=done, success=success, ep_ret=ep_ret, constrained=constrained,
            exec_viol=exec_viol,
            raw_argmax_viol=constrained & (jnp.argmax(p, -1) == FORWARD),
            raw_topk_viol=constrained & jnp.any(topk_mask(p, cfg.k) & ~allowed, -1),
        )
        return (nts, key), info

    def loss_fn(params, batch, adv, targets):
        logits, value = net.apply(params, batch["obs"])
        p, z, T, q = shield(logits, batch["allowed"])
        q = behaviour(q, batch["allowed"])
        logq = _log(q, batch["action"])
        ratio = jnp.exp(logq - batch["logq"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        actor = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
        v_clip = batch["value"] + jnp.clip(value - batch["value"], -cfg.clip_eps, cfg.clip_eps)
        critic = 0.5 * jnp.maximum((value - targets) ** 2, (v_clip - targets) ** 2).mean()
        entropy = -jnp.where(q > 0, q * jnp.log(jnp.maximum(q, 1e-30)), 0.0).sum(-1).mean()
        restoration = 0.5 * ((z - p) ** 2).sum(-1).mean()
        rest_w = cfg.casper_lambda if cfg.shield == "casper" else 0.0
        total = actor + cfg.vf_coef * critic - cfg.ent_coef * entropy + rest_w * restoration
        return total, dict(actor=actor, critic=critic, entropy=entropy, restoration=restoration,
                           approx_kl=((ratio - 1) - (logq - batch["logq"])).mean())

    def train(rng):
        rng, k_init, k_reset = jax.random.split(rng, 3)
        ts = v_reset(jax.random.split(k_reset, cfg.num_envs))
        params = net.init(k_init, ts.observation)
        n_mb = cfg.num_minibatches * cfg.num_epochs
        lr = (lambda c: cfg.lr * (1.0 - (c // n_mb) / num_updates)) if cfg.anneal_lr else cfg.lr
        tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr, eps=1e-5))
        opt_state = tx.init(params)

        def update(state, _):
            params, opt_state, ts, key = state
            (ts, key), traj = jax.lax.scan(
                lambda c, _: rollout_step(params, c, "train"), (ts, key), None, cfg.num_steps)
            _, last_val = net.apply(params, ts.observation)

            def gae_step(carry, t):
                gae, next_val = carry
                delta = t["reward"] + cfg.gamma * next_val * (1 - t["done"]) - t["value"]
                gae = delta + cfg.gamma * cfg.gae_lambda * (1 - t["done"]) * gae
                return (gae, t["value"]), gae

            _, adv = jax.lax.scan(gae_step, (jnp.zeros_like(last_val), last_val),
                                  {k: traj[k] for k in ("reward", "done", "value")}, reverse=True)
            targets = adv + traj["value"]
            data = {k: traj[k] for k in ("obs", "allowed", "action", "logq", "value")}
            flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), (data, adv, targets))
            n = cfg.num_envs * cfg.num_steps

            def epoch(carry, k):
                params, opt_state = carry
                perm = jax.random.permutation(k, n)
                mbs = jax.tree.map(lambda x: x[perm].reshape((cfg.num_minibatches, -1) + x.shape[1:]), flat)

                def mb_step(carry, mb):
                    params, opt_state = carry
                    (_, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, *mb)
                    updates, opt_state = tx.update(grads, opt_state, params)
                    return (optax.apply_updates(params, updates), opt_state), logs

                return jax.lax.scan(mb_step, (params, opt_state), mbs)

            key, k_ep = jax.random.split(key)
            (params, opt_state), logs = jax.lax.scan(epoch, (params, opt_state),
                                                     jax.random.split(k_ep, cfg.num_epochs))
            n_ep = traj["done"].sum()
            n_con = traj["constrained"].sum()
            metrics = dict(
                episodes=n_ep,
                success_rate=traj["success"].sum() / jnp.maximum(n_ep, 1),
                lava_rate=traj["exec_viol"].sum() / jnp.maximum(n_ep, 1),
                episode_return=(traj["ep_ret"] * traj["done"]).sum() / jnp.maximum(n_ep, 1),
                constrained_frac=n_con / traj["done"].size,
                raw_argmax_viol=traj["raw_argmax_viol"].sum() / jnp.maximum(n_con, 1),
                raw_topk_viol=traj["raw_topk_viol"].sum() / jnp.maximum(n_con, 1),
                exec_viol=traj["exec_viol"].sum() / jnp.maximum(n_con, 1),
                n_constrained=n_con,
                **jax.tree.map(jnp.mean, logs),
            )
            return (params, opt_state, ts, key), metrics

        (params, _, _, rng), curves = jax.lax.scan(update, (params, opt_state, ts, rng), None, num_updates)

        def evaluate(key, mode):
            k_r, k_s = jax.random.split(key)
            ts = v_reset(jax.random.split(k_r, cfg.eval_envs))
            _, traj = jax.lax.scan(lambda c, _: rollout_step(params, c, mode), (ts, k_s), None, eval_steps)
            n_ep, n_con = traj["done"].sum(), traj["constrained"].sum()
            return dict(episodes=n_ep,
                        success_rate=traj["success"].sum() / jnp.maximum(n_ep, 1),
                        lava_rate=traj["exec_viol"].sum() / jnp.maximum(n_ep, 1),
                        raw_topk_viol=traj["raw_topk_viol"].sum() / jnp.maximum(n_con, 1),
                        raw_argmax_viol=traj["raw_argmax_viol"].sum() / jnp.maximum(n_con, 1))

        keys = jax.random.split(rng, 3)
        final = {f"eval_{m}": evaluate(k, m) for k, m in zip(keys, ("shield", "raw_topk", "raw_greedy"))}
        return curves, final

    return train, num_updates
