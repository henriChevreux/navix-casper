"""Smoke test: a tiny PPO run for every shield on every task, and the shield's guarantee."""
import jax
import numpy as np
import pytest

from ncasper.ppo import Config, make_train

TASKS = ["Navix-LavaGapS7-v0", "Navix-Dynamic-Obstacles-6x6-v0", "Navix-DistShift1-v0"]


@pytest.mark.parametrize("env_id", TASKS)
@pytest.mark.parametrize("shield", ["mask", "mask_pen", "casper"])
def test_tiny_run(env_id, shield):
    cfg = Config(env_id=env_id, shield=shield, budget=2 * 16 * 16, num_envs=16, num_steps=16,
                 num_minibatches=2, num_epochs=1, eval_envs=8, eval_steps=32)
    train, n_updates = make_train(cfg)
    curves, final = jax.jit(train)(jax.random.PRNGKey(0))
    assert curves["success_rate"].shape == (n_updates,)
    for phase in ("eval_shield", "eval_raw"):
        assert set(final[phase]) >= {"success_rate", "raw_viol", "exec_viol", "n_rule_active"}
    # the shield never executes a forbidden action
    assert float(np.max(curves["exec_viol"])) == 0.0
    assert float(final["eval_shield"]["exec_viol"]) == 0.0
