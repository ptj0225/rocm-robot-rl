import time
from functools import partial

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_mod
from omegaconf import DictConfig

from src.algorithms.base import BaseTrainer
from src.config import parse_sizes


class PPOTrainer(BaseTrainer):
    def train(self, env, cfg: DictConfig):
        c = cfg.ppo
        total = c.num_timesteps
        start = [time.time()]
        last_time = [time.time()]
        last_step = [0]

        def progress_fn(step, metrics):
            now = time.time()
            dt = now - last_time[0]
            instant_sps = (step - last_step[0]) / dt if dt > 0 else 0.0
            elapsed = now - start[0]
            avg_sps = step / elapsed if elapsed > 0 else 0.0
            last_step[0] = step
            last_time[0] = now
            rew = metrics.get("eval/episode_reward", float("nan"))
            print(
                f"  step {step:>10,}/{total:,} "
                f"({step / total * 100:5.1f}%) | "
                f"eval_reward {rew:+9.1f} | "
                f"{instant_sps:8.0f} sps (avg {avg_sps:7.0f}) | {elapsed / 60:5.1f}분"
            )

        network_factory = partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=parse_sizes(c.policy_hidden_layer_sizes),
            value_hidden_layer_sizes=parse_sizes(c.value_hidden_layer_sizes),
        )

        eval_env = env
        mk, params, metrics = ppo_mod.train(
            environment=env,
            eval_env=eval_env,
            num_timesteps=c.num_timesteps,
            num_envs=c.num_envs,
            episode_length=c.episode_length,
            action_repeat=1,
            learning_rate=c.learning_rate,
            entropy_cost=c.entropy_cost,
            discounting=c.discounting,
            unroll_length=c.unroll_length,
            batch_size=c.batch_size,
            num_minibatches=c.num_minibatches,
            num_updates_per_batch=c.num_updates_per_batch,
            normalize_observations=c.normalize_observations,
            reward_scaling=c.reward_scaling,
            num_evals=c.num_evals,
            num_eval_envs=c.num_eval_envs,
            deterministic_eval=c.deterministic_eval,
            seed=cfg.seed,
            network_factory=network_factory,
            progress_fn=progress_fn,
        )
        return eval_env, mk, params

    def evaluate(self, model: tuple, env, cfg: DictConfig, n_episodes: int = 50) -> np.ndarray:
        _eval_env, mk, params = model
        jit_inference = jax.jit(mk(params, deterministic=True))
        rng = jax.random.PRNGKey(cfg.seed + 999)
        rewards = []
        for _ in range(n_episodes):
            rng, reset_rng = jax.random.split(rng)
            state = env.reset(reset_rng)
            ep_reward = 0.0
            done = False
            steps = 0
            max_steps = cfg.ppo.episode_length
            while not done and steps < max_steps:
                rng, act_rng = jax.random.split(rng)
                action, _ = jit_inference(state.obs, act_rng)
                state = env.step(state, action)
                ep_reward += float(np.asarray(state.reward).reshape(-1)[0])
                done = bool(np.asarray(state.done).reshape(-1)[0])
                steps += 1
            rewards.append(ep_reward)
        return np.array(rewards)
