"""MuJoCo Playground 환경(예: Unitree G1 험지 보행)용 PPO 트레이너.

brax PPOTrainer와 다른 점:
- 관측이 dict ({state, privileged_state})인 비대칭 actor-critic → policy/value obs 키 분리
- 험지 강건성을 위한 도메인 랜덤화(registry.get_domain_randomizer) 적용
- playground 전용 brax 래퍼(wrap_for_brax_training) 사용
"""

import time
from functools import partial

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_mod
from omegaconf import DictConfig

from src.algorithms.base import BaseTrainer
from src.config import parse_sizes


class PlaygroundPPOTrainer(BaseTrainer):
    def train(self, env, cfg: DictConfig):
        from mujoco_playground import registry, wrapper

        c = cfg.ppo
        # gfx1201 ROCm에서는 vmap된 모델 물리 커널이 죽어 기본 비활성화 (config 참고).
        randomization_fn = (
            registry.get_domain_randomizer(cfg.env) if c.get("domain_randomization", False) else None
        )

        # 체크포인트 이어받기: 저장된 (normalizer, policy, value) 튜플을 restore_params로 전달.
        restore_params = None
        restore_path = cfg.get("restore_path", "")
        if restore_path:
            from src.algorithms.base import load_policy

            restore_params, _meta = load_policy(restore_path)
            restore_params = jax.tree.map(jax.numpy.asarray, restore_params)
            print(f"체크포인트 이어받기: {restore_path}", flush=True)

        # 네트워크 메타(체크포인트/렌더 복원용) + 실행 폴더/TensorBoard 설정
        from src.runlog import log_metrics, make_checkpoint_fn, setup_run

        net_name = c.get("network", "mlp")
        net_meta = {
            "network": net_name,
            "policy_obs_key": c.policy_obs_key,
            "value_obs_key": c.value_obs_key,
            "simba_policy_blocks": c.simba_policy_blocks,
            "simba_policy_hidden": c.simba_policy_hidden,
            "simba_value_blocks": c.simba_value_blocks,
            "simba_value_hidden": c.simba_value_hidden,
        }
        run_dir, writer = setup_run(cfg, f"{cfg.env}_{net_name}")

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
            log_metrics(writer, int(step), dict(metrics),
                        {"perf/sps": instant_sps, "perf/avg_sps": avg_sps})
            rew = metrics.get("eval/episode_reward", float("nan"))
            print(
                f"  step {step:>11,}/{total:,} "
                f"({step / total * 100:5.1f}%) | "
                f"eval_reward {rew:+9.2f} | "
                f"{instant_sps:8.0f} sps (avg {avg_sps:7.0f}) | {elapsed / 60:5.1f}분",
                flush=True,
            )

        if c.get("network", "mlp") == "simba":
            from src.algorithms.simba import make_simba_ppo_networks

            network_factory = partial(
                make_simba_ppo_networks,
                policy_obs_key=c.policy_obs_key,
                value_obs_key=c.value_obs_key,
                policy_num_blocks=c.simba_policy_blocks,
                policy_hidden=c.simba_policy_hidden,
                value_num_blocks=c.simba_value_blocks,
                value_hidden=c.simba_value_hidden,
            )
            print(
                f"네트워크: SimBa (policy {c.simba_policy_blocks}블록x{c.simba_policy_hidden}, "
                f"value {c.simba_value_blocks}블록x{c.simba_value_hidden})",
                flush=True,
            )
        else:
            network_factory = partial(
                ppo_networks.make_ppo_networks,
                policy_hidden_layer_sizes=parse_sizes(c.policy_hidden_layer_sizes),
                value_hidden_layer_sizes=parse_sizes(c.value_hidden_layer_sizes),
                policy_obs_key=c.policy_obs_key,
                value_obs_key=c.value_obs_key,
            )

        checkpoint_fn = make_checkpoint_fn(
            cfg, run_dir, env=cfg.env,
            policy_sizes=parse_sizes(c.policy_hidden_layer_sizes),
            value_sizes=parse_sizes(c.value_hidden_layer_sizes),
            net=net_meta,
        )

        mk, params, _metrics = ppo_mod.train(
            environment=env,
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
            num_resets_per_eval=c.num_resets_per_eval,
            normalize_observations=c.normalize_observations,
            reward_scaling=c.reward_scaling,
            clipping_epsilon=c.clipping_epsilon,
            gae_lambda=c.gae_lambda,
            max_grad_norm=c.max_grad_norm,
            num_evals=c.num_evals,
            num_eval_envs=c.num_eval_envs,
            deterministic_eval=c.deterministic_eval,
            seed=cfg.seed,
            network_factory=network_factory,
            randomization_fn=randomization_fn,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            restore_params=restore_params,
            progress_fn=progress_fn,
            policy_params_fn=checkpoint_fn,
        )
        if writer is not None:
            writer.flush()
            writer.close()
        return env, mk, params

    def evaluate(self, model: tuple, env, cfg: DictConfig, n_episodes: int = 64) -> np.ndarray:
        """vmap된 env를 lax.scan으로 한 번에 롤아웃 (단일 env 파이썬 루프는 gfx1201에서 불안정).

        브락스 내부 평가기와 동일한 vmap+scan 경로라 학습 때 검증된 안정 구간을 사용한다.
        에피소드 보상은 첫 done까지만 누적한다.
        """
        import jax.numpy as jnp

        _env, mk, params = model
        n = min(int(n_episodes), int(cfg.ppo.num_eval_envs))
        inference = mk(params, deterministic=True)
        steps = int(cfg.ppo.episode_length)

        @jax.jit
        def rollout(rng):
            reset_rng = jax.random.split(rng, n)
            state = jax.vmap(env.reset)(reset_rng)

            def body(carry, _):
                state, rng, ret, alive = carry
                rng, act_rng = jax.random.split(rng)
                act_rng = jax.random.split(act_rng, n)
                action, _ = jax.vmap(inference)(state.obs, act_rng)
                state = jax.vmap(env.step)(state, action)
                ret = ret + state.reward * alive
                alive = alive * (1.0 - state.done)
                return (state, rng, ret, alive), None

            init = (state, rng, jnp.zeros(n), jnp.ones(n))
            (state, _, ret, _), _ = jax.lax.scan(body, init, None, length=steps)
            return ret

        rewards = rollout(jax.random.PRNGKey(cfg.seed + 999))
        return np.asarray(rewards)
