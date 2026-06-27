#!/usr/bin/env python
"""MJX + Brax RL 학습 — OmegaConf 기반 다중 환경·알고리즘.

사용 예:
    .venv-mjx/bin/python train.py --config configs/ant_ppo.yaml
    .venv-mjx/bin/python train.py env=halfcheetah ppo.num_envs=1024 ppo.precision=float32
    .venv-mjx/bin/python train.py env=hopper ppo.num_timesteps=30000000

지원 환경: ant, halfcheetah, hopper, walker2d, humanoid, fetch, reacher, swimmer ...
지원 알고리즘: ppo (확장 가능)
"""

import rocm_setup

rocm_setup.configure_rocm_runtime()

import jax  # noqa: E402

from src.algorithms import REGISTRY  # noqa: E402
from src.config import Config, load_config, parse_sizes  # noqa: E402
from src.envs import make_env  # noqa: E402

SOLVED_REWARD = 6000


def main():
    cfg = load_config(Config)
    jax.config.update("jax_default_matmul_precision", cfg.ppo.precision)

    # playground 백엔드는 비대칭 actor-critic + 도메인 랜덤화 트레이너 사용.
    algo_name = "ppo_playground" if cfg.backend == "playground" else "ppo"
    if algo_name not in REGISTRY:
        raise KeyError(f"알 수 없는 알고리즘: {algo_name}. 등록됨: {list(REGISTRY)}")

    print("=== MJX + Brax PPO 학습 시작 ===")
    print(f"Env: {cfg.env} (backend={cfg.backend})")
    print(f"Devices: {jax.devices()}")
    print(f"Compute(matmul): {cfg.ppo.precision}  |  Physics: float32")
    print(f"Policy net: {parse_sizes(cfg.ppo.policy_hidden_layer_sizes)}  |  Value net: {parse_sizes(cfg.ppo.value_hidden_layer_sizes)}")
    print(f"병렬 환경 수: {cfg.ppo.num_envs}  |  총 스텝: {cfg.ppo.num_timesteps:,}")
    print("-" * 64)

    trainer = REGISTRY[algo_name]()
    env = make_env(cfg)

    model = trainer.train(env, cfg)
    trainer.save(model, cfg, cfg.model_path)

    print("-" * 64)
    print("\n=== 학습된 정책 평가 (50 에피소드) ===")
    rewards = trainer.evaluate(model, env, cfg)
    print(f"평균 보상: {rewards.mean():+.1f} ± {rewards.std():.1f}")
    print(f"최고/최저: {rewards.max():+.1f} / {rewards.min():+.1f}")
    if cfg.env == "ant":
        print(f"Solved 기준({SOLVED_REWARD}) 달성 여부: {'성공' if rewards.mean() >= SOLVED_REWARD else '미달'}")


if __name__ == "__main__":
    main()
