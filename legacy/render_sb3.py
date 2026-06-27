"""학습된 정책의 rollout을 비디오(mp4/gif)로 저장.

렌더링은 headless EGL 오프스크린(MUJOCO_GL=egl) 사용. 디스플레이 불필요.

사용 예:
    .venv/bin/python render_policy.py                     # ppo_ant.zip → rollout.mp4
    .venv/bin/python render_policy.py --steps 800 --gif   # 더 길게, gif로
"""

import argparse
import os

# EGL 백엔드는 gym/mujoco 렌더러 초기화 전에 지정해야 함 (osmesa 고장, glfw는 디스플레이 필요).
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

MODEL_PATH = "ppo_ant.zip"
ENV_ID = "Ant-v5"
FPS = 30


def render_episode(model_path: str, env_id: str, steps: int, deterministic: bool, seed: int):
    model = PPO.load(model_path)
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        frames.append(np.asarray(env.render()))
        if terminated or truncated:
            break
    env.close()
    return frames, total_reward


def main():
    p = argparse.ArgumentParser(description="정책 rollout 비디오 저장 (EGL headless)")
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--env", default=ENV_ID)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", action="store_true", help="mp4 대신 gif 저장")
    p.add_argument("--out", default=None)
    p.add_argument("--stochastic", action="store_true", help="deterministic预测 대신 stochastic")
    args = p.parse_args()

    out = args.out or ("rollout.gif" if args.gif else "rollout.mp4")

    frames, total_reward = render_episode(
        args.model, args.env, args.steps, deterministic=not args.stochastic, seed=args.seed
    )

    if args.gif:
        imageio.mimwrite(out, frames, fps=FPS)
    else:
        imageio.mimwrite(out, frames, fps=FPS, codec="libx264", quality=8)
    print(f"저장: {out} | {len(frames)} 프레임 | 누적 보상 {total_reward:+.1f}")


if __name__ == "__main__":
    main()
