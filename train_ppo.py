import time
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


ENV_ID = "Ant-v5"
N_ENVS = 8
TOTAL_TIMESTEPS = 2_000_000
MODEL_PATH = "ppo_ant.zip"
LOG_EVERY = 50


class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_count = 0
        self.recent_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.ep_count += 1
                ep_rew = info["episode"]["r"]
                ep_len = info["episode"]["l"]
                self.recent_rewards.append(ep_rew)
                if len(self.recent_rewards) > 100:
                    self.recent_rewards.pop(0)
                if self.ep_count % LOG_EVERY == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.num_timesteps / elapsed if elapsed > 0 else 0
                    avg100 = np.mean(self.recent_rewards)
                    remaining = (TOTAL_TIMESTEPS - self.num_timesteps) / fps if fps > 0 else 0
                    print(
                        f"  ep {self.ep_count:5d} | "
                        f"step {self.num_timesteps:>8d}/{TOTAL_TIMESTEPS} "
                        f"({self.num_timesteps / TOTAL_TIMESTEPS * 100:5.1f}%) | "
                        f"reward {ep_rew:+8.1f} | avg100 {avg100:+8.1f} | "
                        f"{fps:6.0f} fps | 남은시간 ~{remaining/60:4.1f}분"
                    )
        return True

    def _on_training_start(self) -> None:
        self.start_time = time.time()


def make_env(env_id, seed):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return _init


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== PPO 학습 시작 ({ENV_ID}) ===")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"Envs: {N_ENVS}  |  Timesteps: {TOTAL_TIMESTEPS:,}")
    print("-" * 60)

    env = DummyVecEnv([make_env(ENV_ID, 42 + i) for i in range(N_ENVS)])
    env = VecMonitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=0,
        device=device,
        seed=42,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=RewardLogger(), progress_bar=False)
    model.save(MODEL_PATH)
    env.close()
    print("-" * 60)
    print(f"모델 저장: {MODEL_PATH}")

    print("\n=== 학습된 정책 평가 ===")
    eval_env = gym.make(ENV_ID)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    eval_env.close()
    print(f"평균 보상: {mean_reward:+.2f} ± {std_reward:.2f} (10 에피소드)")

    print("\n=== 무작위 정책과 비교 ===")
    rand_env = gym.make(ENV_ID)
    rand_rewards = []
    for _ in range(10):
        obs, _ = rand_env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            obs, rew, term, trunc, _ = rand_env.step(rand_env.action_space.sample())
            ep_rew += rew
            done = term or trunc
        rand_rewards.append(ep_rew)
    rand_env.close()
    print(f"무작위 정책: {np.mean(rand_rewards):+.2f} ± {np.std(rand_rewards):.2f}")
    print(f"PPO 정책:    {mean_reward:+.2f} ± {std_reward:.2f}")


if __name__ == "__main__":
    main()
