import gymnasium as gym
import numpy as np


def main():
    env = gym.make("Ant-v5", render_mode="rgb_array")
    observation, info = env.reset()

    print("=== Action Space (로봇이 취할 수 있는 행동) ===")
    print(f"형태: {env.action_space.shape}")
    print(f"범위: {env.action_space.low} ~ {env.action_space.high}")
    print(f"샘플: {env.action_space.sample()}")

    print("\n=== Observation Space (로봇이 관측하는 상태) ===")
    print(f"형태: {env.observation_space.shape}")
    print(f"첫 관측: {observation}")

    print("\n=== 리워드 구조 (1스텝 실행) ===")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"실행한 행동: {action}")
    print(f"받은 보상: {reward}")
    print(f"종료 여부: terminated={terminated}, truncated={truncated}")
    print(f"추가 정보: {info}")

    env.close()


if __name__ == "__main__":
    main()
