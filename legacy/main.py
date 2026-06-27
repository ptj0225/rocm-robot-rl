import gymnasium as gym


def main():
    env = gym.make("Ant-v5", render_mode="rgb_array")
    observation, info = env.reset()

    total_reward = 0.0
    for step in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 100 == 0:
            print(f"Step {step:4d} | Reward: {reward:+.4f} | Total: {total_reward:+.4f}")

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            observation, info = env.reset()
            total_reward = 0.0

    env.close()


if __name__ == "__main__":
    main()
