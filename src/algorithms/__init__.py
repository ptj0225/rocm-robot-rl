from src.algorithms.ppo import PPOTrainer
from src.algorithms.ppo_playground import PlaygroundPPOTrainer

REGISTRY = {
    "ppo": PPOTrainer,
    "ppo_playground": PlaygroundPPOTrainer,
}
