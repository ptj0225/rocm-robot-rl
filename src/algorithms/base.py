import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from omegaconf import DictConfig

from src.config import parse_sizes

# 저장 번들 포맷 식별자. 포맷이 바뀌면 버전을 올리고 load_policy에서 분기.
BUNDLE_FORMAT = "brax_ppo_v1"


def save_policy(path: str, params, *, env: str, policy_sizes, value_sizes, net: dict | None = None) -> None:
    """정책 params + 네트워크 메타데이터를 단일 번들(dict)로 저장.

    렌더링/평가 시 동일한 네트워크 구조를 복원하려면 layer size/네트워크 종류가 필요하므로 함께 저장.
    `net`은 SimBa 등 비-MLP 구조 복원용 메타(예: {"network":"simba","simba_policy_blocks":2,...}).
    """
    import jax

    np_params = jax.tree.map(np.asarray, params)
    bundle = {
        "format": BUNDLE_FORMAT,
        "params": np_params,
        "env": env,
        "policy_sizes": list(policy_sizes),
        "value_sizes": list(value_sizes),
        "net": net or {"network": "mlp"},
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"모델 저장: {path}")


def load_policy(path: str) -> tuple[Any, dict]:
    """저장된 정책을 로드. (params, meta)를 반환.

    - 신규 번들(dict): params와 env/policy_sizes/value_sizes 메타를 그대로 복원.
    - 레거시 raw tuple (normalizer, policy, value): 메타 없이 params만 반환하고
      size는 None으로 둔다(호출 측에서 brax 기본값 사용).
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "params" in obj:
        meta = {k: obj.get(k) for k in ("env", "policy_sizes", "value_sizes")}
        meta["net"] = obj.get("net", {"network": "mlp"})
        return obj["params"], meta

    # 레거시 포맷: brax PPO params 튜플을 그대로 pickle 했던 경우
    return obj, {"env": None, "policy_sizes": None, "value_sizes": None, "net": {"network": "mlp"}}


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, env, cfg: DictConfig) -> tuple[Any, Any, Any]:
        """Returns (eval_env, make_inference_fn, params)."""

    @abstractmethod
    def evaluate(self, model: tuple, env, cfg: DictConfig, n_episodes: int = 50) -> np.ndarray:
        """Returns rewards array of shape (n_episodes,)."""

    def save(self, model: tuple, cfg: DictConfig, path: str) -> None:
        _, _, params = model
        p = cfg.ppo
        net = {
            "network": p.get("network", "mlp"),
            "policy_obs_key": p.get("policy_obs_key", "state"),
            "value_obs_key": p.get("value_obs_key", "state"),
            "simba_policy_blocks": p.get("simba_policy_blocks", 2),
            "simba_policy_hidden": p.get("simba_policy_hidden", 256),
            "simba_value_blocks": p.get("simba_value_blocks", 2),
            "simba_value_hidden": p.get("simba_value_hidden", 512),
        }
        save_policy(
            path,
            params,
            env=cfg.env,
            policy_sizes=parse_sizes(cfg.ppo.policy_hidden_layer_sizes),
            value_sizes=parse_sizes(cfg.ppo.value_hidden_layer_sizes),
            net=net,
        )

    @staticmethod
    def load(path: str) -> tuple[Any, dict]:
        return load_policy(path)
