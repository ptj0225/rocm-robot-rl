import sys
from dataclasses import dataclass, field

from omegaconf import OmegaConf


def parse_sizes(s: str) -> tuple:
    """\"512,256,128\" → (512, 256, 128). OmegaConf DictConfig는 dataclass 메서드를
    보존하지 않으므로 layer size 파싱은 이 모듈 함수를 통해 한다."""
    return tuple(int(x) for x in str(s).split(","))


@dataclass
class PPOConfig:
    num_timesteps: int = 60_000_000
    num_envs: int = 2048
    num_evals: int = 20
    num_eval_envs: int = 128
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.99
    unroll_length: int = 20
    batch_size: int = 1024
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    reward_scaling: float = 0.1
    episode_length: int = 1000
    normalize_observations: bool = True
    precision: str = "bfloat16"
    deterministic_eval: bool = True
    policy_hidden_layer_sizes: str = "256,256,256,256"
    value_hidden_layer_sizes: str = "256,256,256,256,256"
    # brax/playground 공용 추가 파라미터 (기본값은 brax 디폴트와 동일)
    clipping_epsilon: float = 0.3
    max_grad_norm: float = 1.0
    gae_lambda: float = 0.95
    num_resets_per_eval: int = 0
    # 비대칭 actor-critic용 obs 키 (dict 관측 환경에서만 사용; 빈 문자열이면 기본 'state')
    policy_obs_key: str = "state"
    value_obs_key: str = "state"
    # 도메인 랜덤화(playground 한정). jax-rocm 0.9.2+ 에서 gfx1201 버그가 해결되어 사용 가능.
    domain_randomization: bool = False
    # 신경망 구조: "mlp"(기본) 또는 "simba"(residual+LayerNorm, ICLR'25 — 보통 더 우수).
    network: str = "mlp"
    simba_policy_blocks: int = 2
    simba_policy_hidden: int = 256
    simba_value_blocks: int = 2
    simba_value_hidden: int = 512


@dataclass
class Config:
    env: str = "ant"
    backend: str = "mjx"
    seed: int = 0
    model_path: str = "pipeline.pkl"
    # 비우면 처음부터 학습. 경로를 주면 해당 .pkl(번들)에서 파라미터를 이어받아 추가 학습.
    restore_path: str = ""
    # playground: 힘 센서 제거 워크어라운드(jax-rocm 0.9.2+에서는 불필요 → 기본 off).
    rocm_strip_force_sensors: bool = False
    # playground 접촉 버퍼 크기. 0이면 환경 기본값 사용. 대량 envs 시 메모리 절약용으로 축소 가능.
    naconmax: int = 0
    # 실행 로깅/체크포인트 (runs/<run_name>/). run_name 비우면 자동 타임스탬프.
    logdir: str = "runs"
    run_name: str = ""
    tensorboard: bool = True
    checkpoint: bool = True
    ppo: PPOConfig = field(default_factory=PPOConfig)


def load_config(defaults=Config) -> OmegaConf:
    config_path = None
    remaining = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        else:
            remaining.append(args[i])
            i += 1

    cfg = OmegaConf.structured(defaults)
    if config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
    cli = OmegaConf.from_cli(remaining)
    return OmegaConf.merge(cfg, cli)
