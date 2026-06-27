"""학습 실행 폴더(runs/<name>/) 구성 — TensorBoard 로깅 + 주기적 체크포인트.

폴더 구조:
    runs/<run_name>/
        config.yaml              # 이 실행의 설정 스냅샷
        checkpoints/
            step_<10자리>.pkl    # 평가마다 저장
            latest.pkl           # 최신 체크포인트 (이어학습/렌더용)
        events.out.tfevents.*    # TensorBoard 스칼라 로그

TensorBoard 보기:  tensorboard --logdir runs
"""

import os
import time

from omegaconf import OmegaConf


def setup_run(cfg, default_prefix: str):
    """runs/<name> 디렉터리와 TensorBoard writer를 만든다. (run_dir, writer) 반환."""
    base = cfg.get("logdir", "runs")
    name = cfg.get("run_name", "") or f"{default_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(base, name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    try:
        OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
    except Exception:
        pass

    writer = None
    if cfg.get("tensorboard", True):
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=run_dir)
    print(f"실행 폴더: {run_dir}  (TensorBoard: tensorboard --logdir {base})", flush=True)
    return run_dir, writer


def log_metrics(writer, step: int, metrics: dict, extra: dict | None = None):
    """brax metrics dict와 추가 스칼라를 TensorBoard에 기록."""
    if writer is None:
        return
    for k, v in metrics.items():
        try:
            writer.add_scalar(k, float(v), step)
        except (TypeError, ValueError):
            pass  # 스칼라가 아닌 값은 건너뜀
    for k, v in (extra or {}).items():
        writer.add_scalar(k, float(v), step)


def make_checkpoint_fn(cfg, run_dir: str, *, env: str, policy_sizes, value_sizes, net: dict):
    """brax policy_params_fn(step, make_policy, params) 콜백 생성 — 평가마다 체크포인트 저장."""
    from src.algorithms.base import save_policy

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    enabled = bool(cfg.get("checkpoint", True))

    def policy_params_fn(step, _make_policy, params):
        if not enabled:
            return
        path = os.path.join(ckpt_dir, f"step_{int(step):010d}.pkl")
        save_policy(path, params, env=env, policy_sizes=policy_sizes,
                    value_sizes=value_sizes, net=net, verbose=False)
        save_policy(os.path.join(ckpt_dir, "latest.pkl"), params, env=env,
                    policy_sizes=policy_sizes, value_sizes=value_sizes, net=net, verbose=False)

    return policy_params_fn
