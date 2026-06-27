from omegaconf import DictConfig

import jax  # noqa: E402 — rocm_setup must be called before import at entry point
from brax import envs  # noqa: E402

# MJX backend 호환 brax 환경. (swimmer는 MJX 미지원)
SUPPORTED_ENVS = {
    # 보행 (locomotion)
    "ant":                     {"obs": 27,  "act": 8,  "desc": "4족 보행 로봇"},
    "halfcheetah":             {"obs": 18,  "act": 6,  "desc": "2D 치타 달리기"},
    "hopper":                  {"obs": 11,  "act": 3,  "desc": "1족 깡총 뛰기"},
    "walker2d":                {"obs": 17,  "act": 6,  "desc": "2족 보행"},
    "humanoid":                {"obs": 244, "act": 17, "desc": "인간형 전신 (최대)"},
    "humanoidstandup":         {"obs": 244, "act": 17, "desc": "인간형 일어서기"},
    # 조작 (manipulation)
    "pusher":                  {"obs": 23,  "act": 7,  "desc": "물체 밀어 목표까지"},
    "reacher":                 {"obs": 11,  "act": 2,  "desc": "팔 뻗어 목표 도달"},
    # 고전 제어 (classic control)
    "inverted_pendulum":       {"obs": 4,   "act": 1,  "desc": "막대 세우기"},
    "inverted_double_pendulum":{"obs": 8,   "act": 1,  "desc": "이중 막대 세우기"},
    "fast":                    {"obs": 2,   "act": 1,  "desc": "간단한 제어 태스크"},
}


def make_env(cfg: DictConfig):
    backend = cfg.get("backend", "mjx")

    if backend == "playground":
        # MuJoCo Playground 환경 (예: G1JoystickRoughTerrain). cfg.env는 registry 이름 그대로.
        return _make_playground_env(cfg)

    name = cfg.env.lower()
    if name not in SUPPORTED_ENVS:
        raise ValueError(f"지원하지 않는 환경: {name}. 지원 목록: {sorted(SUPPORTED_ENVS)}")
    return envs.get_environment(name, backend=backend)


def _make_playground_env(cfg: DictConfig):
    """Playground 환경 로드. AMD ROCm(gfx1201/RDNA4) 호환 워크어라운드 포함.

    gfx1201 ROCm의 MJX 코드젠 버그: 힘 센서(mjSENS_FORCE)를 vmap+scan으로 컴파일하면
    ROCM_ERROR_ILLEGAL_ADDRESS로 죽는다(물리·다른 센서·접촉 센서는 정상). G1 등은 힘 센서를
    contact_force 보상항에만 쓰므로, 해당 센서를 모델에서 제거하고 그 항을 0으로 무력화하면
    학습이 정상 동작한다. impl='warp'(mujoco_warp)가 깔려 있으면 이 우회가 불필요하지만
    미설치 환경에서는 jax 백엔드로 강제한다.
    """
    from mujoco_playground import registry

    strip_force = bool(cfg.get("rocm_strip_force_sensors", False))
    env_name = cfg.env

    overrides = {"impl": "jax"}
    # 접촉 버퍼 크기. 0이면 환경 기본값(G1=65536). 대량 envs로 VRAM이 부족하면 축소.
    naconmax = int(cfg.get("naconmax", 0))
    if naconmax > 0:
        overrides["naconmax"] = naconmax

    if not strip_force:
        return registry.load(env_name, config_overrides=overrides)

    import re

    from mujoco_playground._src.locomotion.g1 import base as g1base

    _orig_get_assets = g1base.get_assets

    def _patched_get_assets():
        assets = dict(_orig_get_assets())
        for k, v in list(assets.items()):
            if isinstance(v, (bytes, bytearray)) and b"<force " in v:
                assets[k] = re.sub(r"\s*<force\b[^>]*?/>", "", v.decode()).encode()
        return assets

    g1base.get_assets = _patched_get_assets
    try:
        env = registry.load(env_name, config_overrides=overrides)
    finally:
        g1base.get_assets = _orig_get_assets

    # 힘 센서가 사라졌으므로 이를 읽는 contact_force 보상항을 0으로 대체.
    if hasattr(env, "_cost_contact_force"):
        import jax.numpy as jnp

        env._cost_contact_force = lambda data: jnp.zeros(())

    return env


def list_envs():
    for name, info in SUPPORTED_ENVS.items():
        print(f"  {name:28s}  obs={info['obs']:>3d}  act={info['act']:>2d}  {info['desc']}")


if __name__ == "__main__":
    print("=== MJX 호환 환경 목록 ===\n")
    list_envs()
