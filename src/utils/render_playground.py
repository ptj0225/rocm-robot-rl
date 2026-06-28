"""학습된 MuJoCo Playground 정책(예: G1 험지 보행)을 비디오로 렌더링.

State 트래젝토리를 단일 env로 순차 수집(gfx1201에서 vmap+scan 융합 크래시 회피)한 뒤
env.render(states)로 프레임을 만든다. EGL 오프스크린(headless).

사용 예:
    .venv-mjx/bin/python src/utils/render_playground.py pipeline_g1_rough.pkl --env G1JoystickRoughTerrain
    .venv-mjx/bin/python src/utils/render_playground.py pipeline_g1_rough.pkl --steps 500 --x_vel 1.0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("MUJOCO_GL", "egl")
# 롤아웃은 CPU에서 수행: gfx1201 ROCm은 G1 env 스테핑(vmap/scan/force센서)에서 illegal address로
# 죽는다. 렌더링은 성능보다 안정성이 중요하므로 JAX를 CPU로 강제. (렌더 자체는 EGL 오프스크린)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import rocm_setup  # noqa: E402

rocm_setup.configure_rocm_runtime()

import imageio.v2 as imageio  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.algorithms.base import load_policy  # noqa: E402
from src.envs import make_env  # noqa: E402

FPS = 30


def main():
    p = argparse.ArgumentParser(description="Playground 정책 비디오 렌더링 (EGL headless)")
    p.add_argument("model", help="저장된 .pkl 정책 (bundle)")
    p.add_argument("--env", default="G1JoystickRoughTerrain")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--x_vel", type=float, default=1.0, help="전진 속도 명령 (m/s)")
    p.add_argument("--y_vel", type=float, default=0.0)
    p.add_argument("--yaw_vel", type=float, default=0.0)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera", default="track", help="카메라 이름 (로봇 추적: track)")
    args = p.parse_args()

    out = args.out or (f"media/rollout_{args.env}.gif" if args.gif else f"media/rollout_{args.env}.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print("loading policy + env...", flush=True)
    params, meta = load_policy(args.model)
    params = jax.tree.map(jnp.asarray, params)

    env = make_env(OmegaConf.create({"env": args.env, "backend": "playground"}))

    # 학습 시 normalize_observations=True 였으므로 동일한 정규화 전처리기를 써야 정책이 정상 동작.
    # (빠뜨리면 raw obs가 들어가 로봇이 즉시 넘어진다.)
    netm = meta.get("net") or {"network": "mlp"}
    pkey = netm.get("policy_obs_key", "state")
    vkey = netm.get("value_obs_key", "privileged_state")

    if netm.get("network") == "simba":
        from src.algorithms.simba import make_simba_ppo_networks

        ppo_net = make_simba_ppo_networks(
            observation_size=env.observation_size,
            action_size=env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            policy_obs_key=pkey,
            value_obs_key=vkey,
            policy_num_blocks=netm.get("simba_policy_blocks", 2),
            policy_hidden=netm.get("simba_policy_hidden", 256),
            value_num_blocks=netm.get("simba_value_blocks", 2),
            value_hidden=netm.get("simba_value_hidden", 512),
        )
    else:
        net_kwargs = {
            "policy_obs_key": pkey,
            "value_obs_key": vkey,
            "preprocess_observations_fn": running_statistics.normalize,
        }
        if meta["policy_sizes"]:
            net_kwargs["policy_hidden_layer_sizes"] = tuple(meta["policy_sizes"])
        if meta["value_sizes"]:
            net_kwargs["value_hidden_layer_sizes"] = tuple(meta["value_sizes"])
        ppo_net = ppo_networks.make_ppo_networks(
            observation_size=env.observation_size, action_size=env.action_size, **net_kwargs
        )
    inference = jax.jit(ppo_networks.make_inference_fn(ppo_net)(params, deterministic=True))

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    print(f"rolling out {args.steps} steps (cmd: x={args.x_vel} y={args.y_vel} yaw={args.yaw_vel})...", flush=True)
    rng = jax.random.PRNGKey(args.seed)
    state = jit_reset(rng)
    # 속도 명령 주입 (joystick 환경은 info['command']로 목표 속도를 받는다)
    if "command" in state.info:
        cmd = jnp.array([args.x_vel, args.y_vel, args.yaw_vel])
        state = state.replace(info={**state.info, "command": cmd})

    states = [state]
    total_reward = 0.0
    for _ in range(args.steps):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference(state.obs, act_rng)
        state = jit_step(state, action)
        if "command" in state.info:
            state = state.replace(info={**state.info, "command": cmd})
        total_reward += float(state.reward)
        states.append(state)
        if bool(state.done):
            print("  episode terminated early", flush=True)
            break
    print(f"  {len(states)} frames | total reward {total_reward:+.1f}", flush=True)

    print("rendering...", flush=True)
    frames = env.render(states, height=args.height, width=args.width, camera=args.camera)

    if args.gif:
        imageio.mimwrite(out, frames, fps=FPS)
    else:
        imageio.mimwrite(out, frames, fps=FPS, codec="libx264", quality=8)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
