"""MJX/Brax 정책을 EGL 오프스크린으로 비디오 렌더링.

rollout 전체를 lax.scan으로 JIT 컴파일해 단일 GPU 디스패치로 실행 (느린 스텝별 동기화 방지).

사용 예:
    .venv-mjx/bin/python src/utils/render_mjx.py pipeline_pusher.pkl --env pusher
    .venv-mjx/bin/python src/utils/render_mjx.py pipeline_pusher.pkl --env pusher --steps 600 --gif
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("MUJOCO_GL", "egl")
import rocm_setup  # noqa: E402

rocm_setup.configure_rocm_runtime()

import imageio.v2 as imageio  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from brax import envs  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402

from src.algorithms.base import load_policy  # noqa: E402

FPS = 30


def make_jit_rollout(env, jit_inference, max_steps: int):
    """전체 rollout을 lax.scan으로 JIT 컴파일. 단일 GPU 디스패치로 실행."""

    @jax.jit
    def _step(carry, _):
        state, rng = carry
        rng, act_rng = jax.random.split(rng)
        action, _ = jit_inference(state.obs, act_rng)
        state = env.step(state, action)
        reward = state.reward
        done = state.done
        return (state, rng), (state.pipeline_state.qpos, state.pipeline_state.qvel, reward, done)

    @jax.jit
    def rollout(init_state, rng):
        def scan_fn(carry, _):
            return _step(carry, None)

        (final_state, _), (qpos, qvel, rewards, dones) = jax.lax.scan(
            scan_fn, (init_state, rng), None, length=max_steps
        )
        return qpos, qvel, rewards, dones

    return rollout


def main():
    p = argparse.ArgumentParser(description="MJX 정책 비디오 렌더링 (EGL headless)")
    p.add_argument("model", help="저장된 .pkl 정책 파일")
    p.add_argument("--env", default="pusher", help="환경 이름")
    p.add_argument("--steps", type=int, default=200, help="rollout 스텝 수")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", action="store_true", help="mp4 대신 gif")
    p.add_argument("--out", default=None)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()

    out = args.out or (f"rollout_{args.env}.gif" if args.gif else f"rollout_{args.env}.mp4")

    print("loading model...", flush=True)
    params_np, meta = load_policy(args.model)
    params = jax.tree.map(jnp.asarray, params_np)

    if meta["env"] and meta["env"] != args.env:
        print(f"  경고: 저장된 env={meta['env']} 이지만 --env {args.env} 로 렌더링합니다.", flush=True)

    env = envs.get_environment(args.env, backend="mjx")
    # 저장된 네트워크 크기로 복원 (없으면 brax 기본값 — 레거시 raw-tuple 모델).
    # normalize_observations로 학습한 정책은 동일 전처리기가 필요(없으면 로봇이 즉시 넘어짐).
    net_kwargs = {"preprocess_observations_fn": running_statistics.normalize}
    if meta["policy_sizes"]:
        net_kwargs["policy_hidden_layer_sizes"] = tuple(meta["policy_sizes"])
    if meta["value_sizes"]:
        net_kwargs["value_hidden_layer_sizes"] = tuple(meta["value_sizes"])
    ppo_net = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size, action_size=env.action_size, **net_kwargs
    )
    make_policy = ppo_networks.make_inference_fn(ppo_net)
    jit_inference = jax.jit(make_policy(params, deterministic=True))

    print(f"compiling rollout ({args.steps} steps as single JIT)...", flush=True)
    t0 = time.time()
    jit_rollout = make_jit_rollout(env, jit_inference, args.steps)
    init_state = env.reset(jax.random.PRNGKey(args.seed))
    qpos, qvel, rewards, dones = jit_rollout(init_state, jax.random.PRNGKey(args.seed))
    # JIT 컴파일 + 실행 완료 대기 (JAX는 lazy)
    qpos.block_until_ready()
    print(f"  rollout done in {time.time()-t0:.1f}s", flush=True)

    total_reward = float(np.asarray(rewards).sum())
    qpos_np = np.asarray(qpos)
    qvel_np = np.asarray(qvel)
    print(f"  total reward: {total_reward:+.1f} | {len(qpos_np)} frames", flush=True)

    print("rendering (MuJoCo EGL)...", flush=True)
    import mujoco  # noqa: E402

    mj_model = env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, args.height, args.width)
    t0 = time.time()
    frames = []
    for i in range(len(qpos_np)):
        mj_data.qpos[:] = qpos_np[i]
        mj_data.qvel[:] = qvel_np[i]
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data)
        frames.append(renderer.render())
    renderer.close()
    print(f"  {len(frames)} frames in {time.time()-t0:.1f}s", flush=True)

    if args.gif:
        imageio.mimwrite(out, frames, fps=FPS)
    else:
        imageio.mimwrite(out, frames, fps=FPS, codec="libx264", quality=8)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
