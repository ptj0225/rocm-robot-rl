# Running MJX / Brax RL on AMD RDNA4 (gfx1201) — a field guide

Hard-won notes from getting Unitree G1 rough-terrain RL working on a **Radeon AI PRO R9700
(gfx1201 / RDNA4, 32 GB)** with **ROCm 7.2.4**. If you're hitting cryptic ROCm crashes with
JAX/MJX on a new AMD card, start here.

> TL;DR: **upgrade to `jax-rocm 0.9.2`**, don't use SimBa + domain randomization together,
> cap `num_envs` at ~1024, never hard-kill a run, and verify logic on CPU.

---

## The one fix that matters: jax-rocm 0.9.2

On the stock `0.8.2+rocm7.2.4` wheels (from `repo.radeon.com`), training G1 aborts with:

```
ROCM_ERROR_ILLEGAL_ADDRESS
Check failed: ... Failed to set device
```

…even though the **same code runs perfectly on CPU** (`JAX_PLATFORMS=cpu`). That asymmetry is
the tell: it's a **gfx1201 code-generation bug in the ROCm XLA backend**, not your code.

**Fix — install the newer self-contained TheRock wheels from PyPI:**

```bash
uv pip install jax==0.9.2 jaxlib==0.9.2 jax-rocm7-pjrt==0.9.2 jax-rocm7-plugin==0.9.2
```

These bundle their own ROCm, so you don't have to upgrade the system ROCm (7.2.4 here).
With 0.9.2 the force-sensor, domain-randomization, and large-`num_envs` crashes all disappear.

**Do not jump to 0.10.x.** It works on the GPU, but JAX 0.10 removed `jax.device_put_replicated`,
which Brax 0.14.2 still calls → `train()` dies with a deprecation error. **0.9.2 is the sweet spot**
(new enough to fix gfx1201, old enough for Brax).

---

## How we localized the original 0.8.2 bug

A useful methodology if you hit a different gfx1201 crash:

1. **CPU works, GPU aborts** → it's the compiler/driver, not your logic. (`JAX_PLATFORMS=cpu`)
2. **Single `env.step` works; vmap+`lax.scan` aborts** → it's the batched/fused kernel.
3. **Bisect what's in the graph.** We found raw `mjx.step` was fine until we forced the
   **sensor stage** to be kept (XLA had been DCE-ing it). Narrowing further, the trigger was the
   **`mjSENS_FORCE` (force) sensors** specifically — distance/contact-style sensors miscompiled.
4. **brax `humanoid` (17-DoF, simple geoms) trained fine** while G1 didn't → it correlated with
   the contact/force sensor kernels, not DoF count.

(On 0.9.2 all of this is moot — but the *method* — CPU-diff, single-vs-scan, DCE-aware sensor
checks — is how you isolate a ROCm codegen fault.)

---

## Known limits on gfx1201 (with 0.9.2)

| Symptom | Cause | What to do |
|---|---|---|
| `xtile_compiler.cc … transpose` error log | ROCm tiled-compiler noise | **Ignore** — non-fatal, training continues |
| **SimBa + domain randomization** → `MEMORY_APERTURE_VIOLATION` | per-env-model vmap × SimBa graph = codegen abort (reducing memory does **not** help) | Use SimBa **or** DR, not both. MLP+DR and SimBa+noDR each work. |
| `num_envs ≥ ~4096` aborts | training-graph memory on the 32 GB card | cap at **1024** (validated); go higher only on NVIDIA |
| GPU "wedge": small ops pass, big ones abort | a previously killed run left the GPU in a bad state | `sudo rocm-smi --gpureset -d 0`, or reboot |

---

## GPU "wedge" — what it is and why it happens

After you **kill** a JAX process mid-kernel (or after an illegal-address crash), the GPU can be
left in a corrupted state: a tiny matmul still succeeds (health check passes), but the large
training graph aborts with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` ("accessed memory beyond
the largest legal address" = the page-table/aperture mapping is broken).

Why: when a kernel faults, a mature CUDA driver cleanly recovers the context. The **young RDNA4
ROCm driver doesn't fully recover**, so queues/memory mappings stay corrupted — especially since
`XLA_PYTHON_CLIENT_PREALLOCATE=true` grabs ~75% of VRAM that a killed process may not release.

Recovery:
```bash
sudo rocm-smi --gpureset -d 0     # clears queues/state (note: the AMD card may also drive your display)
# if that doesn't take, reboot
```

**Prevention — the practical takeaway:** *don't hard-kill training runs.* This repo checkpoints
every eval and supports `restore_path=`, so you can stop safely and resume instead of `kill -9`.

---

## Environment setup this repo relies on

`rocm_setup.py` (imported before JAX everywhere) sets:

- `HIP_DEVICE_LIB_PATH` → ROCm device bitcode (else GEMM codegen segfaults)
- `XLA_PYTHON_CLIENT_PREALLOCATE=true` (stable for large `num_envs`)
- `--xla_gpu_enable_command_buffer=` (command buffers off — required for stability on ROCm)

Based on AMD's [ROCm + JAX + MuJoCo guide](https://rocm.blogs.amd.com/artificial-intelligence/rocm-jax-mujoco/README.html).

---

## This machine, for reference

- AMD **Radeon AI PRO R9700** (gfx1201, RDNA4, 32 GB) — primary compute (and display)
- NVIDIA RTX 3060 (12 GB) — present; could run Isaac Lab / CUDA JAX if desired
- ROCm 7.2.4, Python 3.12, Linux

If you reproduce these findings on other RDNA4 cards (RX 9070, etc.) or newer ROCm, PRs updating
this doc are very welcome.
