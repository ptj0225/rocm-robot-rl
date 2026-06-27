"""ROCm(gfx1201/RDNA4)에서 JAX/XLA가 안정적으로 동작하도록 환경변수 세팅.

JAX import 전에 configure_rocm_runtime()을 호출해야 함.
AMD ROCm+JAX 공식 가이드(rocm-jax-mujoco) 기반이며, gfx1201 환경에서 검증됨.

해결하는 문제:
- HIP_DEVICE_LIB_PATH 누락 → AMDGPU GEMM 코드젠이 device bitcode를 못 찾아 segfault
- 커맨드 버퍼 → ROCm에서 비활성화해야 안정적
"""

import os

_ROCM_ROOT = "/opt/rocm"
_ROCM_VERSIONS = ("7.2.4", "7.2.0", "7.2")


def _find_bitcode_path() -> str:
    """ROCm device bitcode 경로를 찾는다. (버전별/심볼릭링크 대응)"""
    for root in (_ROCM_ROOT, *(f"/opt/rocm-{v}" for v in _ROCM_VERSIONS)):
        path = f"{root}/lib/llvm/lib/clang/22/lib/amdgcn/bitcode"
        if os.path.isdir(path):
            return path
    # 폴백: clang 버전이 다를 수 있어 마지막으로 발견된 경로 사용
    return f"{_ROCM_ROOT}/lib/llvm/lib/clang/22/lib/amdgcn/bitcode"


def configure_rocm_runtime() -> None:
    """JAX/ROCm 안정화 환경변수 설정. 이미 값이 있으면 덮어쓰지 않는다."""
    os.environ.setdefault("LLVM_PATH", f"{_ROCM_ROOT}/llvm")
    os.environ.setdefault("HIP_DEVICE_LIB_PATH", _find_bitcode_path())
    # 대규모(num_envs) 할당 시 동적 할당이 경계 위반을 일으킬 수 있어 사전할당 사용.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_enable_command_buffer" not in xla_flags:
        os.environ["XLA_FLAGS"] = f"{xla_flags} --xla_gpu_enable_command_buffer=".strip()


if __name__ == "__main__":
    # 진단용: 설정 적용 후 현재 값을 출력
    configure_rocm_runtime()
    for key in ("LLVM_PATH", "HIP_DEVICE_LIB_PATH", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        print(f"{key} = {os.environ.get(key)}")
