"""실험 큐가 끝난 뒤 README를 결과로 자동 마감.

- 각 실험 로그에서 최종 평가 보상/스텝/속도를 파싱해 결과 표 생성
- 가장 잘한 정책으로 데모 GIF(머니샷) 재렌더 (assets/g1_rough_demo.gif 덮어쓰기)
- README의 <!-- RESULTS:START/END --> 구간을 새 표로 교체
- 최고 모델 경로를 stdout 마지막 줄에 'BEST=...' 로 출력 (bash 커밋용)
"""

import glob
import os
import re
import subprocess
import sys

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# (표시이름, 네트워크, 모델파일, 로그파일)
EXPERIMENTS = [
    ("Baseline", "MLP (512,256,128)", "pipeline_g1_rough.pkl", "logs/g1_rough_full.log"),
    ("SimBa (no DR)", "SimBa 2×256/512", "pipeline_g1_simba.pkl", "logs/g1_simba_full.log"),
    ("MLP + DR", "MLP + domain rand.", "pipeline_g1_mlp_dr.pkl", "logs/mlp_dr.log"),
    ("SimBa big", "SimBa 3×384/512", "pipeline_g1_simba_big.pkl", "logs/simba_big.log"),
]
# 이어학습(filler) 결과도 자동 포함
for m in sorted(glob.glob("pipeline_g1_simba_cont*.pkl")):
    base = m[len("pipeline_g1_"):-4]
    EXPERIMENTS.append((base, "SimBa (resumed)", m, f"logs/{base}.log"))


def parse_log(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    out = {}
    m = re.findall(r"평균 보상:\s*([+-]?[\d.]+)\s*±\s*([\d.]+)", txt)
    if m:
        out["eval_mean"], out["eval_std"] = float(m[-1][0]), float(m[-1][1])
    steps = re.findall(r"step\s+([\d,]+)/([\d,]+)", txt)
    if steps:
        out["step"] = int(steps[-1][0].replace(",", ""))
        out["total"] = int(steps[-1][1].replace(",", ""))
    sps = re.findall(r"\(avg\s+(\d+)\)", txt)
    if sps:
        out["sps"] = int(sps[-1])
    er = re.findall(r"eval_reward\s+([+-][\d.]+)", txt)
    if er:
        out["last_eval"] = float(er[-1])
    return out


def fmt_steps(n):
    return f"{n/1e6:.0f} M" if n else "—"


def main():
    rows = []
    best = None  # (mean, name, model)
    for name, net, model, log in EXPERIMENTS:
        if not os.path.exists(model):
            continue
        st = parse_log(log) or {}
        mean = st.get("eval_mean", st.get("last_eval"))
        reward_cell = "—"
        if "eval_mean" in st:
            reward_cell = f"**{st['eval_mean']:+.1f}** ± {st['eval_std']:.1f}"
        elif "last_eval" in st:
            reward_cell = f"{st['last_eval']:+.1f}"
        rows.append(
            f"| {name} | {net} | {fmt_steps(st.get('total',0))} | "
            f"{reward_cell} | {st.get('sps','—')} |"
        )
        if mean is not None and (best is None or mean > best[0]):
            best = (mean, name, model)

    if not rows:
        print("결과 모델 없음 — README 갱신 생략")
        print("BEST=")
        return

    table = (
        "<!-- RESULTS:START -->\n"
        "| Run | Network | Steps | Eval reward | steps/s |\n"
        "|-----|---------|-------|-------------|---------|\n"
        + "\n".join(rows)
        + "\n\n"
        f"_Best policy: **{best[1]}** (eval reward {best[0]:+.1f}). "
        "Single Radeon AI PRO R9700 (gfx1201), 1024 envs._\n"
        "<!-- RESULTS:END -->"
    )

    # 머니샷 GIF (최고 정책)
    print(f"머니샷 렌더: {best[2]}", flush=True)
    r = subprocess.run(
        [PY, "src/utils/render_playground.py", best[2], "--x_vel", "1.0",
         "--camera", "track", "--steps", "360", "--width", "480", "--height", "360",
         "--gif", "--out", "assets/g1_rough_demo.gif"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("머니샷 렌더 실패(기존 GIF 유지):", r.stderr[-300:])

    # README 패치
    readme = open("README.md", encoding="utf-8").read()
    new = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", table, readme, flags=re.S)
    open("README.md", "w", encoding="utf-8").write(new)
    print("README 결과 표 갱신 완료", flush=True)
    print(f"BEST={best[2]}")


if __name__ == "__main__":
    main()
