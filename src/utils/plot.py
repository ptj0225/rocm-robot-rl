"""학습 로그에서 eval_reward 추이를 그래프로 저장.

사용 예:
    .venv-mjx/bin/python plot_training.py                    # logs/train_2048_60m.log → training_curve.png
    .venv-mjx/bin/python plot_training.py logs/other.log out.png
"""

import argparse
import re
import os

os.environ["MPLBACKEND"] = "Agg"  # headless

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

DEFAULT_LOG = "logs/train_2048_60m.log"
DEFAULT_OUT = "training_curve.png"
SOLVED = 6000


def parse_log(path: str) -> tuple[list[int], list[float]]:
    steps, rewards = [], []
    pattern = re.compile(r"step\s+([\d,]+)/[\d,]+\s+.*?eval_reward\s+([+-]?[\d.]+)")
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1).replace(",", "")))
                rewards.append(float(m.group(2)))
    return steps, rewards


def main():
    p = argparse.ArgumentParser(description="학습 곡선 plot")
    p.add_argument("log", nargs="?", default=DEFAULT_LOG)
    p.add_argument("out", nargs="?", default=DEFAULT_OUT)
    p.add_argument("--title", default="MJX + Brax PPO — Ant (bf16, 2048 envs)")
    args = p.parse_args()

    steps, rewards = parse_log(args.log)
    if not steps:
        print(f"ERROR: {args.log}에서 eval_reward 데이터를 찾을 수 없습니다.")
        return

    # steps를 million 단위로 변환
    steps_m = [s / 1e6 for s in steps]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps_m, rewards, color="#1f77b4", linewidth=1.5, marker="o", markersize=3)
    ax.axhline(SOLVED, color="red", linestyle="--", alpha=0.7, label=f"Solved ({SOLVED})")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("Eval Reward")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:+.0f}"))
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"저장: {args.out}")
    print(f"데이터: {len(steps)}개 eval | 최종 eval_reward = {rewards[-1]:+.1f}")


if __name__ == "__main__":
    main()
