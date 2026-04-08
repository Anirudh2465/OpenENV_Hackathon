"""
OpenEnv-Orbital-Command | scripts/benchmark.py

Runs all 5 tasks back-to-back with the rule-based agent (or any backend),
prints a leaderboard-style comparison table, and saves results to the SQLite leaderboard.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --backend gemini --model gemini-2.0-flash
    python scripts/benchmark.py --backend huggingface --no-events
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from env.orbital_env import OrbitalEnv
from agent.llm_agent import create_agent
from scoring.leaderboard import submit_result, get_task_stats

TASKS = {
    1: ("Eclipse Survival",      100, "⭐       Easy"),
    2: ("Storage Bottleneck",    150, "⭐⭐      Medium"),
    3: ("Laser Cross-Link",      120, "⭐⭐⭐     Hard"),
    4: ("Global Swarm Harvest",  200, "⭐⭐⭐⭐    Very Hard"),
    5: ("Emergency Response",    180, "⭐⭐⭐⭐⭐   Expert"),
}

GRADE_COLOUR = {"S": "\033[33m", "A": "\033[32m", "B": "\033[34m", "C": "\033[33m", "F": "\033[31m"}
RESET = "\033[0m"


def _grade_coloured(g: str) -> str:
    return GRADE_COLOUR.get(g, "") + g + RESET


def run_all(backend: str, model: str, max_steps_override: int | None,
            seed: int, events_enabled: bool, agent_name: str) -> None:
    print(f"\n{'═'*70}")
    print(f"  🛰  ORBITAL COMMAND · BENCHMARK RUN")
    print(f"  Backend: {backend}  |  Model: {model}  |  Seed: {seed}")
    print(f"{'═'*70}\n")

    results = []
    total_t0 = time.time()

    for task_id, (task_name, default_steps, diff_label) in TASKS.items():
        max_steps = max_steps_override or default_steps
        agent = create_agent(backend, **({"model": model} if backend != "rule_based" else {}))
        env = OrbitalEnv(task_id=task_id, seed=seed,
                         max_steps=max_steps, events_enabled=events_enabled)
        obs, info = env.reset()

        print(f"  Task {task_id}/5  {diff_label}  —  {task_name}", end="", flush=True)
        t0 = time.time()

        step = 0
        while True:
            action = agent.act(obs)
            obs, r, done, trunc, info = env.step(action)
            agent.record_step(step, action, r)
            step += 1
            if done or trunc:
                break

        elapsed = time.time() - t0
        result = env.get_episode_result()
        ep_id = submit_result(result, agent_name=agent_name,
                              model_name=getattr(agent, "model", backend))
        results.append(result)
        print(f"  →  {_grade_coloured(result.grade)}  {result.normalized_score:.3f}  "
              f"({result.final_score:.0f} pts)  {step} steps  {elapsed:.1f}s")

    total_elapsed = time.time() - total_t0

    # Summary table
    print(f"\n{'═'*70}")
    print(f"  BENCHMARK SUMMARY  —  Total time: {total_elapsed:.1f}s")
    print(f"{'═'*70}")
    print(f"  {'Task':<6} {'Name':<28} {'Score':>7} {'Norm':>6} {'Grade':>5} "
          f"{'DL-GB':>7} {'Steps':>6}")
    print(f"  {'─'*6} {'─'*28} {'─'*7} {'─'*6} {'─'*5} {'─'*7} {'─'*6}")

    for i, r in enumerate(results, 1):
        task_name = TASKS[i][0]
        print(f"  {i:<6} {task_name:<28} {r.final_score:>7.0f} "
              f"{r.normalized_score:>6.3f} {_grade_coloured(r.grade):>12} "   # extra width for ANSI
              f"{r.data_downlinked_gb:>7.2f} {r.total_steps:>6}")
    print(f"{'═'*70}")

    avg_norm = sum(r.normalized_score for r in results) / len(results)
    composite_grade = ("S" if avg_norm >= 0.95 else "A" if avg_norm >= 0.80
                       else "B" if avg_norm >= 0.60 else "C" if avg_norm >= 0.40 else "F")
    print(f"\n  Overall Avg Normalised Score: {avg_norm:.3f}  "
          f"Composite Grade: {_grade_coloured(composite_grade)}")
    print(f"  Results saved to local leaderboard (SQLite).\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all 5 Orbital Command tasks and compare results")
    parser.add_argument("--backend", default="rule_based",
                        choices=["rule_based", "gemini", "huggingface"])
    parser.add_argument("--model",   default="gemini-2.0-flash")
    parser.add_argument("--steps",   type=int, default=None,
                        help="Override max_steps for all tasks (default: task-specific)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--name",    default="BenchmarkAgent")
    parser.add_argument("--no-events", action="store_true")
    args = parser.parse_args()

    run_all(
        backend=args.backend,
        model=args.model,
        max_steps_override=args.steps,
        seed=args.seed,
        events_enabled=not args.no_events,
        agent_name=args.name,
    )
