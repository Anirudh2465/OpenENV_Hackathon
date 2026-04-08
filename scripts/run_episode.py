"""
OpenEnv-Orbital-Command | scripts/run_episode.py
CLI runner for quick testing without the Gradio UI.

Usage:
    python scripts/run_episode.py --task 1 --backend rule_based
    python scripts/run_episode.py --task 3 --backend openai --model gpt-4o
    python scripts/run_episode.py --task 5 --steps 150 --render
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from env.orbital_env import OrbitalEnv
from agent.llm_agent import create_agent
from scoring.leaderboard import submit_result


def run(
    task_id: int = 1,
    backend: str = "rule_based",
    model: str = "gpt-4o",
    max_steps: int = 200,
    seed: int = 42,
    render: bool = False,
    events_enabled: bool = True,
    agent_name: str = "CLI Runner",
):
    print(f"\n{'='*60}")
    print(f"  ORBITAL COMMAND  |  Task {task_id}  |  Backend: {backend}")
    print(f"{'='*60}")

    env   = OrbitalEnv(task_id=task_id, seed=seed,
                       max_steps=max_steps, events_enabled=events_enabled)
    agent = create_agent(backend, **({"model": model} if backend != "rule_based" else {}))
    obs, info = env.reset()

    print(f"\nEpisode ID: {info['episode_id']}")
    print(f"Task: {obs.task_name}")
    print(f"Satellites: {[s.sat_id for s in obs.satellites]}")
    print(f"Requests: {[r.id for r in obs.imaging_requests]}\n")

    step = 0
    total_reward = 0.0
    t0 = time.time()

    while True:
        if render:
            print(env.render())

        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.record_step(step, action, reward)

        total_reward += reward
        step += 1

        # Step summary
        ar = info.get("action_result", {})
        print(f"[{step:03d}] {action.action_type.value:25s} → {action.target_sat_id:15s} "
              f"reward={reward:+7.1f}  total={total_reward:+8.1f}  {ar.get('message', '')[:50]}")

        if terminated or truncated:
            break

    elapsed = time.time() - t0
    result  = env.get_episode_result()

    print(f"\n{'='*60}")
    print(f"  EPISODE COMPLETE in {step} steps ({elapsed:.1f}s)")
    print(f"  Final Score:       {result.final_score:.2f}")
    print(f"  Normalised Score:  {result.normalized_score:.4f}")
    print(f"  Grade:             {result.grade}")
    print(f"  Data Downlinked:   {result.data_downlinked_gb:.2f} GB")
    print(f"  Data Overwritten:  {result.data_overwritten_gb:.2f} GB")
    print(f"  Satellites Alive:  {result.satellites_survived}/{result.total_satellites}")
    print(f"  Requests Done:     {result.requests_fulfilled}/{result.requests_fulfilled + result.requests_missed}")
    print(f"  Agent Avg Latency: {agent.avg_latency_ms:.0f} ms/step")
    print(f"\n  Grader Breakdown:")
    for k, v in result.grader_breakdown.items():
        print(f"    {k:30s}: {v}")
    print(f"{'='*60}\n")

    ep_id = submit_result(result, agent_name=agent_name, model_name=getattr(agent, "model", backend))
    print(f"Result saved to leaderboard (episode_id={ep_id})")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an Orbital Command episode")
    parser.add_argument("--task",    type=int, default=1, choices=[1,2,3,4,5])
    parser.add_argument("--backend", type=str, default="rule_based",
                        choices=["rule_based","openai","anthropic","huggingface"])
    parser.add_argument("--model",   type=str, default="gpt-4o")
    parser.add_argument("--steps",   type=int, default=200)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--render",  action="store_true")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--name",    type=str, default="CLI Runner")
    args = parser.parse_args()

    run(
        task_id=args.task,
        backend=args.backend,
        model=args.model,
        max_steps=args.steps,
        seed=args.seed,
        render=args.render,
        events_enabled=not args.no_events,
        agent_name=args.name,
    )
