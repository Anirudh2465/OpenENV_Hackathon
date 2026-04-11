"""
OpenEnv-Orbital-Command | ui/app.py

Gradio 6.x mission control dashboard.

API KEYS — Three ways to configure (in order of priority):
  1. Type them directly in the '🔐 API Keys' section of the UI (not saved to disk)
  2. Create a `.env` file in the project root (copy `.env.example`)
  3. Set OS environment variables before launching

Launch:
    python -m ui.app                   # Rule-based agent (no key needed)
    python -m ui.app --backend openai  # GPT-4o (requires OPENAI_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# ── Project root on path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Load .env if present (silently skips if python-dotenv not installed) ──
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from env.orbital_env import OrbitalEnv
from env.models import Action, ActionType, Observation
from agent.llm_agent import create_agent, BaseAgent
from ui.orbit_3d import generate_orbit_3d
from scoring.leaderboard import submit_result, get_leaderboard

# ── Plotly (optional — falls back to sparkline if not installed) ──────────
try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_NAMES = {
    1: "☀️  Task 1 — Eclipse Survival  [Easy]",
    2: "💾  Task 2 — Storage Bottleneck  [Medium]",
    3: "🔗  Task 3 — Laser Cross-Link  [Hard]",
    4: "🌐  Task 4 — Swarm Harvest  [Very Hard]",
    5: "🚨  Task 5 — Emergency Response  [Expert]",
}

BACKEND_OPTIONS = {
    "✨  Gemini  (gemini-2.0-flash / 1.5-pro)": "gemini",
    "🤖  Rule-Based (No API key)":              "rule_based",
    "🤗  HuggingFace Inference API":            "huggingface",
}

MODEL_DEFAULTS = {
    "gemini":      "gemini-2.0-flash",
    "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
    "rule_based":  "RuleBased",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,600;1,400&family=Orbitron:wght@400;600;700;900&display=swap');

/* ── Base ─────────────────────────────────────────────── */
:root {
  --bg:         #02040a;
  --bg-card:    rgba(13, 21, 40, 0.4);
  --bg-panel:   rgba(15, 30, 53, 0.5);
  --border:     rgba(42, 114, 214, 0.3);
  --blue:       #2196f3;
  --cyan:       #00e5ff;
  --green:      #00e676;
  --amber:      #ffab40;
  --red:        #ff1744;
  --purple:     #7c4dff;
  --text:       #ddeeff;
  --muted:      #4a6888;
  --glow-b:     0 0 20px rgba(33,150,243,.4);
  --glow-c:     0 0 20px rgba(0,229,255,.4);
}

body {
  background: radial-gradient(circle at 50% 20%, #081224 0%, #010205 100%) !important;
  background-attachment: fixed !important;
  background-size: cover !important;
}
.gradio-container {
  background: transparent !important;
  font-family: 'JetBrains Mono', monospace !important;
  color: var(--text) !important;
}

/* ── Panels (Glassmorphism) ───────────────────────────── */
.panel {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 14px 16px !important;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(33, 150, 243, 0.05) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
}

/* ── Markdown ─────────────────────────────────────────── */
.prose, .markdown-body { color: var(--text) !important; }
table { border-collapse: collapse; width: 100%; border-radius: 8px; overflow: hidden; }
th { background: rgba(17, 30, 51, 0.8); color: var(--cyan); font-size: .75rem; padding: 6px 10px; text-align: left; }
td { border-top: 1px solid rgba(26, 45, 70, 0.5); padding: 5px 10px; font-size: .75rem; }
code { background: rgba(10, 21, 37, 0.6); border: 1px solid rgba(33, 150, 243, 0.2); border-radius: 4px; padding: 1px 5px; color: var(--cyan); }

/* ── Buttons ──────────────────────────────────────────── */
button.lg {
  background: linear-gradient(135deg, rgba(21, 101, 192, 0.8), rgba(13, 71, 161, 0.8)) !important;
  border: 1px solid var(--blue) !important;
  border-radius: 12px !important;
  font-family: 'Orbitron', sans-serif !important;
  font-size: .85rem !important;
  font-weight: 700 !important;
  letter-spacing: .08em !important;
  color: #fff !important;
  box-shadow: 0 0 16px rgba(33,150,243,.5), inset 0 0 10px rgba(255,255,255,0.2) !important;
  backdrop-filter: blur(8px) !important;
  text-shadow: 0 0 6px #fff !important;
  transition: all .2s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
}
button.lg:hover {
  box-shadow: 0 0 26px rgba(33,150,243,.8), inset 0 0 15px rgba(255,255,255,0.4) !important;
  transform: translateY(-2px) scale(1.02) !important;
  background: linear-gradient(135deg, rgba(30, 136, 229, 0.9), rgba(21, 101, 192, 0.9)) !important;
}
button.secondary { background: rgba(17, 30, 51, 0.5) !important; border-color: var(--border) !important; backdrop-filter: blur(4px) !important; }

/* ── Inputs ───────────────────────────────────────────── */
textarea, input[type=text], input[type=password], input[type=number], .block {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .8rem !important;
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.3) !important;
}
label > span { color: var(--cyan) !important; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; opacity: 0.8; }

/* ── Accordion ────────────────────────────────────────── */
.accordion { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; backdrop-filter: blur(12px) !important; }

/* ── Dropdowns ────────────────────────────────────────── */
select, .dropdown { background: var(--bg-panel) !important; border-color: var(--border) !important; color: var(--text) !important; }

/* ── Stat textboxes ───────────────────────────────────── */
#stat-step textarea, #stat-score textarea, #stat-req textarea {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 1.5rem !important;
  font-weight: 900 !important;
  color: #fff !important;
  text-shadow: var(--glow-c) !important;
  text-align: center !important;
  background: rgba(0, 229, 255, 0.05) !important;
  border: 1px solid rgba(0, 229, 255, 0.3) !important;
  box-shadow: inset 0 0 15px rgba(0, 229, 255, 0.1) !important;
}

/* ── Action log ───────────────────────────────────────── */
#action-log textarea {
  font-size: .76rem !important;
  line-height: 1.6 !important;
  min-height: 250px !important;
  background: rgba(0,0,0,0.3) !important;
}

/* ── API key panel ────────────────────────────────────── */
.api-key-note { font-size: .75rem; color: var(--muted); line-height: 1.6; padding: 8px 0 4px 0; }

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(42, 114, 214, 0.5); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* ── Orbit container ──────────────────────────────────── */
#orbit-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 560px;
  background: radial-gradient(circle at center, rgba(13, 71, 161, 0.2) 0%, rgba(0,0,0,0) 70%) !important;
  border: 1px solid rgba(42, 114, 214, 0.2);
  border-radius: 16px;
  box-shadow: inset 0 0 30px rgba(33, 150, 243, 0.1);
  position: relative;
}

/* Add a pulse to the orbit container border */
@keyframes border-pulse {
  0% { border-color: rgba(42, 114, 214, 0.2); box-shadow: inset 0 0 30px rgba(33, 150, 243, 0.1); }
  50% { border-color: rgba(0, 229, 255, 0.5); box-shadow: inset 0 0 50px rgba(0, 229, 255, 0.2); }
  100% { border-color: rgba(42, 114, 214, 0.2); box-shadow: inset 0 0 30px rgba(33, 150, 243, 0.1); }
}
#orbit-container { animation: border-pulse 4s infinite ease-in-out; }

/* ── Grade badge ──────────────────────────────────────── */
.grade-s { color: #ffd700 !important; font-family: 'Orbitron', sans-serif; font-weight: 900; text-shadow: 0 0 10px #ffd700 !important; }
.grade-a { color: #00e676 !important; font-family: 'Orbitron', sans-serif; font-weight: 900; text-shadow: 0 0 10px #00e676 !important; }
.grade-b { color: #2196f3 !important; font-family: 'Orbitron', sans-serif; font-weight: 900; }
.grade-c { color: #ffab40 !important; font-family: 'Orbitron', sans-serif; font-weight: 900; }
.grade-f { color: #ff1744 !important; font-family: 'Orbitron', sans-serif; font-weight: 900; text-shadow: 0 0 10px #ff1744 !important; }
"""

HEADER_HTML = """
<div style="text-align:center; padding:26px 0 10px 0; user-select:none;">
  <div style="font-family:'Orbitron',sans-serif; font-size:2.6rem; font-weight:900;
              background:linear-gradient(130deg,#00e5ff 0%,#2196f3 45%,#7c4dff 100%);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
              letter-spacing:.12em; margin-bottom:6px; line-height:1.1;">
    🛰&nbsp; ORBITAL COMMAND
  </div>
  <div style="font-family:'Orbitron',sans-serif; font-size:.72rem; color:#4a6888;
              letter-spacing:.36em; text-transform:uppercase;">
    Autonomous Constellation Orchestrator &nbsp;·&nbsp; OpenEnv Hackathon
  </div>
</div>
"""

API_KEY_HTML = """
<div class="api-key-note">
  <b style="color:#00e5ff;">Three ways to set your API keys (pick one):</b><br>
  &nbsp;1. <b>Type in the fields below</b> — stored in memory for this session only.<br>
  &nbsp;2. <b>Create a <code>.env</code> file</b> in the project root (copy <code>.env.example</code>).<br>
  &nbsp;3. <b>Set OS environment variables</b> before launching the app.
  <br><br>
  <b style="color:#4ade80;">&#10024; Google Gemini (recommended — free tier available):</b><br>
  &nbsp;→ <code>GOOGLE_API_KEY</code> &nbsp;&nbsp; Get it free at
  <code>aistudio.google.com/app/apikey</code><br>
  &nbsp;&nbsp;&nbsp; Models: <code>gemini-2.0-flash</code> (fast/cheap) &nbsp;·&nbsp;
  <code>gemini-1.5-pro</code> (smartest)<br><br>
  <b style="color:#ffab40;">Other providers:</b><br>
  &nbsp;• HuggingFace → <code>huggingface.co/settings/tokens</code> (READ access)<br>
  <br>
  <span style="color:#546e8a;">Keys are <b>never</b> written to disk from this UI.
  They are only held in Python's <code>os.environ</code> for the current process.</span>
</div>
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _make_state() -> Dict:
    return {
        "env":        None, "agent":    None, "obs":      None,
        "task_id":    1,    "backend":  "rule_based",
        "running":    False,"step":     0,    "score":    0.0,
        "action_log": [],   "reward_log": [], "done":     False,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_3d(obs: Observation, step: int, score: float) -> Any:
    return generate_orbit_3d(
        [s.model_dump() for s in obs.satellites],
        [s.model_dump() for s in obs.ground_stations],
        obs.isl_topology,
        [r.model_dump() for r in obs.imaging_requests],
        step=step, score=score,
    )


def _format_telemetry_md(obs: Observation) -> str:
    rows = ["| Satellite | Pos | Bat % | Stor % | Fuel % | Temp ° | Health | Mode | LoS |",
            "|-----------|-----|-------|--------|--------|--------|--------|------|-----|"]
    for sat in obs.satellites:
        def sym(v, lo, hi): return "🟢" if v < lo else ("🟡" if v < hi else "🔴")
        b  = sym(sat.battery_level,  50, 80)   # green if high is good → invert
        b  = "🟢" if sat.battery_level>=50 else ("🟡" if sat.battery_level>=20 else "🔴")
        st = "🟢" if sat.storage_used<70  else ("🟡" if sat.storage_used<90  else "🔴")
        tp = "🟢" if sat.thermal_level<60 else ("🟡" if sat.thermal_level<80 else "🔴")
        hp = "🟢" if sat.health_index>=90 else ("🟡" if sat.health_index>=70 else "🔴")
        los = sat.line_of_sight_to_ground or "—"
        rows.append(
            f"| `{sat.sat_id}` | {sat.orbital_position}° | {b}{sat.battery_level:.0f} | "
            f"{st}{sat.storage_used:.0f} | {sat.fuel_remaining:.0f} | "
            f"{tp}{sat.thermal_level:.0f} | {hp}{sat.health_index:.0f} | "
            f"{sat.mode.value} | {los} |"
        )
    return "\n".join(rows)


def _format_requests_md(obs: Observation) -> str:
    if not obs.imaging_requests:
        return "✅ No pending requests — all fulfilled or expired."
    rows = ["| ID | Target | Eff. Reward | Priority | Deadline | Status |",
            "|----|--------|-------------|----------|----------|--------|"]
    for req in sorted(obs.imaging_requests,
                      key=lambda r: {"EMERGENCY": 0, "URGENT": 1, "ROUTINE": 2}[r.priority]):
        icon = {"EMERGENCY": "🚨", "URGENT": "⚠️", "ROUTINE": "📋"}.get(req.priority, "")
        dl   = f"min {req.deadline_minute}" if req.deadline_minute else "—"
        assigned = f"→ {req.assigned_to}" if req.assigned_to else "unassigned"
        rows.append(f"| `{req.id}` | {req.target_deg}° | **{req.effective_reward:.0f}** | "
                    f"{icon}{req.priority} | {dl} | {assigned} |")
    return "\n".join(rows)


def _format_events_md(obs: Observation) -> str:
    if not obs.active_events:
        return "*No active events — nominal operations.*"
    parts = []
    for ev in obs.active_events:
        icon = {"solar_flare": "☀️⚡", "ground_outage": "📡🌨️",
                "priority_escalation": "🚨", "atmospheric_drag": "🌫️",
                "bandwidth_congestion": "📡🔴"}.get(ev.event_type.value, "⚠️")
        parts.append(f"- {icon} **{ev.event_type.value.replace('_',' ').upper()}**: "
                     f"{ev.description} *(+{ev.steps_remaining} steps, ×{ev.magnitude:.2f})*")
    return "\n".join(parts)


def _make_reward_plot(reward_log: List[float]) -> Optional[Any]:
    """Return a Plotly figure or None if plotly not installed."""
    if not _PLOTLY or not reward_log:
        return None
    cumulative = []
    running = 0.0
    for r in reward_log:
        running += r
        cumulative.append(running)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=reward_log, mode="lines", name="Step Reward",
        line=dict(color="#2196f3", width=1.2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
    ))
    fig.add_trace(go.Scatter(
        y=cumulative, mode="lines", name="Cumulative",
        line=dict(color="#00e5ff", width=2, dash="dot"),
        yaxis="y2",
    ))
    fig.update_layout(
        paper_bgcolor="#0a1220", plot_bgcolor="#070d1a",
        font=dict(family="JetBrains Mono", color="#7aa0c0", size=10),
        margin=dict(l=40, r=10, t=30, b=30),
        height=180,
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9, color="#4a6888")),
        xaxis=dict(gridcolor="#111e33", title="Step", title_font_size=9, tickfont_size=8),
        yaxis=dict(gridcolor="#111e33", title="Δ Reward", title_font_size=9, tickfont_size=8),
        yaxis2=dict(overlaying="y", side="right", title="Cumulative",
                    title_font_size=9, tickfont_size=8,
                    gridcolor="rgba(0,229,255,0.05)", tickcolor="#00e5ff"),
        hovermode="x unified",
    )
    return fig


def _build_sparkline(reward_log: List[float]) -> str:
    if not reward_log:
        return "*No data yet.*"
    recent = reward_log[-50:]
    mn, mx = min(recent), max(recent)
    rng = mx - mn or 1
    bars = "▁▂▃▄▅▆▇█"
    sl = "".join(bars[min(7, int((v - mn) / rng * 7.99))] for v in recent)
    avg = sum(recent) / len(recent)
    total = sum(reward_log)
    return (f"`{sl}`\n\n"
            f"Total **{total:+.1f}** &nbsp;|&nbsp; Avg {avg:+.1f}/step "
            f"&nbsp;|&nbsp; Min {mn:+.1f} &nbsp;|&nbsp; Max {mx:+.1f}")


def _format_leaderboard_md(task_id: Optional[int] = None) -> str:
    entries = get_leaderboard(task_id=task_id, top_n=15)
    if not entries:
        return "*No entries yet — run an episode to start the leaderboard.*"
    rows = ["| # | Name | Model | Task | Score | Grade | Time |",
            "|---|------|-------|------|-------|-------|------|"]
    for e in entries:
        rows.append(f"| **{e.rank}** | {e.agent_name} | `{e.model_name}` | "
                    f"T{e.task_id} | {e.normalized_score:.3f} | **{e.grade}** | {e.timestamp[:10]} |")
    return "\n".join(rows)


def _parse_task_id(task_choice: str) -> int:
    m = re.search(r"Task (\d)", task_choice)
    return int(m.group(1)) if m else 1


# ---------------------------------------------------------------------------
# API key injection (UI → os.environ)
# ---------------------------------------------------------------------------

def _apply_api_keys(gemini_key: str, hf_token: str) -> str:
    """Write non-empty UI key fields into os.environ for the current process."""
    changed = []
    if gemini_key.strip() and not gemini_key.strip().startswith("AIza..."):
        os.environ["GOOGLE_API_KEY"]   = gemini_key.strip()
        os.environ["GEMINI_API_KEY"]   = gemini_key.strip()  # both aliases
        changed.append("GOOGLE_API_KEY")
    if hf_token.strip() and hf_token.strip() != "hf_...":
        os.environ["HF_TOKEN"] = hf_token.strip()
        changed.append("HF_TOKEN")
    if changed:
        return f"✅ Saved to environment: {', '.join(changed)}"
    return "ℹ️ No new keys entered — existing environment kept."


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

_EMPTY_OUTPUTS = ("", "", "", "", "", "0", "0.0", "0/0", None, "*No data.*", "")


def on_reset(task_choice: str, backend_choice: str, decentralized: bool, state: Dict) -> Tuple:
    task_id = _parse_task_id(task_choice)
    backend = BACKEND_OPTIONS.get(backend_choice, "rule_based")

    try:
        env   = OrbitalEnv(task_id=task_id, seed=42, events_enabled=True)
        obs, info = env.reset()
        agent = create_agent(backend, decentralized=decentralized)
    except Exception as ex:
        return (state, f"<div style='color:#ff1744; padding:40px; font-family:monospace;'>"
                f"❌ Reset failed:<br><pre>{ex}</pre></div>",
                *_EMPTY_OUTPUTS)

    state.update({"env": env, "agent": agent, "obs": obs,
                  "task_id": task_id, "backend": backend,
                  "running": False, "step": 0, "score": 0.0,
                  "action_log": [], "reward_log": [], "done": False})

    plot_3d = _obs_to_3d(obs, 0, 0.0)
    plot = _make_reward_plot([])
    return (
        state,
        plot_3d,
        _format_telemetry_md(obs),
        _format_requests_md(obs),
        _format_events_md(obs),
        "**Episode reset.** Press ▶ STEP or ⚡ RUN to begin.",
        obs.task_description,
        "0", "0.0", f"0 / {len(obs.imaging_requests)}",
        plot, _build_sparkline([]), "",
    )


def on_step(state: Dict) -> Tuple:
    if state.get("done"):
        return _build_outputs(state, extra_log_prefix="🏁 Episode already finished. Press RESET.\n\n")

    env: OrbitalEnv    = state.get("env")
    agent: BaseAgent   = state.get("agent")
    obs: Observation   = state.get("obs")
    if env is None or obs is None:
        return (state, "<div style='color:#ffab40;padding:40px;'>⚠ Press RESET first.</div>",
                *_EMPTY_OUTPUTS)

    try:
        action = agent.act(obs)
        obs_new, reward, terminated, truncated, info = env.step(action)
    except Exception as ex:
        return (state, f"<div style='color:#ff1744;padding:20px;'><pre>{ex}</pre></div>",
                *_EMPTY_OUTPUTS)

    if isinstance(action, list):
        for a in action:
            agent.record_step(state["step"], a, reward)
    else:
        agent.record_step(state["step"], action, reward)

    state["obs"]      = obs_new
    state["step"]    += 1
    state["score"]   += reward
    state["reward_log"].append(reward)
    state["done"]     = terminated or truncated

    ars = info.get("action_results", [])
    if isinstance(action, list):
        for i, act in enumerate(action):
            msg = ars[i].get("message", "") if i < len(ars) else ""
            think = (act.reasoning or "")[:130]
            log   = (f"[{state['step']:03d}] {act.action_type.value:<22} → {act.target_sat_id:<14}"
                     f"  {reward:+7.1f} pts\n"
                     f"   💭 {think}\n"
                     f"   ↳  {msg}\n")
            state["action_log"].append(log)
    else:
        ar    = ars[0] if ars else info.get("action_result", {})
        msg   = ar.get("message", "")
        think = (action.reasoning or "")[:130]
        log   = (f"[{state['step']:03d}] {action.action_type.value:<22} → {action.target_sat_id:<14}"
                 f"  {reward:+7.1f} pts\n"
                 f"   💭 {think}\n"
                 f"   ↳  {msg}\n")
        state["action_log"].append(log)

    if state["done"]:
        result = env.get_episode_result()
        submit_result(result, agent_name=agent.name,
                      model_name=getattr(agent, "model", "RuleBased"))
        prefix = (f"🏁 EPISODE COMPLETE  —  Grade: {result.grade}  "
                  f"Score: {result.final_score:.1f}  "
                  f"({result.satellites_survived}/{result.total_satellites} sats alive)\n\n")
    else:
        prefix = ""

    return _build_outputs(state, extra_log_prefix=prefix)


def on_run(state: Dict, max_steps: int) -> Tuple:
    for _ in range(min(int(max_steps), 250)):
        outputs = on_step(state)
        state = outputs[0]
        if state.get("done"):
            break
    return outputs


def _build_outputs(state: Dict, extra_log_prefix: str = "") -> Tuple:
    obs   = state.get("obs")
    if obs is None:
        return (state, None, *_EMPTY_OUTPUTS[1:]) # handle gr.Plot correctly

    plot_3d   = _obs_to_3d(obs, state["step"], state["score"])
    tel_md    = _format_telemetry_md(obs)
    req_md    = _format_requests_md(obs)
    ev_md     = _format_events_md(obs)
    log_txt   = extra_log_prefix + "\n".join(reversed(state["action_log"][-40:]))
    obj_txt   = obs.task_description
    step_txt  = str(state["step"])
    score_txt = f"{state['score']:.1f}"
    req_txt   = (f"{len(obs.completed_requests)} / "
                 f"{len(obs.imaging_requests) + len(obs.completed_requests)}")
    plot      = _make_reward_plot(state["reward_log"])
    spark     = _build_sparkline(state["reward_log"])
    lb_txt    = _format_leaderboard_md(state["task_id"]) if state["done"] else ""

    return (state, plot_3d, tel_md, req_md, ev_md,
            log_txt, obj_txt, step_txt, score_txt, req_txt,
            plot, spark, lb_txt)


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ORBITAL COMMAND — Mission Control") as demo:

        state = gr.State(_make_state())

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML(HEADER_HTML)

        # ── API Key accordion (prominent, near top) ─────────────────────────
        with gr.Accordion("🔐  API Keys & LLM Configuration", open=False):
            gr.HTML(API_KEY_HTML)
            gr.HTML('<div style="font-family:JetBrains Mono,monospace; font-size:.72rem; '
                    'color:#4ade80; padding:6px 0 8px 0;">'
                    '✨ Google Gemini — recommended (free tier) &nbsp;·&nbsp; '
                    'get key at <b>aistudio.google.com/app/apikey</b></div>')
            with gr.Row():
                gemini_key = gr.Textbox(
                    label="✨ Google API Key  (Gemini)",
                    placeholder="AIza...",
                    type="password",
                    value=os.environ.get("GOOGLE_API_KEY",
                            os.environ.get("GEMINI_API_KEY", "")),
                    scale=5,
                )
                hf_token = gr.Textbox(
                    label="HuggingFace Token",
                    placeholder="hf_...",
                    type="password",
                    value=os.environ.get("HF_TOKEN", ""),
                    scale=5,
                )
            with gr.Row():
                model_txt = gr.Textbox(
                    label="Model name",
                    placeholder="gemini-2.0-flash",
                    value="gemini-2.0-flash",
                    scale=4,
                )
                save_keys_btn  = gr.Button("💾  Save Keys to Session", variant="primary", scale=2)
                key_status_txt = gr.Textbox(label="Status", interactive=False, scale=4, max_lines=1)
            save_keys_btn.click(
                _apply_api_keys,
                inputs=[gemini_key, hf_token],
                outputs=[key_status_txt],
            )

        # ── Controls ────────────────────────────────────────────────────────
        with gr.Row():
            task_dd    = gr.Dropdown(choices=list(TASK_NAMES.values()),
                                     value=list(TASK_NAMES.values())[0],
                                     label="📋  Task", scale=3)
            backend_dd = gr.Dropdown(choices=list(BACKEND_OPTIONS.keys()),
                                     value=list(BACKEND_OPTIONS.keys())[0],
                                     label="🤖  LLM Backend", scale=3)
            decentralized_cb = gr.Checkbox(label="🔗 Swarm Mode", value=False, scale=1)
            reset_btn  = gr.Button("🔄  RESET",  variant="primary", scale=1)
            step_btn   = gr.Button("▶  STEP",   variant="primary", scale=1)
            run_slider = gr.Slider(5, 200, value=50, step=5, label="Auto-steps", scale=2)
            run_btn    = gr.Button("⚡  RUN",    variant="primary", scale=1)

        # ── Stats bar ────────────────────────────────────────────────────────
        with gr.Row():
            step_out  = gr.Textbox("0",   label="📌  Step",      interactive=False,
                                   scale=1, max_lines=1, elem_id="stat-step")
            score_out = gr.Textbox("0.0", label="⭐  Score",     interactive=False,
                                   scale=1, max_lines=1, elem_id="stat-score")
            req_out   = gr.Textbox("0/0", label="📷  Requests",  interactive=False,
                                   scale=1, max_lines=1, elem_id="stat-req")
            obj_out   = gr.Textbox("",    label="📋  Objective", interactive=False,
                                   scale=5, max_lines=2)

        # ── Main split ──────────────────────────────────────────────────────
        with gr.Row():
            # Left — orbit visualiser
            with gr.Column(scale=5, min_width=540):
                orbit_plot = gr.Plot(label="", elem_id="orbit-container")

            # Right — panels
            with gr.Column(scale=3):
                with gr.Tab("🛰 Telemetry"):
                    telemetry_md = gr.Markdown("*Waiting for reset…*")
                with gr.Tab("🎯 Requests"):
                    requests_md = gr.Markdown("*Waiting for reset…*")
                with gr.Tab("⚡ Events"):
                    events_md = gr.Markdown("*Nominal.*")

        # ── Reward chart + action log ────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📈  Reward History")
                if _PLOTLY:
                    reward_plot = gr.Plot(label="")
                else:
                    reward_plot = gr.Markdown("*Install plotly for chart: `pip install plotly`*")
                reward_spark = gr.Markdown("*No data.*")

            with gr.Column(scale=3):
                gr.Markdown("### 📜  Action Log")
                action_log = gr.Textbox(
                    value="", lines=14, max_lines=18,
                    interactive=False, elem_id="action-log", label="",
                )

        # ── Leaderboard ──────────────────────────────────────────────────────
        with gr.Accordion("🏆  Leaderboard", open=False):
            lb_md = gr.Markdown("*Run an episode to populate this.*")
            with gr.Row():
                lb_refresh = gr.Button("🔄  Refresh", scale=1)
                lb_task_filter = gr.Dropdown(
                    choices=["All Tasks"] + [f"Task {i}" for i in range(1, 6)],
                    value="All Tasks", label="Filter by task", scale=2,
                )
            def _refresh_lb(task_filter: str):
                tid = None if task_filter == "All Tasks" else int(task_filter[-1])
                return _format_leaderboard_md(tid)
            lb_refresh.click(_refresh_lb, inputs=[lb_task_filter], outputs=[lb_md])

        # ── Footer & JS Injection ───────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:14px 0 6px 0;
                    font-size:.68rem; color:rgba(255,255,255,0.4); font-family:JetBrains Mono,monospace;">
          OpenEnv-Orbital-Command &nbsp;·&nbsp;
          <a href="https://github.com" style="color:#2196f3;">GitHub</a> &nbsp;·&nbsp;
          Deterministic LEO Simulation — pure numpy, runs on 2 vCPU / 4 GB RAM
        </div>
        <script>
        // Auto-Spin the Plotly Earth Globe continuously!
        if(!window.spinInterval) {
            window.spinInterval = setInterval(() => {
                const el = document.getElementById("orbit-container");
                if (el) {
                    const plots = el.getElementsByClassName("js-plotly-plot");
                    if (plots.length > 0 && plots[0].layout && plots[0].layout.geo) {
                        const p = plots[0];
                        let lon = p.layout.geo.center.lon || 0;
                        lon = (lon + 0.3) % 360;  // Smoothly increment longitude
                        Plotly.relayout(p, {"geo.center.lon": lon});
                    }
                }
            }, 50); // Updates every ~50ms
        }
        </script>
        """)

        # ── Output list (must match every callback's return) ──────────────
        OUTPUTS = [
            state, orbit_plot,
            telemetry_md, requests_md, events_md,
            action_log, obj_out,
            step_out, score_out, req_out,
            reward_plot, reward_spark, lb_md,
        ]

        reset_btn.click(on_reset, inputs=[task_dd, backend_dd, decentralized_cb, state], outputs=OUTPUTS)
        step_btn.click(on_step,   inputs=[state],                       outputs=OUTPUTS)
        run_btn.click( on_run,    inputs=[state, run_slider],            outputs=OUTPUTS)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orbital Command — Mission Control UI")
    parser.add_argument("--port",    type=int, default=7860)
    parser.add_argument("--share",   action="store_true", help="Create a public Gradio link")
    parser.add_argument("--backend", default="rule_based",
                        choices=["rule_based", "gemini", "huggingface"])
    args = parser.parse_args()

    print(f"\n{'═'*55}")
    print(f"  🛰  ORBITAL COMMAND — launching on port {args.port}")
    print(f"{'═'*55}")
    print(f"  Default backend: {args.backend}")
    print(f"  API keys loaded from environment:")
    for key in ["GOOGLE_API_KEY", "HF_TOKEN"]:
        val = os.environ.get(key, "")
        masked = (val[:6] + "..." + val[-3:]) if len(val) > 12 else ("(not set)" if not val else "(set)")
        star = " ★" if key == "GOOGLE_API_KEY" else ""
        print(f"    {key}: {masked}{star}")
    print(f"{'\u2550'*55}\n")

    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share, css=CSS)
