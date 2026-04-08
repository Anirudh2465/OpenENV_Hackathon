"""
OpenEnv-Orbital-Command | agent/prompt_builder.py

Builds structured, LLM-optimised prompts from Observation objects.
Uses a ReAct-style format: [Observe → Think → Act] loop.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.models import Observation, Action, ActionType
from env.physics import (
    steps_until_eclipse, steps_until_sunlight, los_window_duration,
    is_in_sunlight, ECLIPSE_START_DEG, ECLIPSE_END_DEG
)


SYSTEM_PROMPT = """You are ORBITAL-COMMAND, an autonomous AI orchestrator for a Low-Earth Orbit satellite constellation.

Your mission is to schedule satellite operations to maximize data revenue while preventing battery failure, storage overflow, and thermal damage.

## Resources You Control
- **Battery (0–100%)**: Charges in sunlight (+4%/step) | Drains in eclipse. Hits 0% → satellite dies.
- **Storage (0–100%)**: Fills when imaging. Must downlink before overflow → data destroyed.
- **Fuel (0–100%)**: Used for station-keeping burns and ISL attitude control.
- **Thermal (0–100%)**: Rises in sunlight when active. Above 70% → penalty. Above 100% → damage.

## Available Actions (issue exactly ONE per step)
| action_type             | Key Parameters                  | Battery Cost |
|-------------------------|---------------------------------|-------------|
| capture_image           | target_sat_id, request_id       | -8%         |
| downlink_data           | target_sat_id, target_station   | -5%         |
| sleep_mode              | target_sat_id                   | 0% (charges)|
| inter_satellite_link    | target_sat_id, relay_chain[]    | -10%+relay  |
| station_keeping         | target_sat_id                   | -12%, -8fuel|
| emergency_transmit      | target_sat_id, target_station   | -15%        |
| thermal_vent            | target_sat_id                   | 0%          |

## Eclipse Zone: {eclipse_start}° → {eclipse_end}° (NO solar charging inside)
## ISL Range: Satellites within 45° of each other can cross-link.

## Reward Function
R = (Data_Downlinked × 10) - (Data_Overwritten × 20) - BatteryPenalty - ThermalPenalty
BatteryPenalty: 0 if bat≥50%, linear if 20–50%, EXPONENTIAL if <20%, DEATH at 0%.

## CRITICAL RULES
1. NEVER let battery drop to 0% — the satellite dies and the episode ends catastrophically.
2. Downlink before storage hits 90% or data will be permanently overwritten.
3. You can only capture an image when within 15° of the target.
4. Sleep mode charges the battery — use it proactively.
5. ALWAYS include a reasoning field explaining your strategic thinking.
""".format(eclipse_start=ECLIPSE_START_DEG, eclipse_end=ECLIPSE_END_DEG)


def build_observation_prompt(obs: Observation, step_history: Optional[List[dict]] = None) -> str:
    """Build a human-readable observation prompt for the LLM."""
    lines = []

    # Header
    lines.append(f"# STEP {obs.step_number} | Task: {obs.task_name} | Orbit Minute: {obs.current_orbit_minute}")
    lines.append(f"**Episode Score:** {obs.episode_score:.1f} / {obs.max_possible_score:.0f}  |  "
                 f"Requests: {len(obs.completed_requests)} done, {len(obs.failed_requests)} failed, {len(obs.imaging_requests)} pending\n")

    # Satellite telemetry
    lines.append("## 🛰️ SATELLITE TELEMETRY")
    lines.append("| Satellite | Pos° | Battery | Storage | Fuel | Thermal | Health | Sunlight | LoS | Mode | Pending GB |")
    lines.append("|-----------|------|---------|---------|------|---------|--------|----------|-----|------|------------|")

    for sat in obs.satellites:
        sun_icon = "☀️" if sat.in_sunlight else "🌑"
        bat_icon = "🟢" if sat.battery_level >= 50 else ("🟡" if sat.battery_level >= 20 else "🔴")
        stor_icon = "🟢" if sat.storage_used < 70 else ("🟡" if sat.storage_used < 90 else "🔴")
        therm_icon = "🟢" if sat.thermal_level < 60 else ("🟡" if sat.thermal_level < 80 else "🔴")
        los = sat.line_of_sight_to_ground or "—"

        # Predict eclipse/sunlight timing
        timing_hint = ""
        if sat.in_sunlight:
            steps_to_dark = steps_until_eclipse(sat.orbital_position)
            if steps_to_dark is not None:
                timing_hint = f"(eclipse in {steps_to_dark}s)"
        else:
            steps_to_light = steps_until_sunlight(sat.orbital_position)
            if steps_to_light is not None:
                timing_hint = f"(sun in {steps_to_light}s)"

        lines.append(
            f"| {sat.sat_id} | {sat.orbital_position}° | "
            f"{bat_icon}{sat.battery_level:.0f}% | "
            f"{stor_icon}{sat.storage_used:.0f}% | "
            f"{sat.fuel_remaining:.0f}% | "
            f"{therm_icon}{sat.thermal_level:.0f}° | "
            f"{sat.health_index:.0f}% | "
            f"{sun_icon}{timing_hint} | "
            f"{los} | "
            f"{sat.mode.value} | "
            f"{sat.pending_data_gb:.2f} GB |"
        )

    # ISL topology
    lines.append("\n## 🔗 ISL TOPOLOGY (within 45°)")
    if obs.isl_topology:
        for src, neighbors in obs.isl_topology.items():
            if neighbors:
                lines.append(f"  {src} ↔ {', '.join(neighbors)}")
    else:
        lines.append("  No ISL connections currently feasible.")

    # Ground stations
    lines.append("\n## 📡 GROUND STATIONS")
    lines.append("| Station | Position | FoV | Bandwidth | Status | Received GB |")
    lines.append("|---------|----------|-----|-----------|--------|-------------|")
    for stn in obs.ground_stations:
        status_icon = "🟢 online" if stn.status == "online" else ("⚫ offline" if stn.status == "offline" else "🟡 congested")
        lines.append(
            f"| {stn.station_id} | {stn.position_deg}° | ±{stn.fov_deg}° | "
            f"{stn.bandwidth_gbps} Gbps | {status_icon} | {stn.total_received_gb:.2f} GB |"
        )
        # Add LoS windows for each satellite
        for sat in obs.satellites:
            if stn.status == "online":
                steps_to, window_len = los_window_duration(sat.orbital_position, stn.position_deg, stn.fov_deg)
                los_now = sat.line_of_sight_to_ground == stn.station_id
                if los_now:
                    lines.append(f"    ↳ {sat.sat_id}: **IN LoS NOW** (window: {window_len} more steps)")
                elif steps_to <= 20:
                    lines.append(f"    ↳ {sat.sat_id}: LoS in {steps_to} steps ({window_len} step window)")

    # Imaging requests
    lines.append("\n## 🎯 PENDING IMAGING REQUESTS")
    if not obs.imaging_requests:
        lines.append("  ✅ No pending requests.")
    else:
        lines.append("| ID | Target° | Reward | Priority | Deadline | Size GB | Description |")
        lines.append("|----|---------|--------|----------|----------|---------|-------------|")
        for req in sorted(obs.imaging_requests, key=lambda r: {"EMERGENCY": 0, "URGENT": 1, "ROUTINE": 2}[r.priority]):
            dl = f"min {req.deadline_minute}" if req.deadline_minute else "none"
            priority_icon = {"EMERGENCY": "🚨", "URGENT": "⚠️", "ROUTINE": "📋"}.get(req.priority, "")
            # Find nearest satellite
            nearest_sat = None
            nearest_steps = 999
            for sat in obs.satellites:
                if sat.mode.value != "dead":
                    s, _ = los_window_duration(sat.orbital_position, req.target_deg)
                    if s < nearest_steps:
                        nearest_steps = s
                        nearest_sat = sat.sat_id
            nearest_hint = f"nearest: {nearest_sat} in {nearest_steps}s" if nearest_sat else ""

            lines.append(
                f"| {req.id} | {req.target_deg}° | {req.effective_reward:.0f} | "
                f"{priority_icon}{req.priority} | {dl} | {req.data_size_gb:.1f} | "
                f"{req.target_description[:35]} [{nearest_hint}] |"
            )

    # Active events
    if obs.active_events:
        lines.append("\n## ⚡ ACTIVE STOCHASTIC EVENTS")
        for ev in obs.active_events:
            lines.append(f"  - [{ev.event_type.upper()}] {ev.description} (remaining: {ev.steps_remaining} steps, magnitude: {ev.magnitude:.2f})")

    # Recent action history
    if step_history:
        lines.append("\n## 📜 RECENT ACTIONS (last 5 steps)")
        for entry in step_history[-5:]:
            act = entry.get("action", {})
            rew = entry.get("reward", 0.0)
            lines.append(f"  Step {entry['step']}: {act.get('action_type','?')} on {act.get('target_sat_id','?')} → reward {rew:+.1f}")

    # Task description
    lines.append(f"\n## 📋 TASK OBJECTIVE\n{obs.task_description}")

    # Action request
    lines.append("""
## 🎮 YOUR TURN — Issue ONE Action

Think step-by-step:
1. Which satellite is most at risk right now? (battery, thermal, storage)
2. Which request can be captured NOW vs needs preparation?
3. Is any ground station window about to close?
4. Should any satellite sleep to recover battery before the eclipse?

Respond with ONLY valid JSON matching this schema:
```json
{
  "action_type": "<one of the action types above>",
  "target_sat_id": "<satellite ID>",
  "request_id": "<request ID, if capturing>",
  "target_station": "<station ID, if downlinking>",
  "relay_chain": ["<sat1>", "<sat2>", ...],
  "reasoning": "<your step-by-step strategic thinking>"
}
```
""")

    return "\n".join(lines)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT
