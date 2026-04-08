"""
OpenEnv-Orbital-Command | physics.py
Lightweight 1-D orbital mechanics on a 0–359° ring.
No external physics engine required — pure numpy arithmetic.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Orbital constants
# ---------------------------------------------------------------------------

ORBIT_DEGREES       = 360
DEGREES_PER_STEP    = 2      # 2° per simulation step
ISL_RANGE_DEG       = 45     # Max inter-satellite link range
STATION_FOV_DEG     = 15     # Default ground station field-of-view half-angle

# Eclipse zone — Earth's shadow (shadow cone, simplified to arc)
ECLIPSE_START_DEG   = 170
ECLIPSE_END_DEG     = 350

# Power & thermal
SOLAR_CHARGE_RATE   = 4.0    # % battery per step in sunlight (passive)
PASSIVE_DRAIN_RATE  = 1.0    # % battery per step always
ACTIVE_EXTRA_DRAIN  = 1.5    # additional drain while in non-sleep mode
ECLIPSE_HEATING     = -3.0   # thermal change per step in eclipse (cooling)
SUNLIGHT_HEATING    = 2.5    # thermal change per step in sunlight (active mode)
SLEEP_HEATING       = 0.5    # reduced thermal change while sleeping
THERMAL_VENT_DELTA  = -15.0  # forced thermal reduction when venting

# Action costs
ACTION_BATTERY_COST = {
    "capture_image":        8.0,
    "downlink_data":        5.0,
    "inter_satellite_link": 10.0,  # per hop (sender) + 6% for relay
    "isl_relay_pass":       6.0,
    "station_keeping":      12.0,
    "emergency_transmit":   15.0,
    "thermal_vent":         0.0,
    "sleep_mode":           0.0,
}
ACTION_FUEL_COST = {
    "station_keeping":      8.0,
    "inter_satellite_link": 0.5,
    "capture_image":        0.2,  # gimbal slew
}
ACTION_STORAGE_DELTA = {
    "capture_image":        +15.0,  # % of total
    "downlink_data":        "CLEAR",
    "inter_satellite_link": "TRANSFER",
}


# ---------------------------------------------------------------------------
# Position utilities
# ---------------------------------------------------------------------------

def normalize(pos: float) -> int:
    """Wrap any position to [0, 359]."""
    return int(pos % ORBIT_DEGREES)


def advance_position(pos: int, steps: int = 1, drift_deg: float = 0.0) -> int:
    """Move satellite forward by `steps` steps (plus accumulated drift)."""
    return normalize(pos + steps * DEGREES_PER_STEP + drift_deg)


def angular_distance(a: int, b: int) -> int:
    """Minimum angular separation on the ring."""
    d = abs(a - b) % ORBIT_DEGREES
    return min(d, ORBIT_DEGREES - d)


def steps_to_position(current: int, target: int,
                       speed: int = DEGREES_PER_STEP) -> int:
    """Forward-only steps until satellite arrives at target."""
    forward = (target - current) % ORBIT_DEGREES
    return max(1, forward // speed)


def predict_position(current: int, steps: int) -> int:
    return normalize(current + steps * DEGREES_PER_STEP)


# ---------------------------------------------------------------------------
# Sunlight & eclipse
# ---------------------------------------------------------------------------

def is_in_sunlight(pos: int) -> bool:
    """True when satellite is NOT in Earth's shadow."""
    p = normalize(pos)
    return not (ECLIPSE_START_DEG <= p <= ECLIPSE_END_DEG)


def steps_until_eclipse(pos: int, speed: int = DEGREES_PER_STEP) -> Optional[int]:
    """Steps until satellite enters shadow zone. None if already in shadow."""
    if not is_in_sunlight(pos):
        return None
    for s in range(ORBIT_DEGREES // speed + 2):
        if not is_in_sunlight(normalize(pos + s * speed)):
            return s
    return None


def steps_until_sunlight(pos: int, speed: int = DEGREES_PER_STEP) -> Optional[int]:
    """Steps until satellite exits shadow zone. None if already in sunlight."""
    if is_in_sunlight(pos):
        return None
    for s in range(ORBIT_DEGREES // speed + 2):
        if is_in_sunlight(normalize(pos + s * speed)):
            return s
    return None


def eclipse_duration_steps(pos: int, speed: int = DEGREES_PER_STEP) -> int:
    """Total shadow duration in steps starting from current pos (or current shadow)."""
    entry = steps_until_eclipse(pos, speed)
    if entry is None:  # already in shadow
        return steps_until_sunlight(pos, speed) or 0
    exit_ = steps_until_sunlight(normalize(pos + entry * speed), speed) or 0
    return exit_


# ---------------------------------------------------------------------------
# Ground station line-of-sight
# ---------------------------------------------------------------------------

def has_line_of_sight(sat_pos: int, station_pos: int,
                      fov_deg: int = STATION_FOV_DEG) -> bool:
    return angular_distance(sat_pos, station_pos) <= fov_deg


def steps_until_los(sat_pos: int, station_pos: int,
                    fov_deg: int = STATION_FOV_DEG,
                    speed: int = DEGREES_PER_STEP) -> Optional[int]:
    """Steps until next LoS contact with ground station."""
    if has_line_of_sight(sat_pos, station_pos, fov_deg):
        return 0
    for s in range(1, ORBIT_DEGREES // speed + 2):
        p = normalize(sat_pos + s * speed)
        if has_line_of_sight(p, station_pos, fov_deg):
            return s
    return None


def los_window_duration(sat_pos: int, station_pos: int,
                        fov_deg: int = STATION_FOV_DEG,
                        speed: int = DEGREES_PER_STEP) -> Tuple[int, int]:
    """
    Returns (steps_to_start, window_length) for next LoS window.
    steps_to_start=0 means currently in LoS.
    """
    start = steps_until_los(sat_pos, station_pos, fov_deg, speed) or 0
    entry_pos = normalize(sat_pos + start * speed)
    length = 0
    for s in range(ORBIT_DEGREES // speed + 2):
        if has_line_of_sight(normalize(entry_pos + s * speed), station_pos, fov_deg):
            length += 1
        elif length > 0:
            break
    return (start, max(length, 1))


# ---------------------------------------------------------------------------
# Inter-satellite link topology
# ---------------------------------------------------------------------------

def sat_positions_to_isl_graph(sat_positions: Dict[str, int],
                                isl_range: int = ISL_RANGE_DEG) -> Dict[str, List[str]]:
    """Build adjacency map of ISL-reachable satellites."""
    ids = list(sat_positions.keys())
    graph: Dict[str, List[str]] = {sid: [] for sid in ids}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if angular_distance(sat_positions[a], sat_positions[b]) <= isl_range:
                graph[a].append(b)
                graph[b].append(a)
    return graph


def find_min_hop_path(graph: Dict[str, List[str]],
                      src: str,
                      station_pos: int,
                      sat_positions: Dict[str, int],
                      fov_deg: int = STATION_FOV_DEG) -> Optional[List[str]]:
    """BFS relay path from src satellite to a ground station. Returns ordered sat IDs."""
    from collections import deque

    # Direct LoS?
    if has_line_of_sight(sat_positions[src], station_pos, fov_deg):
        return [src]

    queue: deque[Tuple[str, List[str]]] = deque([(src, [src])])
    visited = {src}

    while queue:
        node, path = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor in visited:
                continue
            new_path = path + [neighbor]
            visited.add(neighbor)
            if has_line_of_sight(sat_positions[neighbor], station_pos, fov_deg):
                return new_path
            queue.append((neighbor, new_path))

    return None  # No path found


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def battery_penalty(level: float) -> float:
    """
    Non-linear battery penalty:
      ≥50%  → 0
      20–50%→ linear 0→2
      <20%  → exponential (hardware damage regime)
    """
    if level >= 50.0:
        return 0.0
    if level >= 20.0:
        return 2.0 * (50.0 - level) / 30.0
    return 10.0 * float(np.exp(-(level / 5.0)))


def thermal_penalty(level: float, max_safe: float = 70.0,
                    critical: float = 100.0) -> float:
    if level <= max_safe:
        return 0.0
    if level <= critical:
        return 3.0 * (level - max_safe) / (critical - max_safe)
    return 8.0  # Critical failure


def latency_penalty(deadline_minute: Optional[int],
                    current_minute: int,
                    fulfilled: bool = False) -> float:
    """Penalty for missing or approaching a deadline."""
    if deadline_minute is None or fulfilled:
        return 0.0
    overdue = current_minute - deadline_minute
    if overdue <= 0:
        return 0.0
    return min(50.0, 2.0 * overdue)  # Caps at 50 pts penalty
