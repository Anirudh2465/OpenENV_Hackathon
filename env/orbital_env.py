"""
OpenEnv-Orbital-Command | orbital_env.py

Core gymnasium-compatible environment.

OrbitalEnv wraps the full orbital simulation:
  - 1-D ring orbital mechanics (0–359°)
  - Multi-resource budgeting: battery, storage, fuel, thermal
  - Stochastic events injected by EventEngine
  - Three resource-reward dimensions: data revenue, overwrite penalty, battery health
  - Supports 5 tasks via reset(task_id=N)
"""
from __future__ import annotations

import copy
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from .models import (
    Action, ActionResult, ActionType, EpisodeResult,
    GroundStation, ImagingRequest, Observation,
    RequestPriority, SatelliteMode, SatelliteTelemetry, StochasticEvent,
)
from .physics import (
    ACTION_BATTERY_COST, ACTION_FUEL_COST,
    ACTIVE_EXTRA_DRAIN, DEGREES_PER_STEP,
    ECLIPSE_HEATING, ISL_RANGE_DEG, PASSIVE_DRAIN_RATE,
    SLEEP_HEATING, SOLAR_CHARGE_RATE, SUNLIGHT_HEATING, THERMAL_VENT_DELTA,
    BATTERY_DEGRADATION_RATE, THERMAL_DEGRADATION_RATE,
    advance_position, battery_penalty, find_min_hop_path,
    has_line_of_sight, is_in_sunlight, latency_penalty,
    normalize, sat_positions_to_isl_graph, thermal_penalty,
)
from .events import EventEngine

# Reward coefficients
ALPHA = 10.0   # Data downlinked reward per GB
BETA  = 20.0   # Overwrite penalty per GB
DEATH_PENALTY = -500.0
TASK_COMPLETE_BONUS = 200.0


class OrbitalEnv(gym.Env):
    """
    OpenEnv-Orbital-Command simulation environment.

    Key design choices:
    - Pure Python / numpy — no external physics engine.
    - Deterministic given a seed (reproducible benchmarks).
    - Rich structured Observation/Action via Pydantic — LLM-friendly.
    - gym.Env interface: reset() / step() / render().
    """

    metadata = {"render_modes": ["ansi", "rgb_array"]}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        task_id: int = 1,
        seed: int = 42,
        max_steps: int = 200,
        events_enabled: bool = True,
        render_mode: str = "ansi",
    ):
        super().__init__()
        self.task_id = task_id
        self.seed_val = seed
        self.max_steps = max_steps
        self.events_enabled = events_enabled
        self.render_mode = render_mode

        # Populated on reset()
        self._satellites: Dict[str, Dict[str, Any]] = {}
        self._stations: Dict[str, Dict[str, Any]] = {}
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._completed: List[str] = []
        self._failed: List[str] = []
        self._step: int = 0
        self._minute: int = 0
        self._episode_score: float = 0.0
        self._action_history: List[Dict[str, Any]] = []
        self._reward_history: List[float] = []
        self._event_engine: Optional[EventEngine] = None
        self._start_time: float = 0.0
        self._episode_id: str = ""
        self._data_downlinked_gb: float = 0.0
        self._data_overwritten_gb: float = 0.0
        self._done: bool = False
        self._task_cfg: Dict[str, Any] = {}

        # gym spaces (simplified for compatibility — actual I/O is Pydantic)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Dict({})

    # ------------------------------------------------------------------
    # Public gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Observation, Dict]:
        if task_id is not None:
            self.task_id = task_id
        if seed is not None:
            self.seed_val = seed

        self._rng = np.random.default_rng(self.seed_val)
        self._episode_id = str(uuid.uuid4())[:12]
        self._start_time = time.time()
        self._step = 0
        self._minute = 0
        self._episode_score = 0.0
        self._action_history = []
        self._reward_history = []
        self._completed = []
        self._failed = []
        self._data_downlinked_gb = 0.0
        self._data_overwritten_gb = 0.0
        self._done = False

        import sys as _sys
        from pathlib import Path as _Path
        _root = str(_Path(__file__).parent.parent)
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from tasks import get_task_config
        self._task_cfg = get_task_config(self.task_id)

        # Load task config into runtime state
        self._satellites = {
            s["sat_id"]: dict(s) for s in self._task_cfg["satellites"]
        }
        self._stations = {
            s["station_id"]: dict(s) for s in self._task_cfg["stations"]
        }
        self._requests = {
            r["id"]: dict(r) for r in self._task_cfg["requests"]
        }

        # Event engine
        self._event_engine = EventEngine(
            seed=self.seed_val + self._step,
            task_difficulty=self._task_cfg.get("difficulty", 1),
            events_enabled=self.events_enabled,
        )

        obs = self._build_observation([])
        return obs, {"episode_id": self._episode_id}

    def step(self, actions: Union[Action, List[Action]]) -> Tuple[Observation, float, bool, bool, Dict]:
        """
        Execute one or multiple actions and advance the simulation by one step.

        Returns: (observation, reward, terminated, truncated, info)
        """
        if self._done:
            raise RuntimeError("Episode has ended. Call reset() before stepping.")

        reward = 0.0
        info: Dict[str, Any] = {}
        new_events: List[str] = []

        if not isinstance(actions, list):
            actions = [actions]

        # Clear inboxes BEFORE applying actions for this step
        for sat in self._satellites.values():
            sat["inbox"] = []

        # 1. Validate & apply actions
        info["action_results"] = []
        for act in actions:
            result = self._apply_action(act)
            reward += result.reward_delta
            info["action_results"].append(result.dict())
            if result.new_events_triggered:
                new_events.extend(result.new_events_triggered)

        # 2. Advance physics for all satellites
        self._tick_physics()

        # 3. Process stochastic events
        if self._event_engine:
            active = self._event_engine.tick(
                sat_ids=list(self._satellites.keys()),
                station_ids=list(self._stations.keys()),
                request_ids=[r for r in self._requests if not self._requests[r].get("done")],
                step=self._step,
            )
            self._event_engine.apply_to_state(
                self._satellites, self._stations, self._requests
            )
        else:
            active = []

        # 4. Compute step reward
        step_reward = self._compute_step_reward()
        reward += step_reward

        # 5. Check deadline failures
        for req_id, req in self._requests.items():
            if req.get("done"):
                continue
            dl = req.get("deadline_minute")
            if dl is not None and self._minute > dl:
                if req_id not in self._failed:
                    self._failed.append(req_id)
                    reward -= latency_penalty(dl, self._minute)
                    req["done"] = True  # mark expired

        # 6. Check terminal conditions / Lethal Events
        terminated = False
        for sat_id, sat in self._satellites.items():
            if sat.get("battery_level", 100.0) <= 0.0 and sat.get("mode") != "dead":
                sat["mode"] = "dead"
                reward += DEATH_PENALTY
                terminated = True
                info[f"death_{sat_id}"] = "Battery depleted"

        # Check lethal events in history
        if self._event_engine:
            for ev in self._event_engine.history:
                # If a space_debris event expired naturally (reached 0) without being cleared, it hits.
                if ev.event_type == "space_debris" and ev.steps_remaining <= 0 and ev.affected_target in self._satellites:
                    target_sat = self._satellites[ev.affected_target]
                    if target_sat.get("mode") != "dead":
                        target_sat["mode"] = "dead"
                        reward += DEATH_PENALTY
                        terminated = True
                        info[f"death_{ev.affected_target}"] = "Kessler Syndrome (Collision)"
                        
                        # Prevent triggering death repeatedly for the same historical event
                        ev.event_type = "space_debris_resolved"

        # Check task-specific completion
        if not terminated and self._check_task_complete():
            reward += TASK_COMPLETE_BONUS
            terminated = True
            info["task_complete"] = True

        self._step += 1
        self._minute += 2  # 2 minutes per step (simplified orbit)
        truncated = self._step >= self.max_steps

        self._episode_score += reward
        self._reward_history.append(reward)
        self._action_history.append({
            "step": self._step,
            "actions": [a.model_dump() for a in actions],
            "reward": reward,
        })
        self._done = terminated or truncated

        obs = self._build_observation(active)
        return obs, reward, terminated, truncated, info

    def get_episode_result(self) -> EpisodeResult:
        """Compute and return the full episode summary."""
        total = len(self._completed) + len(self._failed)
        sats_alive = sum(
            1 for s in self._satellites.values() if s.get("mode") != "dead"
        )
        norm_score = min(1.0, max(0.0, self._episode_score / max(1.0, self._task_cfg.get("max_score", 1000.0))))
        grade = self._compute_grade(norm_score)
        emergency_handled = sum(
            1 for req_id in self._completed
            if self._requests.get(req_id, {}).get("priority") == "EMERGENCY"
        )

        return EpisodeResult(
            task_id=self.task_id,
            task_name=self._task_cfg.get("name", f"Task {self.task_id}"),
            total_steps=self._step,
            final_score=round(self._episode_score, 2),
            normalized_score=round(norm_score, 4),
            grade=grade,
            data_downlinked_gb=round(self._data_downlinked_gb, 2),
            data_overwritten_gb=round(self._data_overwritten_gb, 2),
            satellites_survived=sats_alive,
            total_satellites=len(self._satellites),
            requests_fulfilled=len(self._completed),
            requests_missed=len(self._failed),
            emergency_requests_handled=emergency_handled,
            action_history=self._action_history,
            reward_history=self._reward_history,
            grader_breakdown=self._task_cfg.get("grader_breakdown_fn", lambda *a: {})(self),
            duration_seconds=round(time.time() - self._start_time, 2),
        )

    def render(self) -> str:
        """ANSI text render of current state."""
        lines = [f"\n{'='*60}", f"  ORBITAL COMMAND  |  Step {self._step} | Task {self.task_id}",
                 f"{'='*60}"]
        for sid, s in self._satellites.items():
            pos = s.get("orbital_position", 0)
            bat = s.get("battery_level", 0.0)
            stor = s.get("storage_used", 0.0)
            sun = "☀️" if is_in_sunlight(pos) else "🌑"
            los = s.get("line_of_sight_to_ground", "None")
            mode = s.get("mode", "active")
            bat_bar = "█" * int(bat / 10) + "░" * (10 - int(bat / 10))
            lines.append(f"  {sid:12s} {sun} pos={pos:3d}° bat=[{bat_bar}]{bat:5.1f}% "
                         f"stor={stor:5.1f}% mode={mode:12s} LoS={los}")
        lines.append(f"\n  Score: {self._episode_score:.1f}  |  Step: {self._step}/{self.max_steps}")
        lines.append("="*60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action) -> ActionResult:
        sat_id = action.target_sat_id
        sat = self._satellites.get(sat_id)

        if sat is None:
            return ActionResult(success=False, message=f"Unknown satellite {sat_id}", reward_delta=0.0)
        if sat.get("mode") == "dead":
            return ActionResult(success=False, message=f"{sat_id} is DEAD", reward_delta=0.0)

        # Hardware Fault Check
        if sat.get("health_index", 100.0) < 20.0 and self._rng.random() < 0.2:
            return ActionResult(success=False, message="HARDWARE FAULT: Action failed due to critical degradation.", reward_delta=-5.0)

        at = action.action_type
        reward = 0.0
        msg = ""
        new_events = []

        # --- SLEEP MODE ---
        if at == ActionType.SLEEP_MODE:
            sat["mode"] = "sleep"
            msg = f"{sat_id} entering sleep mode"

        # --- CAPTURE IMAGE ---
        elif at == ActionType.CAPTURE_IMAGE:
            req_id = action.request_id
            req = self._requests.get(req_id, {})
            if not req:
                return ActionResult(success=False, message=f"Request {req_id} not found", reward_delta=-5.0)
            if req.get("done"):
                return ActionResult(success=False, message=f"Request {req_id} already completed", reward_delta=-2.0)

            target_deg = req.get("target_deg", 0)
            if not has_line_of_sight(sat.get("orbital_position", 0), target_deg, fov_deg=15):
                return ActionResult(success=False,
                                    message=f"{sat_id} not in range of target {target_deg}°",
                                    reward_delta=-3.0)

            cost = ACTION_BATTERY_COST.get("capture_image", 8.0)
            if sat.get("battery_level", 0.0) < cost + 2:
                return ActionResult(success=False, message="Insufficient battery for capture", reward_delta=-2.0)

            # Apply costs
            sat["battery_level"] = max(0.0, sat["battery_level"] - cost)
            sat["fuel_remaining"] = max(0.0, sat.get("fuel_remaining", 100.0) - ACTION_FUEL_COST.get("capture_image", 0.2))
            new_storage = sat.get("storage_used", 0.0) + 15.0

            # Overwrite check
            overwrite_penalty = 0.0
            if new_storage > 100.0:
                overflow = new_storage - 100.0
                overwrite_gb = (overflow / 100.0) * req.get("data_size_gb", 1.0)
                self._data_overwritten_gb += overwrite_gb
                overwrite_penalty = overwrite_gb * BETA
                new_storage = 100.0

            sat["storage_used"] = new_storage
            sat["pending_data_gb"] = sat.get("pending_data_gb", 0.0) + req.get("data_size_gb", 1.0)
            sat["mode"] = "active"
            req["done"] = True
            req["captured_by"] = sat_id
            self._completed.append(req_id)

            priority_mult = {"ROUTINE": 1.0, "URGENT": 2.0, "EMERGENCY": 5.0}.get(req.get("priority", "ROUTINE"), 1.0)
            base_reward = req.get("reward", 100.0) * priority_mult
            reward = base_reward - overwrite_penalty
            msg = f"{sat_id} captured {req_id} (+{base_reward:.0f} pts)"

        # --- DOWNLINK DATA ---
        elif at == ActionType.DOWNLINK_DATA:
            station_id = action.target_station
            station = self._stations.get(station_id, {})
            if not station:
                return ActionResult(success=False, message=f"Station {station_id} not found", reward_delta=-5.0)
            if station.get("status") == "offline":
                return ActionResult(success=False, message=f"Station {station_id} is OFFLINE", reward_delta=-1.0)

            sat_pos = sat.get("orbital_position", 0)
            stn_pos = station.get("position_deg", 0)
            if not has_line_of_sight(sat_pos, stn_pos, fov_deg=station.get("fov_deg", 15)):
                return ActionResult(success=False, message=f"No LoS to {station_id}", reward_delta=-2.0)

            data_gb = sat.get("pending_data_gb", 0.0)
            bw_factor = station.get("bw_factor", 1.0)
            actual_transfer = min(data_gb, station.get("bandwidth_gbps", 1.0) * bw_factor)

            cost = ACTION_BATTERY_COST.get("downlink_data", 5.0)
            if sat.get("battery_level", 0.0) < cost:
                return ActionResult(success=False, message="Insufficient battery for downlink", reward_delta=-1.0)

            sat["battery_level"] = max(0.0, sat["battery_level"] - cost)
            sat["storage_used"] = max(0.0, sat["storage_used"] - (actual_transfer / max(data_gb, 0.01)) * sat["storage_used"])
            sat["pending_data_gb"] = max(0.0, data_gb - actual_transfer)
            station["total_received_gb"] = station.get("total_received_gb", 0.0) + actual_transfer
            self._data_downlinked_gb += actual_transfer
            reward = actual_transfer * ALPHA
            msg = f"{sat_id} downlinked {actual_transfer:.2f} GB to {station_id} (+{reward:.1f} pts)"

        # --- INTER-SATELLITE LINK ---
        elif at == ActionType.INTER_SATELLITE_LINK:
            chain = action.relay_chain or []
            if len(chain) < 2:
                return ActionResult(success=False, message="relay_chain must have at least 2 entries", reward_delta=-5.0)

            # Validate chain feasibility
            sat_positions = {sid: s.get("orbital_position", 0) for sid, s in self._satellites.items()}
            isl_graph = sat_positions_to_isl_graph(sat_positions, ISL_RANGE_DEG)

            valid = True
            for i in range(len(chain) - 1):
                if chain[i + 1] not in isl_graph.get(chain[i], []):
                    valid = False
                    break

            if not valid:
                return ActionResult(success=False, message="ISL chain not feasible (satellites out of range)", reward_delta=-5.0)

            # Transfer data along chain
            src_id = chain[0]
            dest_id = chain[-1]
            src_sat = self._satellites.get(src_id, {})
            dest_sat = self._satellites.get(dest_id, {})

            if not src_sat or not dest_sat:
                return ActionResult(success=False, message="Invalid chain endpoints", reward_delta=-5.0)

            data_to_transfer = src_sat.get("pending_data_gb", 0.0)
            if data_to_transfer <= 0:
                return ActionResult(success=False, message="No data to transfer", reward_delta=-1.0)

            # Drain battery for sender + each relay
            for hop_idx, hop_id in enumerate(chain):
                hop_sat = self._satellites.get(hop_id, {})
                if hop_sat.get("mode") == "dead":
                    return ActionResult(success=False, message=f"Relay {hop_id} is DEAD", reward_delta=-10.0)
                cost = ACTION_BATTERY_COST.get("inter_satellite_link", 10.0) if hop_idx == 0 else ACTION_BATTERY_COST.get("isl_relay_pass", 6.0)
                hop_sat["battery_level"] = max(0.0, hop_sat.get("battery_level", 0.0) - cost)
                hop_sat["mode"] = "isl_relay"

            # Move data
            src_sat["pending_data_gb"] = 0.0
            src_sat["storage_used"] = max(0.0, src_sat["storage_used"] - data_to_transfer * 20)  # approx %
            dest_sat["pending_data_gb"] = dest_sat.get("pending_data_gb", 0.0) + data_to_transfer
            dest_sat["storage_used"] = min(100.0, dest_sat.get("storage_used", 0.0) + data_to_transfer * 20)

            reward = 5.0 * data_to_transfer  # Routing bonus
            msg = f"ISL relay: {' → '.join(chain)} ({data_to_transfer:.2f} GB)"

        # --- STATION KEEPING ---
        elif at == ActionType.STATION_KEEPING:
            fuel_cost = ACTION_FUEL_COST.get("station_keeping", 8.0)
            bat_cost = ACTION_BATTERY_COST.get("station_keeping", 12.0)
            if sat.get("fuel_remaining", 0.0) < fuel_cost:
                return ActionResult(success=False, message="Insufficient fuel", reward_delta=-2.0)
            sat["fuel_remaining"] = max(0.0, sat["fuel_remaining"] - fuel_cost)
            sat["battery_level"] = max(0.0, sat["battery_level"] - bat_cost)
            sat["orbital_drift_deg"] = 0.0  # Correct drift
            reward = 2.0  # Small bonus for maintenance
            msg = f"{sat_id} performed station-keeping burn"

            # Check for debris evasion
            if self._event_engine:
                for ev in self._event_engine.active_events:
                    if ev.event_type == "space_debris" and ev.affected_target == sat_id:
                        self._event_engine.clear_event(ev.event_id)
                        reward += 50.0  # Evasion bonus
                        msg += f" (EVADED SPACE DEBRIS!)"

        # --- EMERGENCY TRANSMIT ---
        elif at == ActionType.EMERGENCY_TRANSMIT:
            station_id = action.target_station
            station = self._stations.get(station_id, {})
            if not station:
                return ActionResult(success=False, message=f"Station {station_id} not found", reward_delta=-5.0)

            sat_pos = sat.get("orbital_position", 0)
            stn_pos = station.get("position_deg", 0)
            # Emergency transmit works even outside normal FoV (wider beam)
            if not has_line_of_sight(sat_pos, stn_pos, fov_deg=25):
                return ActionResult(success=False, message="Too far even for emergency TX", reward_delta=-5.0)

            cost = ACTION_BATTERY_COST.get("emergency_transmit", 15.0)
            if sat.get("battery_level", 0.0) < cost:
                return ActionResult(success=False, message="Insufficient battery for emergency TX", reward_delta=-5.0)

            sat["battery_level"] = max(0.0, sat["battery_level"] - cost)
            data_gb = sat.get("pending_data_gb", 0.0)
            actual = min(data_gb, 0.5)  # Emergency burst is limited
            sat["pending_data_gb"] = max(0.0, data_gb - actual)
            self._data_downlinked_gb += actual
            station["total_received_gb"] = station.get("total_received_gb", 0.0) + actual
            reward = actual * ALPHA * 1.5  # Emergency uplink bonus
            msg = f"{sat_id} emergency TX to {station_id}: {actual:.2f} GB"

        # --- THERMAL VENT ---
        elif at == ActionType.THERMAL_VENT:
            old_thermal = sat.get("thermal_level", 50.0)
            sat["thermal_level"] = max(0.0, old_thermal + THERMAL_VENT_DELTA)
            reward = 1.0  # Small housekeeping reward
            msg = f"{sat_id} thermal vent: {old_thermal:.1f}°→{sat['thermal_level']:.1f}°"

        # --- SEND MESSAGE ---
        elif at == ActionType.SEND_MESSAGE:
            recipient_id = action.recipient_sat_id
            if not recipient_id:
                return ActionResult(success=False, message="SEND_MESSAGE requires recipient_sat_id", reward_delta=-1.0)
            
            sat_positions = {s_id: s.get("orbital_position", 0) for s_id, s in self._satellites.items()}
            isl_graph = sat_positions_to_isl_graph(sat_positions, ISL_RANGE_DEG)

            if recipient_id not in isl_graph.get(sat_id, []):
                return ActionResult(success=False, message=f"{recipient_id} is out of ISL range", reward_delta=-1.0)
            
            recipient = self._satellites.get(recipient_id)
            if not recipient or recipient.get("mode") == "dead":
                return ActionResult(success=False, message=f"{recipient_id} is unavailable", reward_delta=-1.0)

            # Insert message
            payload = action.message_payload or ""
            parsed_msg = f"From {sat_id}: {payload}"
            recipient.setdefault("inbox", []).append(parsed_msg)
            msg = f"{sat_id} sent message to {recipient_id}"
            reward = 0.5  # minor reward for communication

        # --- MAINTENANCE CYCLE ---
        elif at == ActionType.MAINTENANCE_CYCLE:
            if not sat.get("in_sunlight", True):
                return ActionResult(success=False, message="Maintenance requires solar power (in sunlight)", reward_delta=-5.0)
            cost = 15.0
            if sat.get("battery_level", 0.0) < cost:
                return ActionResult(success=False, message="Insufficient battery for maintenance", reward_delta=-2.0)
            
            sat["battery_level"] -= cost
            old_health = sat.get("health_index", 100.0)
            sat["health_index"] = min(100.0, old_health + 15.0)
            reward = 10.0
            msg = f"{sat_id} performed maintenance cycle: {old_health:.0f}% -> {sat['health_index']:.0f}%"

        sat["last_action"] = at.value
        return ActionResult(
            success=True,
            message=msg,
            reward_delta=reward,
            new_events_triggered=new_events,
        )

    # ------------------------------------------------------------------
    # Physics tick
    # ------------------------------------------------------------------

    def _tick_physics(self):
        for sat_id, sat in self._satellites.items():
            if sat.get("mode") == "dead":
                continue

            # Position advance (+ drift correction)
            drag = sat.get("drag_deg", 0.0)
            drift = sat.get("orbital_drift_deg", 0.0)
            sat["orbital_position"] = normalize(
                sat["orbital_position"] + DEGREES_PER_STEP - drag
            )
            sat["orbital_drift_deg"] = drift + drag * 0.1  # accumulate drift
            sat["drag_deg"] = 0.0  # Reset per-step drag

            # Sunlight check
            sunlit = is_in_sunlight(sat["orbital_position"])
            sat["in_sunlight"] = sunlit

            # Battery dynamics
            solar_mult = sat.pop("solar_multiplier", 1.0)  # event modifier
            charge = SOLAR_CHARGE_RATE * solar_mult if sunlit else 0.0
            mode = sat.get("mode", "active")
            extra = 0.0 if mode == "sleep" else ACTIVE_EXTRA_DRAIN
            bat = sat.get("battery_level", 0.0) + charge - PASSIVE_DRAIN_RATE - extra
            sat["battery_level"] = float(np.clip(bat, 0.0, 100.0))

            if sunlit:
                sat["steps_in_eclipse"] = 0
            else:
                sat["steps_in_eclipse"] = sat.get("steps_in_eclipse", 0) + 1

            # Thermal dynamics
            if mode == "sleep":
                heat = SLEEP_HEATING if sunlit else ECLIPSE_HEATING
            else:
                heat = SUNLIGHT_HEATING if sunlit else ECLIPSE_HEATING
            thermal = sat.get("thermal_level", 50.0) + heat
            sat["thermal_level"] = float(np.clip(thermal, 0.0, 100.0))

            # Health Degradation & Battery Cap
            health = sat.get("health_index", 100.0)
            if sat["battery_level"] < 10.0:
                health -= BATTERY_DEGRADATION_RATE
            if sat["thermal_level"] > 80.0:
                health -= THERMAL_DEGRADATION_RATE
            
            sat["health_index"] = float(np.clip(health, 0.0, 100.0))
            max_capacity = 100.0 * (sat["health_index"] / 100.0)
            # Clip battery dynamically according to health percentage cap
            if sat["battery_level"] > max_capacity:
                sat["battery_level"] = max_capacity

            # Update LoS
            positions = {sid: s.get("orbital_position", 0) for sid, s in self._stations.items()}
            sat["line_of_sight_to_ground"] = None
            for stn_id, stn_pos in positions.items():
                if (self._stations[stn_id].get("status") != "offline"
                        and has_line_of_sight(sat["orbital_position"], stn_pos,
                                              fov_deg=self._stations[stn_id].get("fov_deg", 15))):
                    sat["line_of_sight_to_ground"] = stn_id
                    break

            # Station BW factor reset (set fresh by events each step)
            for stn in self._stations.values():
                stn.pop("bw_factor", None)
                if stn.get("offline_countdown", 0) > 0:
                    stn["offline_countdown"] -= 1
                    if stn["offline_countdown"] <= 0:
                        stn["status"] = "online"

    # ------------------------------------------------------------------
    # Step reward
    # ------------------------------------------------------------------

    def _compute_step_reward(self) -> float:
        r = 0.0
        for sat in self._satellites.values():
            if sat.get("mode") == "dead":
                continue
            bat = sat.get("battery_level", 100.0)
            thermal = sat.get("thermal_level", 50.0)
            r -= battery_penalty(bat)
            r -= thermal_penalty(thermal)
        return r

    # ------------------------------------------------------------------
    # Task completion check
    # ------------------------------------------------------------------

    def _check_task_complete(self) -> bool:
        check_fn = self._task_cfg.get("completion_check_fn")
        if check_fn:
            return check_fn(self)
        return False

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, active_events: List[StochasticEvent]) -> Observation:
        sats = []
        sat_positions = {sid: s.get("orbital_position", 0) for sid, s in self._satellites.items()}
        isl_graph = sat_positions_to_isl_graph(sat_positions, ISL_RANGE_DEG)

        for sat_id, s in self._satellites.items():
            # Determine LoS
            los = s.get("line_of_sight_to_ground")
            sats.append(SatelliteTelemetry(
                sat_id=sat_id,
                orbital_position=normalize(s.get("orbital_position", 0)),
                battery_level=round(s.get("battery_level", 100.0), 1),
                storage_used=round(s.get("storage_used", 0.0), 1),
                fuel_remaining=round(s.get("fuel_remaining", 100.0), 1),
                thermal_level=round(s.get("thermal_level", 50.0), 1),
                health_index=round(s.get("health_index", 100.0), 1),
                in_sunlight=bool(s.get("in_sunlight", True)),
                mode=SatelliteMode(s.get("mode", "active")),
                line_of_sight_to_ground=los,
                pending_data_gb=round(s.get("pending_data_gb", 0.0), 2),
                steps_in_eclipse=s.get("steps_in_eclipse", 0),
                orbital_drift_deg=round(s.get("orbital_drift_deg", 0.0), 2),
                last_action=s.get("last_action"),
            ))

        stations = []
        for stn_id, s in self._stations.items():
            stations.append(GroundStation(
                station_id=stn_id,
                position_deg=s.get("position_deg", 0),
                fov_deg=s.get("fov_deg", 15),
                bandwidth_gbps=s.get("bandwidth_gbps", 1.0),
                status=s.get("status", "online"),
                queue_depth_gb=s.get("queue_depth_gb", 0.0),
                total_received_gb=round(s.get("total_received_gb", 0.0), 2),
                offline_countdown=s.get("offline_countdown", 0),
            ))

        requests = []
        for req_id, r in self._requests.items():
            if not r.get("done") and req_id not in self._failed:
                requests.append(ImagingRequest(
                    id=req_id,
                    target_deg=r.get("target_deg", 0),
                    reward=r.get("reward", 100.0),
                    priority=RequestPriority(r.get("priority", "ROUTINE")),
                    deadline_minute=r.get("deadline_minute"),
                    data_size_gb=r.get("data_size_gb", 1.0),
                    target_description=r.get("target_description", ""),
                    assigned_to=r.get("assigned_to"),
                    created_at_minute=r.get("created_at_minute", 0),
                ))

        last_reward = self._reward_history[-1] if self._reward_history else 0.0
        breakdown = {
            "data_revenue": max(0.0, last_reward),
            "battery_penalty": sum(battery_penalty(s.get("battery_level", 100.0))
                                   for s in self._satellites.values()),
            "thermal_penalty": sum(thermal_penalty(s.get("thermal_level", 50.0))
                                   for s in self._satellites.values()),
        }

        return Observation(
            current_orbit_minute=self._minute,
            step_number=self._step,
            task_id=self.task_id,
            task_name=self._task_cfg.get("name", f"Task {self.task_id}"),
            task_description=self._task_cfg.get("description", ""),
            satellites=sats,
            imaging_requests=requests,
            completed_requests=list(self._completed),
            failed_requests=list(self._failed),
            ground_stations=stations,
            active_events=active_events,
            episode_score=round(self._episode_score, 2),
            max_possible_score=self._task_cfg.get("max_score", 1000.0),
            reward_breakdown=breakdown,
            isl_topology=isl_graph,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_grade(norm_score: float) -> str:
        if norm_score >= 0.95: return "S"
        if norm_score >= 0.80: return "A"
        if norm_score >= 0.60: return "B"
        if norm_score >= 0.40: return "C"
        return "F"
