"""
OpenEnv-Orbital-Command | events.py
Stochastic event engine — injects dynamic disruptions into the simulation.

Events add realism and test the agent's adaptability beyond pattern-memorisation.
"""
from __future__ import annotations
import random
import uuid
from typing import Dict, List, Optional, Any
from .models import StochasticEvent, EventType


# ---------------------------------------------------------------------------
# Event probability tables  (per-step probability of each event spawning)
# ---------------------------------------------------------------------------

EVENT_PROBABILITIES: Dict[str, float] = {
    EventType.SOLAR_FLARE:          0.03,   # 3% per step
    EventType.GROUND_OUTAGE:        0.04,
    EventType.PRIORITY_ESCALATION:  0.05,
    EventType.ATMOSPHERIC_DRAG:     0.02,
    EventType.BANDWIDTH_CONGESTION: 0.04,
}

EVENT_DURATIONS: Dict[str, tuple] = {
    # (min_steps, max_steps)
    EventType.SOLAR_FLARE:          (2, 6),
    EventType.GROUND_OUTAGE:        (5, 15),
    EventType.PRIORITY_ESCALATION:  (1, 1),   # Instant
    EventType.ATMOSPHERIC_DRAG:     (3, 8),
    EventType.BANDWIDTH_CONGESTION: (4, 10),
}

EVENT_MAGNITUDES: Dict[str, tuple] = {
    # (min, max) — semantics are event-specific
    EventType.SOLAR_FLARE:          (1.5, 3.0),  # Charge rate multiplier
    EventType.GROUND_OUTAGE:        (1.0, 1.0),  # Binary
    EventType.PRIORITY_ESCALATION:  (1.0, 1.0),  # Binary
    EventType.ATMOSPHERIC_DRAG:     (0.5, 2.0),  # Degrees of drag
    EventType.BANDWIDTH_CONGESTION: (0.3, 0.8),  # BW reduction factor
}

EVENT_DESCRIPTIONS: Dict[str, str] = {
    EventType.SOLAR_FLARE: (
        "⚡ SOLAR FLARE: Elevated proton flux detected. "
        "Charging boosted but component damage risk increased."
    ),
    EventType.GROUND_OUTAGE: (
        "🌨️ GROUND OUTAGE: Station {target} offline due to weather/maintenance."
    ),
    EventType.PRIORITY_ESCALATION: (
        "🚨 PRIORITY ESCALATION: Request escalated to EMERGENCY by mission control."
    ),
    EventType.ATMOSPHERIC_DRAG: (
        "🌫️ ATMOSPHERIC DRAG: Upper atmosphere density spike slowing Sat {target}."
    ),
    EventType.BANDWIDTH_CONGESTION: (
        "📡 BANDWIDTH CONGESTION: Station {target} queue saturated — reduced throughput."
    ),
}


# ---------------------------------------------------------------------------
# EventEngine
# ---------------------------------------------------------------------------

class EventEngine:
    """
    Generates and manages stochastic events during an episode.

    Usage:
        engine = EventEngine(seed=42, task_difficulty=2)
        new_events = engine.tick(sat_ids, station_ids, pending_requests)
    """

    def __init__(self, seed: int = 0, task_difficulty: int = 1,
                 events_enabled: bool = True):
        self.rng = random.Random(seed)
        self.difficulty = task_difficulty      # 1-3; scales probability
        self.enabled = events_enabled
        self._active: List[StochasticEvent] = []
        self._history: List[StochasticEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(
        self,
        sat_ids: List[str],
        station_ids: List[str],
        request_ids: List[str],
        step: int,
    ) -> List[StochasticEvent]:
        """
        Advance one simulation step:
        1. Tick down durations on active events.
        2. Expire finished events.
        3. Probabilistically spawn new events.
        Returns the updated list of active events.
        """
        if not self.enabled:
            return []

        # Tick & expire
        still_active: List[StochasticEvent] = []
        for ev in self._active:
            ev.steps_remaining -= 1
            if ev.steps_remaining > 0:
                still_active.append(ev)
            else:
                self._history.append(ev)
        self._active = still_active

        # Spawn new events
        scale = 1.0 + 0.3 * (self.difficulty - 1)  # higher difficulty → more events
        for event_type in EventType:
            base_prob = EVENT_PROBABILITIES.get(event_type, 0.0)
            if self.rng.random() < base_prob * scale:
                target = self._select_target(event_type, sat_ids, station_ids, request_ids)
                if target:
                    ev = self._spawn(event_type, target)
                    self._active.append(ev)

        return list(self._active)

    def apply_to_state(
        self,
        sat_states: Dict[str, Dict[str, Any]],
        station_states: Dict[str, Dict[str, Any]],
        request_states: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Apply active events to the raw state dicts. Returns a dict of side-effects
        for the caller to log/react to.
        """
        effects: Dict[str, Any] = {}

        for ev in self._active:
            et = ev.event_type
            tgt = ev.affected_target

            if et == EventType.SOLAR_FLARE:
                if tgt in sat_states:
                    sat_states[tgt]["solar_multiplier"] = float(ev.magnitude)
                    # Small chance of health damage
                    if self.rng.random() < 0.15:
                        sat_states[tgt]["health_index"] = max(
                            0.0, sat_states[tgt].get("health_index", 100.0) - 5.0
                        )
                        effects[f"damage_{tgt}"] = "Solar particle event degraded component"

            elif et == EventType.GROUND_OUTAGE:
                if tgt in station_states:
                    station_states[tgt]["status"] = "offline"

            elif et == EventType.ATMOSPHERIC_DRAG:
                if tgt in sat_states:
                    sat_states[tgt]["drag_deg"] = float(ev.magnitude)

            elif et == EventType.BANDWIDTH_CONGESTION:
                if tgt in station_states:
                    station_states[tgt]["bw_factor"] = float(ev.magnitude)

            elif et == EventType.PRIORITY_ESCALATION:
                if tgt in request_states:
                    p = request_states[tgt].get("priority", "ROUTINE")
                    if p == "ROUTINE":
                        request_states[tgt]["priority"] = "URGENT"
                    elif p == "URGENT":
                        request_states[tgt]["priority"] = "EMERGENCY"

        return effects

    @property
    def active_events(self) -> List[StochasticEvent]:
        return list(self._active)

    @property
    def history(self) -> List[StochasticEvent]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_target(
        self,
        event_type: EventType,
        sat_ids: List[str],
        station_ids: List[str],
        request_ids: List[str],
    ) -> Optional[str]:
        if event_type in (EventType.SOLAR_FLARE, EventType.ATMOSPHERIC_DRAG):
            return self.rng.choice(sat_ids) if sat_ids else None
        elif event_type in (EventType.GROUND_OUTAGE, EventType.BANDWIDTH_CONGESTION):
            return self.rng.choice(station_ids) if station_ids else None
        elif event_type == EventType.PRIORITY_ESCALATION:
            return self.rng.choice(request_ids) if request_ids else None
        return None

    def _spawn(self, event_type: EventType, target: str) -> StochasticEvent:
        dur_min, dur_max = EVENT_DURATIONS[event_type]
        mag_min, mag_max = EVENT_MAGNITUDES[event_type]
        duration = self.rng.randint(dur_min, dur_max)
        magnitude = self.rng.uniform(mag_min, mag_max)
        desc = EVENT_DESCRIPTIONS.get(event_type, "Unknown event").format(target=target)

        return StochasticEvent(
            event_id=str(uuid.uuid4())[:8],
            event_type=event_type,
            affected_target=target,
            duration_steps=duration,
            steps_remaining=duration,
            magnitude=round(magnitude, 2),
            description=desc,
        )
