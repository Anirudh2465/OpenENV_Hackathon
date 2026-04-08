"""
OpenEnv-Orbital-Command | models.py
Pydantic v2 data models for the full simulation interface.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import time


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    CAPTURE_IMAGE        = "capture_image"
    DOWNLINK_DATA        = "downlink_data"
    SLEEP_MODE           = "sleep_mode"
    INTER_SATELLITE_LINK = "inter_satellite_link"
    STATION_KEEPING      = "station_keeping"   # Fuel burn to correct orbital drift
    EMERGENCY_TRANSMIT   = "emergency_transmit" # High-power burst, costs 15% battery
    THERMAL_VENT         = "thermal_vent"       # Dump heat, costs 1 step


class SatelliteMode(str, Enum):
    ACTIVE      = "active"
    SLEEP       = "sleep"
    TRANSMITTING= "transmitting"
    ISL_RELAY   = "isl_relay"
    THERMAL_SAFE= "thermal_safe"
    DEAD        = "dead"


class RequestPriority(str, Enum):
    ROUTINE   = "ROUTINE"
    URGENT    = "URGENT"
    EMERGENCY = "EMERGENCY"


class EventType(str, Enum):
    SOLAR_FLARE         = "solar_flare"       # Charge +150%, random component damage
    GROUND_OUTAGE       = "ground_outage"     # Station offline for N steps
    PRIORITY_ESCALATION = "priority_escalation" # Request priority bumps up
    ATMOSPHERIC_DRAG    = "atmospheric_drag"  # Satellite slows ~1 deg
    BANDWIDTH_CONGESTION= "bandwidth_congestion" # Station queue fills


# ---------------------------------------------------------------------------
# Core Telemetry & State
# ---------------------------------------------------------------------------

class SatelliteTelemetry(BaseModel):
    sat_id: str
    orbital_position: int       = Field(..., ge=0, le=359, description="Position in 0-359° arc")
    battery_level: float        = Field(..., ge=0.0, le=100.0, description="State of charge %")
    storage_used: float         = Field(..., ge=0.0, le=100.0, description="Flash storage fill %")
    fuel_remaining: float       = Field(..., ge=0.0, le=100.0, description="Delta-V reserve %")
    thermal_level: float        = Field(..., ge=0.0, le=100.0, description="Component temperature %")
    health_index: float         = Field(100.0, ge=0.0, le=100.0, description="Cumulative hardware health")
    in_sunlight: bool
    mode: SatelliteMode         = SatelliteMode.ACTIVE
    line_of_sight_to_ground: Optional[str] = None
    pending_data_gb: float      = Field(0.0, ge=0.0, description="Data awaiting downlink (GB)")
    steps_in_eclipse: int       = Field(0, ge=0, description="Consecutive steps in eclipse")
    orbital_drift_deg: float    = Field(0.0, description="Accumulated drift from nominal orbit")
    last_action: Optional[str]  = None

    # Derived helper (not stored)
    @property
    def is_critically_low_battery(self) -> bool:
        return self.battery_level < 20.0

    @property
    def is_storage_critical(self) -> bool:
        return self.storage_used > 90.0


class ImagingRequest(BaseModel):
    id: str
    target_deg: int             = Field(..., ge=0, le=359)
    reward: float               = Field(..., gt=0.0)
    priority: RequestPriority   = RequestPriority.ROUTINE
    deadline_minute: Optional[int] = None  # None = no hard deadline
    data_size_gb: float         = Field(1.0, gt=0.0)
    target_description: str     = ""
    assigned_to: Optional[str]  = None    # sat_id if claimed
    created_at_minute: int      = 0

    @property
    def effective_reward(self) -> float:
        """Priority multiplier on base reward."""
        mult = {"ROUTINE": 1.0, "URGENT": 2.0, "EMERGENCY": 5.0}
        return self.reward * mult.get(self.priority, 1.0)


class GroundStation(BaseModel):
    station_id: str
    position_deg: int           = Field(..., ge=0, le=359)
    fov_deg: int                = Field(15, ge=5, le=45)
    bandwidth_gbps: float       = Field(1.0, gt=0.0)
    status: str                 = "online"   # "online" | "offline" | "congested"
    queue_depth_gb: float       = Field(0.0, ge=0.0)
    total_received_gb: float    = Field(0.0, ge=0.0)
    offline_countdown: int      = 0           # Steps remaining in offline state


class StochasticEvent(BaseModel):
    event_id: str
    event_type: EventType
    affected_target: str         # sat_id or station_id
    duration_steps: int
    steps_remaining: int
    magnitude: float             = Field(1.0, description="Scaling factor for event impact")
    description: str             = ""


# ---------------------------------------------------------------------------
# Observation / Action interface
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    current_orbit_minute: int
    step_number: int
    task_id: int
    task_name: str
    task_description: str
    satellites: List[SatelliteTelemetry]
    imaging_requests: List[ImagingRequest]       # Pending requests
    completed_requests: List[str]                # IDs fulfilled this episode
    failed_requests: List[str]                   # IDs missed (deadline passed)
    ground_stations: List[GroundStation]
    active_events: List[StochasticEvent]
    episode_score: float
    max_possible_score: float
    reward_breakdown: Dict[str, float]           # Detailed last-step reward components
    isl_topology: Dict[str, List[str]]           # sat_id -> [reachable sat_ids via ISL]

    # Convenience helpers that an agent can query
    def get_satellite(self, sat_id: str) -> Optional[SatelliteTelemetry]:
        return next((s for s in self.satellites if s.sat_id == sat_id), None)

    def get_station(self, station_id: str) -> Optional[GroundStation]:
        return next((s for s in self.ground_stations if s.station_id == station_id), None)


class Action(BaseModel):
    action_type: ActionType
    target_sat_id: str
    request_id: Optional[str]      = None
    target_station: Optional[str]  = None
    relay_chain: Optional[List[str]] = None   # Ordered relay sat IDs for ISL routing
    reasoning: Optional[str]       = None     # LLM chain-of-thought (logged, not executed)

    @model_validator(mode="after")
    def validate_action_fields(self) -> "Action":
        if self.action_type == ActionType.CAPTURE_IMAGE and not self.request_id:
            raise ValueError("capture_image requires a request_id")
        if self.action_type == ActionType.DOWNLINK_DATA and not self.target_station:
            raise ValueError("downlink_data requires a target_station")
        if self.action_type == ActionType.INTER_SATELLITE_LINK and not self.relay_chain:
            raise ValueError("inter_satellite_link requires a relay_chain")
        return self


class ActionResult(BaseModel):
    success: bool
    message: str
    reward_delta: float
    satellite_after: Optional[SatelliteTelemetry] = None
    new_events_triggered: List[str] = []


class EpisodeResult(BaseModel):
    task_id: int
    task_name: str
    total_steps: int
    final_score: float
    normalized_score: float          # 0.0 – 1.0
    grade: str                       # "S" / "A" / "B" / "C" / "F"
    data_downlinked_gb: float
    data_overwritten_gb: float
    satellites_survived: int
    total_satellites: int
    requests_fulfilled: int
    requests_missed: int
    emergency_requests_handled: int
    action_history: List[Dict[str, Any]]
    reward_history: List[float]
    grader_breakdown: Dict[str, float]
    duration_seconds: float
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @property
    def survival_rate(self) -> float:
        if self.total_satellites == 0:
            return 0.0
        return self.satellites_survived / self.total_satellites

    @property
    def fulfillment_rate(self) -> float:
        total = self.requests_fulfilled + self.requests_missed
        if total == 0:
            return 1.0
        return self.requests_fulfilled / total


# ---------------------------------------------------------------------------
# Leaderboard entry
# ---------------------------------------------------------------------------

class LeaderboardEntry(BaseModel):
    rank: int
    agent_name: str
    model_name: str
    task_id: int
    normalized_score: float
    grade: str
    timestamp: str
    episode_id: str
