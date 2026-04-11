"""
Task 5 — Emergency Response Protocol (Expert)
═══════════════════════════════════════════════════════════════════════
Scenario:
  A major earthquake has struck a region at 155°. Three Emergency
  imaging requests from different relief agencies arrive simultaneously.
  Each has a strict 60-minute deadline and a 5x reward multiplier.

  The satellite constellation (5 sats) is in various states of readiness:
  - One satellite is in thermal-safe mode (overheating)
  - Another is doing routine maintenance (station-keeping burn)
  - A third has 15% battery — dangerously low

  Additionally, a second disaster imaging cluster at 310° (flooding) is
  dispatched 30 minutes in — the agent must re-prioritise mid-episode.

  The agent acts as FEMA's orbital coordinator. Speed saves lives.
  Every step of latency costs the mission score.

Agent must:
  1. Triage satellites: prioritise those with sufficient battery AND LoS.
  2. Capture all emergency images before deadlines.
  3. Downlink to any available station (3 options with varying bandwidth).
  4. Reallocate mid-episode to handle the second disaster cluster.
  5. Keep satellites alive — a dead satellite during emergency = mission failure.

Grader:
  - Time-to-first-image: how quickly the first emergency image was captured/linked.
  - Deadline success rate: % of emergency requests fulfilled before deadline.
  - Survival rate: all 5 satellites must be alive at the end.
  - Re-prioritisation score: did the agent successfully pivot to the second cluster?
"""

def build_task5() -> dict:
    _second_cluster_added = {"done": False}

    def completion_check(env):
        # All emergency requests fulfilled and all sats alive
        emergency_ids = [r_id for r_id, r in env._requests.items()
                         if r.get("priority") == "EMERGENCY"]
        all_emergency_done = all(r_id in env._completed for r_id in emergency_ids)
        all_alive = all(s.get("mode") != "dead" for s in env._satellites.values())
        return all_emergency_done and all_alive

    def grader_breakdown(env) -> dict:
        em_req = [r for r in env._requests.values() if r.get("priority") == "EMERGENCY"]
        em_done = sum(1 for r in em_req if r.get("done") and not r.get("failed"))
        em_rate = em_done / max(1, len(em_req))
        alive = sum(1 for s in env._satellites.values() if s.get("mode") != "dead")
        survival = alive / max(1, len(env._satellites))
        latency_score = max(0.0, 1.0 - env._step / 80.0) if em_done > 0 else 0.0
        return {
            "emergency_fulfillment_rate": round(em_rate, 4),
            "satellite_survival_rate":    round(survival, 4),
            "latency_score":              round(latency_score, 4),
            "data_downlinked_gb":         round(env._data_downlinked_gb, 2),
            "composite_grade":            round((em_rate + survival + latency_score) / 3.0, 4),
        }

    return {
        "name":        "Emergency Response Protocol",
        "difficulty":  3,
        "description": (
            "Earthquake at 155°! Coordinate 5 satellites to capture and downlink "
            "all emergency imagery before the 60-minute deadline. A second disaster "
            "cluster at 310° spawns at minute 30. Re-prioritise or lose points."
        ),
        "max_steps":   180,
        "max_score":   5000.0,
        "satellites": [
            {
                "sat_id":           "Response-1",
                "orbital_position": 100,
                "battery_level":    88.0,
                "storage_used":     8.0,
                "fuel_remaining":   85.0,
                "thermal_level":    38.0,
                "health_index":     100.0,
                "in_sunlight":      True,
                "mode":             "active",
                "pending_data_gb":  0.0,
                "steps_in_eclipse": 0,
                "orbital_drift_deg": 0.0,
            },
            {
                "sat_id":           "Response-2",
                "orbital_position": 148,
                "battery_level":    75.0,
                "storage_used":     22.0,
                "fuel_remaining":   70.0,
                "thermal_level":    40.0,
                "health_index":     100.0,
                "in_sunlight":      True,
                "mode":             "active",
                "pending_data_gb":  0.5,
                "steps_in_eclipse": 0,
                "orbital_drift_deg": 0.0,
            },
            {
                "sat_id":           "Response-3",
                "orbital_position": 200,
                "battery_level":    15.0,    # CRITICAL
                "storage_used":     5.0,
                "fuel_remaining":   60.0,
                "thermal_level":    35.0,
                "health_index":     92.0,
                "in_sunlight":      False,
                "mode":             "sleep",  # Trying to survive eclipse
                "pending_data_gb":  0.0,
                "steps_in_eclipse": 8,
                "orbital_drift_deg": 0.0,
            },
            {
                "sat_id":           "Response-4",
                "orbital_position": 255,
                "battery_level":    60.0,
                "storage_used":     50.0,
                "fuel_remaining":   50.0,
                "thermal_level":    78.0,    # OVERHEATING — must vent first
                "health_index":     88.0,
                "in_sunlight":      False,
                "mode":             "thermal_safe",
                "pending_data_gb":  1.5,
                "steps_in_eclipse": 5,
                "orbital_drift_deg": 0.8,
            },
            {
                "sat_id":           "Response-5",
                "orbital_position": 320,
                "battery_level":    95.0,
                "storage_used":     3.0,
                "fuel_remaining":   90.0,
                "thermal_level":    32.0,
                "health_index":     100.0,
                "in_sunlight":      True,
                "mode":             "active",
                "pending_data_gb":  0.0,
                "steps_in_eclipse": 0,
                "orbital_drift_deg": 0.0,
            },
        ],
        "stations": [
            {
                "station_id":      "Station_Tokyo",
                "position_deg":    140,
                "fov_deg":         15,
                "bandwidth_gbps":  4.0,
                "status":          "online",
                "queue_depth_gb":  0.0,
                "total_received_gb": 0.0,
                "offline_countdown": 0,
            },
            {
                "station_id":      "Station_Norway",
                "position_deg":    50,
                "fov_deg":         15,
                "bandwidth_gbps":  3.0,
                "status":          "online",
                "queue_depth_gb":  0.0,
                "total_received_gb": 0.0,
                "offline_countdown": 0,
            },
            {
                "station_id":      "Station_Chile",
                "position_deg":    245,
                "fov_deg":         12,
                "bandwidth_gbps":  2.0,
                "status":          "online",
                "queue_depth_gb":  0.0,
                "total_received_gb": 0.0,
                "offline_countdown": 0,
            },
        ],
        "requests": [
            {
                "id":                 "DISASTER-1",
                "target_deg":         152,
                "reward":             500.0,
                "priority":           "EMERGENCY",
                "deadline_minute":    60,
                "data_size_gb":       1.0,
                "target_description": "🌍 Earthquake epicentre structural damage overview",
                "done":               False,
                "created_at_minute":  0,
            },
            {
                "id":                 "DISASTER-2",
                "target_deg":         158,
                "reward":             500.0,
                "priority":           "EMERGENCY",
                "deadline_minute":    60,
                "data_size_gb":       1.0,
                "target_description": "🌍 Earthquake zone refugee camp survey",
                "done":               False,
                "created_at_minute":  0,
            },
            {
                "id":                 "DISASTER-3",
                "target_deg":         162,
                "reward":             500.0,
                "priority":           "EMERGENCY",
                "deadline_minute":    60,
                "data_size_gb":       1.5,
                "target_description": "🌍 Earthquake coastal tsunami risk zone",
                "done":               False,
                "created_at_minute":  0,
            },
            # Second cluster — injected at minute 30 (step 15) by env tick
            {
                "id":                 "FLOOD-1",
                "target_deg":         308,
                "reward":             400.0,
                "priority":           "EMERGENCY",
                "deadline_minute":    90,
                "data_size_gb":       1.2,
                "target_description": "🌊 Flash flood damage assessment",
                "done":               False,
                "created_at_minute":  30,  # Injected mid-episode
            },
            {
                "id":                 "FLOOD-2",
                "target_deg":         315,
                "reward":             400.0,
                "priority":           "EMERGENCY",
                "deadline_minute":    90,
                "data_size_gb":       0.8,
                "target_description": "🌊 Flood displacement mapping",
                "done":               False,
                "created_at_minute":  30,
            },
        ],
        "completion_check_fn":  completion_check,
        "grader_breakdown_fn":  grader_breakdown,
    }
