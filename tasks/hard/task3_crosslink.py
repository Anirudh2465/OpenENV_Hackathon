"""
Task 3 — Laser Cross-Linking (Hard)
═══════════════════════════════════════════════════════════════════════
Scenario:
  A critical geopolitical imagery request has landing at the Pentagon
  (Washington DC ground station, position 45°). Sat-1 just captured the
  image at position 215°—opposite side of Earth. Direct LoS to DC won't
  occur for ~45 sim-minutes.

  Instead, the agent must daisy-chain an ISL relay: Sat-1 → Sat-2 →
  Sat-3 → Sat-4, where Sat-4 is approaching DC line-of-sight.

  Complication: Each relay hop costs significant battery. Sat-3 is
  already running at 35% battery and risks death if used carelessly.

Agent must:
  1. Assess ISL topology and find the minimum-hop path.
  2. Ensure each relay satellite has enough battery for the hop.
  3. Maybe sleep Sat-3 briefly before the relay to recover charge.
  4. Execute the full ISL chain to deliver data to DC.

Grader:
  - Speed score: inverse of steps taken to deliver (min-steps = perfect).
  - Routing score: 1.0 if optimal path used, partial for suboptimal.
  - Penalty: any satellite battery death during relay = -0.5 on grade.
"""

def build_task3() -> dict:
    _relay_delivered = {"done": False}

    def completion_check(env):
        # Done when DC station has received data
        dc = env._stations.get("Station_DC", {})
        return dc.get("total_received_gb", 0.0) > 0.5

    def grader_breakdown(env) -> dict:
        dc = env._stations.get("Station_DC", {})
        received = dc.get("total_received_gb", 0.0)
        steps = env._step
        # Optimal is ~12 steps (6 for relay + 6 to reach DC LoS)
        speed_score = max(0.0, 1.0 - max(0.0, steps - 12) / 50.0)
        routing_score = 1.0 if received >= 2.0 else received / 2.0
        deaths = sum(1 for s in env._satellites.values() if s.get("mode") == "dead")
        death_penalty = deaths * 0.5
        return {
            "data_received_dc_gb": round(received, 2),
            "speed_score":         round(speed_score, 4),
            "routing_score":       round(routing_score, 4),
            "death_penalty":       round(death_penalty, 2),
            "composite":           round(max(0.0, (speed_score + routing_score) / 2.0 - death_penalty), 4),
        }

    return {
        "name":        "Laser Cross-Link Relay",
        "difficulty":  3,
        "description": (
            "Route critical imagery from Sat-1 (215°) to DC station (45°) "
            "via ISL relay chain. Every step counts. Sat-3 is battery-critical."
        ),
        "max_steps":   120,
        "max_score":   1200.0,
        "satellites": [
            {
                "sat_id":           "Sat-1",
                "orbital_position": 215,
                "battery_level":    70.0,
                "storage_used":     40.0,
                "fuel_remaining":   85.0,
                "thermal_level":    38.0,
                "health_index":     100.0,
                "in_sunlight":      False,  # Eclipse — can't charge yet
                "mode":             "active",
                "pending_data_gb":  2.0,    # The critical image
                "steps_in_eclipse": 3,
                "orbital_drift_deg": 0.0,
            },
            {
                "sat_id":           "Sat-2",
                "orbital_position": 258,
                "battery_level":    80.0,
                "storage_used":     12.0,
                "fuel_remaining":   90.0,
                "thermal_level":    35.0,
                "health_index":     100.0,
                "in_sunlight":      False,
                "mode":             "active",
                "pending_data_gb":  0.0,
                "steps_in_eclipse": 1,
                "orbital_drift_deg": 0.0,
            },
            {
                "sat_id":           "Sat-3",
                "orbital_position": 310,
                "battery_level":    35.0,   # CRITICAL — Must manage carefully
                "storage_used":     5.0,
                "fuel_remaining":   70.0,
                "thermal_level":    32.0,
                "health_index":     95.0,
                "in_sunlight":      True,   # Charging now
                "mode":             "sleep",  # Agent must wake it at right moment
                "pending_data_gb":  0.0,
                "steps_in_eclipse": 0,
                "orbital_drift_deg": 0.5,
            },
            {
                "sat_id":           "Sat-4",
                "orbital_position": 25,
                "battery_level":    90.0,
                "storage_used":     8.0,
                "fuel_remaining":   95.0,
                "thermal_level":    30.0,
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
                "station_id":      "Station_DC",
                "position_deg":    45,
                "fov_deg":         15,
                "bandwidth_gbps":  5.0,    # High priority station
                "status":          "online",
                "queue_depth_gb":  0.0,
                "total_received_gb": 0.0,
                "offline_countdown": 0,
            },
            {
                "station_id":      "Station_Hawaii",
                "position_deg":    200,
                "fov_deg":         12,
                "bandwidth_gbps":  1.0,
                "status":          "online",
                "queue_depth_gb":  0.0,
                "total_received_gb": 0.0,
                "offline_countdown": 0,
            },
        ],
        "requests": [
            {
                "id":                 "REQ-CRITICAL-1",
                "target_deg":         215,
                "reward":             1000.0,
                "priority":           "EMERGENCY",
                "deadline_minute":    90,             # ~45 sim minutes to deliver
                "data_size_gb":       2.0,
                "target_description": "CLASSIFIED: Geopolitical imagery for Pentagon",
                "done":               True,           # Already captured — must relay
                "captured_by":        "Sat-1",
                "created_at_minute":  0,
            }
        ],
        "completion_check_fn":  completion_check,
        "grader_breakdown_fn":  grader_breakdown,
    }
