"""
Task 1 — Eclipse Survival (Easy)
═══════════════════════════════════════════════════════════════════════
Scenario:
  A single satellite (Sat-Alpha) approaches a high-value imaging target
  at 200° — located deep inside the eclipse zone (170°–350°). Its battery
  starts at 60%. Without planning the agent will run out of power before
  it can complete the capture and survive the dark zone.

Agent must:
  1. Enter sleep_mode while in sunlight to fully charge the battery.
  2. Wake at the exact position to capture the target.
  3. Return to sleep immediately to survive the rest of the eclipse.

Grader:
  - 1.0 normalised if image captured AND all satellites alive at mission end.
  - 0.0 if battery hits 0% or the imaging window is missed.
"""

def build_task1() -> dict:
    def completion_check(env):
        # Done when the one imaging request is captured and the sat survives
        all_captured = all(r.get("done") for r in env._requests.values())
        all_alive = all(s.get("mode") != "dead" for s in env._satellites.values())
        return all_captured and all_alive

    def grader_breakdown(env) -> dict:
        captured = "REQ-1" in env._completed
        alive = env._satellites.get("Sat-Alpha", {}).get("mode") != "dead"
        return {
            "image_captured":   float(captured),
            "satellite_alive":  float(alive),
            "binary_score":     float(captured and alive),
        }

    return {
        "name":        "Eclipse Survival",
        "difficulty":  1,
        "description": (
            "Sat-Alpha must capture an image at 200° (inside eclipse) "
            "without running out of battery. Budget your sleep cycles wisely."
        ),
        "max_steps":   100,
        "max_score":   500.0,
        "satellites": [
            {
                "sat_id":           "Sat-Alpha",
                "orbital_position": 120,
                "battery_level":    60.0,
                "storage_used":     10.0,
                "fuel_remaining":   90.0,
                "thermal_level":    40.0,
                "health_index":     100.0,
                "in_sunlight":      True,
                "mode":             "active",
                "pending_data_gb":  0.0,
                "steps_in_eclipse": 0,
                "orbital_drift_deg": 0.0,
            }
        ],
        "stations": [
            {
                "station_id":      "Station_Norway",
                "position_deg":    50,
                "fov_deg":         15,
                "bandwidth_gbps":  2.0,
                "status":          "online",
                "queue_depth_gb":  0.0,
                "total_received_gb": 0.0,
                "offline_countdown": 0,
            }
        ],
        "requests": [
            {
                "id":                 "REQ-1",
                "target_deg":         200,
                "reward":             400.0,
                "priority":           "URGENT",
                "deadline_minute":    120,
                "data_size_gb":       2.0,
                "target_description": "Pacific Ocean thermal anomaly survey",
                "done":               False,
                "created_at_minute":  0,
            }
        ],
        "completion_check_fn":  completion_check,
        "grader_breakdown_fn":  grader_breakdown,
    }
