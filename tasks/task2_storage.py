"""
Task 2 — Storage Bottleneck (Medium)
═══════════════════════════════════════════════════════════════════════
Scenario:
  Three satellites are generating massive imagery over Europe. Onboard
  storage is at 80% and filling fast. The primary downlink station
  (Norway) is OFFLINE due to a blizzard. The secondary station
  (Antarctica) has a narrow 4-step LoS window that the agent must hit.

  Additionally, a stochastic solar flare mid-episode may corrupt
  one satellite if the agent isn't paying attention to health_index.

Agent must:
  1. Hold data (avoid captures that would overflow storage → overwrite penalty).
  2. Calculate the Antarctica downlink window across all three satellites.
  3. Downlink within the window (4-step pass) before storage overflows.
  4. Optionally use ISL to route data from the satellite with most data
     to the one with best Antarctica LoS angle.

Grader:
  - Continuous: (data_downlinked_gb / total_capturable_gb) * 100
  - Heavy penalty if any satellite hits 100% storage (overwrite event).
"""

def build_task2() -> dict:
    def completion_check(env):
        # Episode ends when all storage is cleared or max_steps exceeded
        all_clear = all(s.get("storage_used", 0) < 5.0 for s in env._satellites.values())
        return all_clear

    def grader_breakdown(env) -> dict:
        max_possible = 8.0  # 3 sats × ~2.67 GB each could downlink
        downlinked = env._data_downlinked_gb
        overwritten = env._data_overwritten_gb
        return {
            "data_downlinked_gb":  round(downlinked, 2),
            "data_overwritten_gb": round(overwritten, 2),
            "throughput_score":    round(min(1.0, downlinked / max(0.01, max_possible)), 4),
            "overwrite_penalty_total": round(overwritten * 20.0, 2),
        }

    return {
        "name":        "Storage Bottleneck",
        "difficulty":  2,
        "description": (
            "Norway is offline. Three satellites are filling up fast. "
            "Hit the Antarctic downlink window before data is overwritten."
        ),
        "max_steps":   150,
        "max_score":   800.0,
        "satellites": [
            {
                "sat_id":           "Sat-1",
                "orbital_position": 30,
                "battery_level":    85.0,
                "storage_used":     78.0,
                "fuel_remaining":   75.0,
                "thermal_level":    42.0,
                "health_index":     100.0,
                "in_sunlight":      True,
                "mode":             "active",
                "pending_data_gb":  3.0,
                "steps_in_eclipse": 0,
                "orbital_drift_deg": 0.0,
            },
            {
                "sat_id":           "Sat-2",
                "orbital_position": 80,
                "battery_level":    70.0,
                "storage_used":     82.0,
                "fuel_remaining":   80.0,
                "thermal_level":    45.0,
                "health_index":     100.0,
                "in_sunlight":      True,
                "mode":             "active",
                "pending_data_gb":  2.5,
                "steps_in_eclipse": 0,
                "orbital_drift_deg": 0.0,
            },
            {
                "sat_id":           "Sat-3",
                "orbital_position": 140,
                "battery_level":    60.0,
                "storage_used":     90.0,
                "fuel_remaining":   65.0,
                "thermal_level":    50.0,
                "health_index":     100.0,
                "in_sunlight":      True,
                "mode":             "active",
                "pending_data_gb":  2.0,
                "steps_in_eclipse": 0,
                "orbital_drift_deg": 0.0,
            },
        ],
        "stations": [
            {
                "station_id":      "Station_Norway",
                "position_deg":    50,
                "fov_deg":         15,
                "bandwidth_gbps":  3.0,
                "status":          "offline",        # Primary offline
                "queue_depth_gb":  0.0,
                "total_received_gb": 0.0,
                "offline_countdown": 999,            # Blizzard — stays down
            },
            {
                "station_id":      "Station_Antarctica",
                "position_deg":    280,
                "fov_deg":         12,               # Narrow window
                "bandwidth_gbps":  1.5,
                "status":          "online",
                "queue_depth_gb":  0.0,
                "total_received_gb": 0.0,
                "offline_countdown": 0,
            },
        ],
        "requests": [
            # A few pending requests the agent can choose to capture (extra revenue)
            {
                "id":                 "REQ-BONUS-1",
                "target_deg":         60,
                "reward":             120.0,
                "priority":           "ROUTINE",
                "deadline_minute":    None,
                "data_size_gb":       0.5,
                "target_description": "Alpine glacier survey bonus",
                "done":               False,
                "created_at_minute":  0,
            },
            {
                "id":                 "REQ-BONUS-2",
                "target_deg":         100,
                "reward":             80.0,
                "priority":           "ROUTINE",
                "deadline_minute":    None,
                "data_size_gb":       0.3,
                "target_description": "North Sea shipping lane imagery",
                "done":               False,
                "created_at_minute":  0,
            },
        ],
        "completion_check_fn":  completion_check,
        "grader_breakdown_fn":  grader_breakdown,
    }
