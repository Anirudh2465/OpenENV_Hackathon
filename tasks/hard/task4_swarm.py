"""
Task 4 — Swarm Harvest (Very Hard)
═══════════════════════════════════════════════════════════════════════
Scenario:
  Six satellites must cooperatively image 8 high-value targets spread
  across the orbit within a single pass. Multiple ground stations are
  competing for downlink bandwidth. A solar flare is probabilistically
  scheduled mid-episode.

  The challenge: Satellites must divide imaging assignments optimally
  (no two satellites should image the same target), downlink their
  data within their respective LoS windows, and survive battery death
  from eclipse zones along the way.

Agent must:
  1. Assign each imaging request to an appropriate satellite.
  2. Coordinate downlinks to avoid bandwidth contention.
  3. Use ISL to route data from satellites that missed their window.
  4. Manage 3 resources (battery, storage, fuel) across 6 satellites simultaneously.

Grader:
  - Sum of rewards for all successfully downlinked images.
  - Collaborative bonus: if all 8 targets imaged = +500 pts.
  - Parallel efficiency: measures how many satellites were active vs idle.
"""

def build_task4() -> dict:
    def completion_check(env):
        all_done = all(r.get("done") for r in env._requests.values())
        all_down = env._data_downlinked_gb >= 6.0
        return all_done and all_down

    def grader_breakdown(env) -> dict:
        total_requests = len(env._requests)
        completed = len(env._completed)
        coverage = completed / max(1, total_requests)
        bonus = 500.0 if completed == total_requests else 0.0
        active_steps = sum(
            1 for a in env._action_history
            if a.get("action", {}).get("action_type") != "sleep_mode"
        )
        total_steps = max(1, env._step)
        efficiency = active_steps / (total_steps * len(env._satellites))
        return {
            "coverage_score":   round(coverage, 4),
            "swarm_bonus":      bonus,
            "efficiency_score": round(efficiency, 4),
            "total_downlinked": round(env._data_downlinked_gb, 2),
        }

    sats = []
    positions = [10, 55, 105, 165, 220, 290]
    batteries = [90.0, 80.0, 75.0, 85.0, 70.0, 95.0]
    storages  = [15.0, 20.0, 10.0, 30.0, 18.0, 12.0]

    for i, (pos, bat, stor) in enumerate(zip(positions, batteries, storages)):
        sats.append({
            "sat_id":           f"Swarm-{i+1}",
            "orbital_position": pos,
            "battery_level":    bat,
            "storage_used":     stor,
            "fuel_remaining":   80.0 - i * 5,
            "thermal_level":    40.0 + i * 2,
            "health_index":     100.0,
            "in_sunlight":      True if pos < 170 or pos > 350 else False,
            "mode":             "active",
            "pending_data_gb":  0.0,
            "steps_in_eclipse": 0,
            "orbital_drift_deg": 0.0,
        })

    target_positions = [25, 75, 120, 190, 230, 270, 315, 345]
    target_names = [
        "Amazon deforestation survey",
        "Arctic ice sheet mapping",
        "Mediterranean drought index",
        "South Atlantic volcanic activity",
        "Indian Ocean cyclone track",
        "Pacific fishery survey",
        "Antarctic ozone hole measurement",
        "Himalayas glacier retreat survey",
    ]
    rewards = [150, 200, 180, 300, 250, 120, 280, 190]
    priorities = ["ROUTINE", "URGENT", "ROUTINE", "EMERGENCY",
                  "URGENT", "ROUTINE", "URGENT", "ROUTINE"]

    requests = []
    for i, (tpos, tname, rew, pri) in enumerate(zip(target_positions, target_names, rewards, priorities)):
        requests.append({
            "id":                 f"HARVEST-{i+1}",
            "target_deg":         tpos,
            "reward":             float(rew),
            "priority":           pri,
            "deadline_minute":    200,
            "data_size_gb":       round(0.5 + i * 0.2, 1),
            "target_description": tname,
            "done":               False,
            "created_at_minute":  0,
        })

    return {
        "name":        "Global Swarm Harvest",
        "difficulty":  3,
        "description": (
            "6 satellites, 8 imaging targets, 3 ground stations. "
            "Coordinate the swarm to maximize data revenue in a single orbital pass."
        ),
        "max_steps":   200,
        "max_score":   3000.0,
        "satellites":  sats,
        "stations": [
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
                "station_id":      "Station_Singapore",
                "position_deg":    130,
                "fov_deg":         12,
                "bandwidth_gbps":  2.0,
                "status":          "online",
                "queue_depth_gb":  0.0,
                "total_received_gb": 0.0,
                "offline_countdown": 0,
            },
            {
                "station_id":      "Station_Chile",
                "position_deg":    245,
                "fov_deg":         15,
                "bandwidth_gbps":  2.5,
                "status":          "online",
                "queue_depth_gb":  0.0,
                "total_received_gb": 0.0,
                "offline_countdown": 0,
            },
        ],
        "requests": requests,
        "completion_check_fn":  completion_check,
        "grader_breakdown_fn":  grader_breakdown,
    }
