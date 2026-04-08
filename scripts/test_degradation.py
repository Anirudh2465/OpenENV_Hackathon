import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.orbital_env import OrbitalEnv
from env.models import Action, ActionType

env = OrbitalEnv(task_id=2, seed=42)
obs, info = env.reset()

sat_id = obs.satellites[0].sat_id

print("\\n--- Testing Thermal Degradation ---")
# Force high thermal to trigger degradation
env._satellites[sat_id]["thermal_level"] = 90.0

action = Action(action_type=ActionType.SLEEP_MODE, target_sat_id=sat_id, reasoning="")
obs, _, _, _, _ = env.step([action])
# Now health should be 100 - THERMAL_DEGRADATION_RATE (0.15)
sat_health = obs.satellites[0].health_index
print(f"Health after 1 step at 90 thermal: {sat_health}%")

print("\\n--- Testing Low Battery Degradation ---")
env._satellites[sat_id]["battery_level"] = 5.0
env._satellites[sat_id]["thermal_level"] = 50.0  # reset thermal
obs, _, term, _, info = env.step([action])
sat_health2 = obs.satellites[0].health_index
print(f"Health after low battery step: {sat_health2}%")

print("\\n--- Testing MAINTENANCE_CYCLE ---")
# Reset battery to high so we can afford maintenance
env._satellites[sat_id]["battery_level"] = 100.0
env._satellites[sat_id]["in_sunlight"] = True

maint_action = Action(action_type=ActionType.MAINTENANCE_CYCLE, target_sat_id=sat_id, reasoning="Fix it")
obs, _, _, _, _ = env.step([maint_action])
sat_health3 = obs.satellites[0].health_index
print(f"Health after maintenance: {sat_health3}%")

