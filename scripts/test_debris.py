import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.orbital_env import OrbitalEnv
from env.models import Action, ActionType, EventType

env = OrbitalEnv(task_id=2, seed=42)
obs, info = env.reset()

sat_id = obs.satellites[0].sat_id
print(f"Spawning debris for {sat_id}...")

event = env._event_engine._spawn(EventType.SPACE_DEBRIS, sat_id)
# Force duration to 2 steps for testing
event.duration_steps = 2
event.steps_remaining = 2
env._event_engine._active.append(event)

print(f"Active events: {[e.event_type for e in env._event_engine.active_events]}")

print("\n--- STEP 1: Not evading ---")
action1 = Action(action_type=ActionType.SLEEP_MODE, target_sat_id=sat_id, reasoning="Testing")
obs, rew, term, trunc, info = env.step([action1])
print(f"Reward: {rew}")
for e in env._event_engine.active_events:
    if e.event_type == "space_debris":
        print(f"Debris steps left: {e.steps_remaining}")

print("\n--- STEP 2: Evading via STATION_KEEPING ---")
action2 = Action(action_type=ActionType.STATION_KEEPING, target_sat_id=sat_id, reasoning="Evade!")
# We will send action2 for the target, and sleep for others so env.step processes it
actions = [action2]
for sat in obs.satellites:
    if sat.sat_id != sat_id:
        actions.append(Action(action_type=ActionType.SLEEP_MODE, target_sat_id=sat.sat_id, reasoning=""))

obs, rew, term, trunc, info = env.step(actions)
for ar in info["action_results"]:
    if ar.get("message") and "EVADED" in ar.get("message", ""):
        print(ar["message"])

print(f"Active events: {[e.event_type for e in env._event_engine.active_events]}")
print(f"Terminated: {term}")

print("\n--- STEP 3: Spawning another and letting it hit ---")
event = env._event_engine._spawn(EventType.SPACE_DEBRIS, sat_id)
event.duration_steps = 1
event.steps_remaining = 1
env._event_engine._active.append(event)
obs, rew, term, trunc, info = env.step([Action(action_type=ActionType.SLEEP_MODE, target_sat_id=sat_id, reasoning="Boom")])
print(f"Terminated: {term}")
print(f"Info keys: {info.keys()}")
if f"death_{sat_id}" in info:
    print(f"Death reason: {info[f'death_{sat_id}']}")
