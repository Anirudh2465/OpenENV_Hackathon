"""
OpenEnv-Orbital-Command | tasks/__init__.py
Task registry — maps task IDs to configuration factories.
"""
from .easy.task1_eclipse     import build_task1
from .medium.task2_storage     import build_task2
from .hard.task3_crosslink   import build_task3
from .hard.task4_swarm       import build_task4
from .extreme.task5_emergency   import build_task5

TASK_REGISTRY = {
    1: build_task1,
    2: build_task2,
    3: build_task3,
    4: build_task4,
    5: build_task5,
}

def get_task_config(task_id: int) -> dict:
    factory = TASK_REGISTRY.get(task_id)
    if factory is None:
        raise ValueError(f"Unknown task_id {task_id}. Available: {list(TASK_REGISTRY)}")
    return factory()
