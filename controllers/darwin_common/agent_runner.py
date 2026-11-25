

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

try:  # entorno como módulo del paquete o como módulo suelto (pythonpath).
    from .race_env import DarwinRaceEnv
except ImportError:  # pragma: no cover - ruta usada bajo pythonpath plano
    from race_env import DarwinRaceEnv


def make_env(robot_name: str, backend, **overrides) -> DarwinRaceEnv:
    
    return DarwinRaceEnv(backend, robot_name=robot_name, **overrides)


def train(algo_cls, env, timesteps: int, save_path: str, **hp):
   
    model = algo_cls("MlpPolicy", env, **hp)
    model.learn(total_timesteps=timesteps)
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    model.save(save_path)
    return model


def load(algo_cls, path: str):
   
    return algo_cls.load(path)


def run_race(model, env, deterministic: bool = True) -> Dict[str, object]:
    
    obs, _info = env.reset()
    cumulative_reward = 0.0
    steps = 0
    falls = 0
    collisions = 0
    success = False
    checkpoints_reached = 0

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _state = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += float(reward)
        steps += 1
        if info.get("fall"):
            falls += 1
        if info.get("collision"):
            collisions += 1
        if info.get("success"):
            success = True
        checkpoints_reached = max(checkpoints_reached, int(info.get("checkpoints_reached", 0)))

    return {
        "success": bool(success),
        "steps": int(steps),
        "cumulative_reward": float(cumulative_reward),
        "falls": int(falls),
        "collisions": int(collisions),
        "checkpoints_reached": int(checkpoints_reached),
    }


def get_mode(argv: Optional[list] = None, default: str = "race") -> str:
   
    if argv is None:
        argv = sys.argv[1:]

    for arg in argv:
        value = str(arg).strip().lower()
        if value in ("train", "race"):
            return value

    env_mode = os.environ.get("MODE")
    if env_mode is not None:
        value = env_mode.strip().lower()
        if value in ("train", "race"):
            return value

    return default
