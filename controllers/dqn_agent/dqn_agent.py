
from __future__ import annotations

import os
import sys


_COMMON = os.path.join(os.path.dirname(__file__), "..", "darwin_common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

import agent_runner

ROBOT_NAME = "DQN"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "dqn.zip")

# Pasos de entrenamiento por defecto (se puede ajustar con TIMESTEPS en el env).
TRAIN_TIMESTEPS = int(os.environ.get("TIMESTEPS", "50000"))

# Hiperparámetros default-ish de DQN (valores por defecto de SB3 salvo seed):
#   learning_rate=1e-4, buffer_size=1e6, learning_starts=100,
#   batch_size=32, gamma=0.99, train_freq=4, target_update_interval=1e4,
#   exploration_fraction=0.1. Solo fijamos la semilla para reproducibilidad.
HYPERPARAMS = {
    "seed": 0,
    "verbose": 1,
}


def _build_backend():
    
    from controller import Supervisor

    return Supervisor()


def main():
    mode = agent_runner.get_mode()
    backend = _build_backend()
    env = agent_runner.make_env(ROBOT_NAME, backend)

    if mode == "train":
        from stable_baselines3 import DQN

        agent_runner.train(DQN, env, TRAIN_TIMESTEPS, MODEL_PATH, **HYPERPARAMS)
    else:
        from stable_baselines3 import DQN

        model = agent_runner.load(DQN, MODEL_PATH)
        metrics = agent_runner.run_race(model, env)
        print("[dqn] métricas de carrera:", metrics)


if __name__ == "__main__":
    main()
