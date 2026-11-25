
from __future__ import annotations

import os
import sys


_COMMON = os.path.join(os.path.dirname(__file__), "..", "darwin_common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

import agent_runner

ROBOT_NAME = "actor - critic"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "a2c.zip")

# Pasos de entrenamiento por defecto (se puede ajustar con TIMESTEPS en el env).
TRAIN_TIMESTEPS = int(os.environ.get("TIMESTEPS", "50000"))

# Hiperparámetros default-ish de A2C (valores por defecto de SB3 salvo seed):
#   learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=1.0,
#   ent_coef=0.0, vf_coef=0.5. Solo fijamos la semilla para reproducibilidad.
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
        from stable_baselines3 import A2C

        agent_runner.train(A2C, env, TRAIN_TIMESTEPS, MODEL_PATH, **HYPERPARAMS)
    else:
        from stable_baselines3 import A2C

        model = agent_runner.load(A2C, MODEL_PATH)
        metrics = agent_runner.run_race(model, env)
        print("[a2c] métricas de carrera:", metrics)


if __name__ == "__main__":
    main()
