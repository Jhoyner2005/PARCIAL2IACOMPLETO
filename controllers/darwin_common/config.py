"""Configuración compartida optimizada de la carrera DARwIn-OP."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

# --- Geometría del mundo (eje de carrera = y) -------------------------------

GOAL_Y: float = 9.0
START_YAW: float = 0.0

START_POSITIONS: Dict[str, Tuple[float, float, float]] = {
    "actor - critic": (0.70, 5.70, 0.24),
    "ppo": (0.61, 8.11, 0.24),
    "DQN": (0.56, 10.08, 0.24),
    "agente propio": (0.55, 12.32, 0.24),
}

ACTION_MAP: Dict[int, str] = {
    0: "walk_forward",
    1: "turn_left",
    2: "turn_right",
}

# --- Checkpoints de la vuelta (malla densa de 16 puntos) -------------------

CHECKPOINT_RADIUS: float = 2.4
CHECKPOINT_PROGRESS_WEIGHT: float = 22.0
CHECKPOINT_BONUS: float = 25.0

CHECKPOINT_ORDER: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
TOTAL_TARGETS: int = len(CHECKPOINT_ORDER)

# 2.6 m + 1.5 m = 4.1 m de margen máximo para cubrir el carril exterior (y=12.32)
OFF_TRACK_PENALTY_WEIGHT: float = 0.8
OFF_TRACK_TOLERANCE: float = 3.8

CHECKPOINT_COORDS: Dict[int, Tuple[float, float]] = {
    0: (0.0, 9.0),       # Meta / Salida
    1: (4.5, 8.8),       # Recta superior derecha
    2: (8.8, 8.0),       # Entrada curva 1
    3: (12.0, 5.2),      # Curva 1 - tramo A
    4: (13.5, 1.5),      # Ápice curva 1
    5: (12.0, -3.5),     # Curva 1 - tramo B
    6: (8.8, -8.0),      # Salida curva 1
    7: (5.0, -8.8),      # Recta inferior derecha
    8: (1.5, -9.0),      # Centro recta inferior
    9: (-4.5, -8.8),     # Recta inferior izquierda
    10: (-8.5, -8.25),   # Entrada curva 2
    11: (-11.8, -4.0),   # Curva 2 - tramo A
    12: (-13.5, 1.5),    # Ápice curva 2
    13: (-11.8, 5.5),    # Curva 2 - tramo B
    14: (-8.5, 8.2),     # Salida curva 2
    15: (-4.2, 8.8),     # Recta final hacia meta
}

# --- Safe learning ----------------------------------------------------------

SAFETY_RADIUS: float = 0.85
OBSTACLE_PENALTY_WEIGHT: float = 0.9
OTHER_ROBOT_PENALTY_WEIGHT: float = 0.0
HARD_COLLISION_DISTANCE: float = 0.15

FALL_TILT_THRESHOLD: float = 0.6
ACCEL_CENTER: float = 512.0
ACCEL_FALL_TOLERANCE: float = 80.0
FALL_HEIGHT_Z: float = 0.14
FALL_ACCEL_PERSIST_STEPS: int = 100

# --- Episodio ---------------------------------------------------------------

MAX_EPISODE_STEPS: int = 1600

# --- Locomoción: gait sinusoidal (DARwIn-OP) --------------------------------

MOTOR_NAMES = (
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR", "ArmLowerL",
    "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR", "LegUpperL",
    "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR", "FootL",
    "Neck", "Head",
)

STANDING_POSE: Dict[str, float] = {
    "ShoulderR": 0.8, "ShoulderL": -0.8,
    "ArmUpperR": 0.4, "ArmUpperL": -0.4,
    "ArmLowerR": -0.6, "ArmLowerL": 0.6,
    "PelvYR": 0.0, "PelvYL": 0.0,
    "PelvR": 0.0, "PelvL": 0.0,
    "LegUpperR": -0.6, "LegUpperL": 0.6,
    "LegLowerR": 1.2, "LegLowerL": -1.2,
    "AnkleR": -0.6, "AnkleL": 0.6,
    "FootR": 0.0, "FootL": 0.0,
    "Neck": 0.0, "Head": 0.0,
}

GAIT_PERIOD: int = 40
GAIT_FREQUENCY: float = 1.0 / GAIT_PERIOD
GAIT_X_AMPLITUDE: float = 1.0
GAIT_A_AMPLITUDE: float = 0.6
GAIT_HIP_AMPLITUDE: float = 0.35
GAIT_KNEE_AMPLITUDE: float = 0.4
GAIT_ANKLE_AMPLITUDE: float = 0.2
GAIT_LATERAL_SWAY: float = 0.12
GAIT_LEG_PHASE_OFFSET: float = math.pi
MOTOR_VELOCITY: float = 8.0

JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "ShoulderR": (-3.14, 3.14), "ShoulderL": (-3.14, 3.14),
    "ArmUpperR": (-0.68, 2.30), "ArmUpperL": (-2.25, 0.77),
    "ArmLowerR": (-1.65, 1.16), "ArmLowerL": (-1.18, 1.63),
    "PelvYR": (-2.42, 0.66), "PelvYL": (-0.69, 2.50),
    "PelvR": (-1.01, 1.01), "PelvL": (-1.00, 0.93),
    "LegUpperR": (-1.77, 0.45), "LegUpperL": (-0.50, 1.68),
    "LegLowerR": (-0.02, 2.25), "LegLowerL": (-2.25, 0.03),
    "AnkleR": (-1.24, 1.38), "AnkleL": (-1.39, 1.22),
    "FootR": (-0.68, 1.04), "FootL": (-1.02, 0.60),
    "Neck": (-1.81, 1.81), "Head": (-0.36, 0.94),
}

JOINT_LIMIT_MARGIN: float = 0.05
STABILIZATION_STEPS: int = 5

# --- Modo de locomoción: Plan B (cinemático optimizado para carrera) --------

LOCOMOTION_MODE: str = "kinematic"
KINEMATIC_LINEAR_STEP: float = 0.09
KINEMATIC_ANGULAR_STEP: float = 0.035
KINEMATIC_COLLISION_RADIUS: float = 0.55
KINEMATIC_STAND_Z: float = 0.24

KINEMATIC_STANDING_POSE: Dict[str, float] = {
    "ShoulderR": 0.0, "ShoulderL": 0.0,
    "ArmUpperR": 0.0, "ArmUpperL": 0.0,
    "ArmLowerR": 0.0, "ArmLowerL": 0.0,
    "PelvYR": 0.0, "PelvYL": 0.0,
    "PelvR": 0.0, "PelvL": 0.0,
    "LegUpperR": 0.0, "LegUpperL": 0.0,
    "LegLowerR": 0.1, "LegLowerL": -0.1,
    "AnkleR": 0.0, "AnkleL": 0.0,
    "FootR": 0.0, "FootL": 0.0,
    "Neck": 0.0, "Head": 0.0,
}


def resolve_locomotion_mode() -> str:
    valid = {"gait", "kinematic"}
    raw = os.environ.get("LOCOMOTION_MODE", LOCOMOTION_MODE)
    mode = str(raw).strip().lower()
    if mode not in valid:
        return LOCOMOTION_MODE
    return mode


# --- Normalización de la observación ----------------------------------------

MAX_DIST: float = 60.0

# --- Pesos de recompensa ----------------------------------------------------

PROGRESS_WEIGHT: float = 22.0
GOAL_BONUS: float = 250.0
FALL_PENALTY: float = 50.0
COLLISION_PENALTY: float = 40.0
STEP_PENALTY: float = 0.01


@dataclass
class RaceParams:
    goal_y: float = GOAL_Y
    start_yaw: float = START_YAW
    start_positions: Dict[str, Tuple[float, float, float]] = field(
        default_factory=lambda: dict(START_POSITIONS)
    )
    safety_radius: float = SAFETY_RADIUS
    obstacle_penalty_weight: float = OBSTACLE_PENALTY_WEIGHT
    other_robot_penalty_weight: float = OTHER_ROBOT_PENALTY_WEIGHT
    hard_collision_distance: float = HARD_COLLISION_DISTANCE
    fall_tilt_threshold: float = FALL_TILT_THRESHOLD
    max_episode_steps: int = MAX_EPISODE_STEPS
    max_dist: float = MAX_DIST
    progress_weight: float = PROGRESS_WEIGHT
    goal_bonus: float = GOAL_BONUS
    fall_penalty: float = FALL_PENALTY
    collision_penalty: float = COLLISION_PENALTY
    step_penalty: float = STEP_PENALTY
    checkpoint_radius: float = CHECKPOINT_RADIUS
    checkpoint_progress_weight: float = CHECKPOINT_PROGRESS_WEIGHT
    checkpoint_bonus: float = CHECKPOINT_BONUS
    off_track_penalty_weight: float = OFF_TRACK_PENALTY_WEIGHT
    off_track_tolerance: float = OFF_TRACK_TOLERANCE
    checkpoint_order: Tuple[int, ...] = field(
        default_factory=lambda: tuple(CHECKPOINT_ORDER)
    )
    checkpoint_coords: Dict[int, Tuple[float, float]] = field(
        default_factory=lambda: dict(CHECKPOINT_COORDS)
    )

    def override(self, **kwargs) -> "RaceParams":
        return replace(self, **kwargs)


DEFAULT_PARAMS = RaceParams()