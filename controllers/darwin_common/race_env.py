from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from . import config
    from .darwin_robot import DarwinRobot
except ImportError:
    import config
    from darwin_robot import DarwinRobot


def _segment_distance(px: float, py: float,
                      ax: float, ay: float,
                      bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


class CheckpointTracker:
    def __init__(self, targets: Sequence[Tuple[float, float]],
                 radius: float = config.CHECKPOINT_RADIUS,
                 origin: Optional[Tuple[float, float]] = None):
        self.targets: List[Tuple[float, float]] = [
            (float(x), float(y)) for (x, y) in targets
        ]
        self.radius = float(radius)
        if origin is not None:
            self.origin = (float(origin[0]), float(origin[1]))
        elif self.targets:
            self.origin = self.targets[-1]
        else:
            self.origin = (0.0, 0.0)
        self.reset()

    def reset(self) -> None:
        self.reached_count = 0
        self.prev_dist: Optional[float] = None
        self.done = False

    def current_target(self) -> Optional[Tuple[float, float]]:
        if self.done or self.reached_count >= len(self.targets):
            return None
        return self.targets[self.reached_count]

    def _predecessor(self) -> Tuple[float, float]:
        if self.reached_count <= 0:
            return self.origin
        return self.targets[self.reached_count - 1]

    def current_segment(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        target = self.current_target()
        if target is None:
            return None
        return (self._predecessor(), target)

    def lap_progress(self) -> float:
        if not self.targets:
            return 1.0
        return self.reached_count / len(self.targets)

    def finish_point(self) -> Tuple[float, float]:
        if self.targets:
            return self.targets[-1]
        return self.origin

    def update(self, px: float, py: float) -> Dict[str, object]:
        if self.done:
            return {
                "progress": 0.0,
                "reached": False,
                "done": True,
                "reached_count": self.reached_count,
                "target_ordinal": self.reached_count,
                "off_segment": None,
            }

        target = self.current_target()
        segment = self.current_segment()
        dist = math.hypot(px - target[0], py - target[1])
        progress = (self.prev_dist - dist) if self.prev_dist is not None else 0.0

        reached = dist <= self.radius
        if reached:
            self.reached_count += 1
            if self.reached_count >= len(self.targets):
                self.done = True
                self.prev_dist = None
            else:
                nxt = self.targets[self.reached_count]
                self.prev_dist = math.hypot(px - nxt[0], py - nxt[1])
        else:
            self.prev_dist = dist

        return {
            "progress": float(progress),
            "reached": bool(reached),
            "done": bool(self.done),
            "reached_count": int(self.reached_count),
            "target_ordinal": int(self.reached_count),
            "off_segment": segment,
        }


class DarwinRaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        backend,
        robot_name: str = "DQN",
        params: Optional[config.RaceParams] = None,
        start_position: Optional[Tuple[float, float, float]] = None,
        goal_y: Optional[float] = None,
    ):
        super().__init__()
        self.params = params or config.RaceParams()
        if goal_y is not None:
            self.params = self.params.override(goal_y=goal_y)
        self.robot_name = robot_name
        self.robot = DarwinRobot(backend, robot_name)

        if start_position is not None:
            self.start_position = tuple(start_position)
        else:
            self.start_position = self.params.start_positions.get(
                robot_name, (0.0, 0.0, 0.3)
            )

        targets = self.robot.ordered_checkpoints(
            order=self.params.checkpoint_order, goal_y=self.params.goal_y
        )
        self.tracker = CheckpointTracker(
            targets, radius=self.params.checkpoint_radius
        )

        self.action_space = spaces.Discrete(len(config.ACTION_MAP))
        high = np.ones(7, dtype=np.float32)
        low = np.array([0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        x, y, z = self.start_position
        self.robot.reset_pose(x, y, z)
        self._steps = 0
        self.tracker.reset()
        px, py, _ = self.robot.get_position()
        target = self.tracker.current_target()
        if target is not None:
            self.tracker.prev_dist = math.hypot(px - target[0], py - target[1])
        return self._get_obs(), self._base_info()

    def step(self, action):
        action = int(action)
        method = config.ACTION_MAP[action]
        getattr(self.robot, method)()
        self.robot.tick()
        self._steps += 1

        px, py, _ = self.robot.get_position()
        result = self.tracker.update(px, py)
        progress = float(result["progress"])

        nearest_obstacle = self._nearest_distance(self.robot.find_barriers())
        nearest_other = self._nearest_distance(self.robot.find_others())

        angle_err = abs(self._angle_to_target())
        alignment_bonus = 0.25 * math.cos(angle_err)
        stall_penalty = 0.5 if progress <= 0.001 else 0.0

        reward = (
            self.params.checkpoint_progress_weight * progress
            + alignment_bonus
            - stall_penalty
            - self.params.step_penalty
        )
        terminated = False
        truncated = False

        if result["reached"] and not result["done"]:
            reward += self.params.checkpoint_bonus

        if px < -1.5 and self.tracker.reached_count == 0:
            reward -= 30.0
            terminated = True

        # Nota: antes había aquí una terminación por "banda radial"
        # (dist_from_centerline fuera de [4.2, 13.6]) que mataba el episodio al
        # cruzar el ápice de las curvas (checkpoints 3->4->5, x~13.5, y~0),
        # impidiendo completar la vuelta. Se eliminó: la contención en pista la
        # ejerce la penalización off_track de abajo (distancia al segmento de la
        # línea de carrera), que es correcta para el óvalo y NO termina el
        # episodio. El límite de duración real es MAX_EPISODE_STEPS.
        off_segment = result["off_segment"]
        if off_segment is not None:
            (ax, ay), (bx, by) = off_segment
            deviation = _segment_distance(px, py, ax, ay, bx, by)
            if deviation > self.params.off_track_tolerance:
                reward -= self.params.off_track_penalty_weight * (
                    deviation - self.params.off_track_tolerance
                )

        fall = self.robot.is_fallen()
        collision = False
        margin_violations = 0

        if nearest_obstacle is not None and nearest_obstacle < self.params.safety_radius:
            intrusion = self.params.safety_radius - nearest_obstacle
            reward -= self.params.obstacle_penalty_weight * intrusion
            margin_violations += 1

        if nearest_other is not None and nearest_other < self.params.safety_radius:
            intrusion = self.params.safety_radius - nearest_other
            reward -= self.params.other_robot_penalty_weight * intrusion
            margin_violations += 1

        success = bool(result["done"])
        if success:
            reward += self.params.goal_bonus
            terminated = True

        if fall:
            reward -= self.params.fall_penalty
            terminated = True
        if collision:
            reward -= self.params.collision_penalty
            terminated = True

        if self._steps >= self.params.max_episode_steps:
            truncated = True

        info = self._base_info()
        info.update(
            {
                "fall": fall,
                "collision": collision,
                "margin_violations": margin_violations,
                "progress": float(progress),
                "success": success,
                "checkpoints_reached": int(result["reached_count"]),
            }
        )
        return self._get_obs(), float(reward), terminated, truncated, info

    def _target_point(self) -> Tuple[float, float]:
        target = self.tracker.current_target()
        if target is not None:
            return target
        if self.tracker.targets:
            return self.tracker.targets[-1]
        return (0.0, self.params.goal_y)

    def _dist_to_target(self) -> float:
        px, py, _ = self.robot.get_position()
        tx, ty = self._target_point()
        return math.hypot(px - tx, py - ty)

    def _dist_to_finish(self) -> float:
        px, py, _ = self.robot.get_position()
        fx, fy = self.tracker.finish_point()
        return math.hypot(px - fx, py - fy)

    def _angle_to_target(self) -> float:
        px, py, _ = self.robot.get_position()
        tx, ty = self._target_point()
        dx = tx - px
        dy = ty - py
        target_heading = math.atan2(dy, dx) if (dx or dy) else 0.0
        heading = self.robot.get_orientation()
        diff = target_heading - heading
        return math.atan2(math.sin(diff), math.cos(diff))

    def _nearest_distance(self, nodes: Dict[str, Tuple[float, float]]) -> Optional[float]:
        if not nodes:
            return None
        px, py, _ = self.robot.get_position()
        return min(math.hypot(px - x, py - y) for (x, y) in nodes.values())

    def _get_obs(self) -> np.ndarray:
        dist = self._dist_to_target()
        angle = self._angle_to_target()
        heading = self.robot.get_orientation()
        nearest_obstacle = self._nearest_distance(self.robot.find_barriers())
        nearest_other = self._nearest_distance(self.robot.find_others())

        max_dist = self.params.max_dist
        obs = np.array(
            [
                min(dist / max_dist, 1.0),
                max(-1.0, min(angle / math.pi, 1.0)),
                max(-1.0, min(heading / math.pi, 1.0)),
                max(0.0, min(self.tracker.lap_progress(), 1.0)),
                max(0.0, min(self._dist_to_finish() / max_dist, 1.0)),
                1.0 if nearest_obstacle is None else min(nearest_obstacle / max_dist, 1.0),
                1.0 if nearest_other is None else min(nearest_other / max_dist, 1.0),
            ],
            dtype=np.float32,
        )
        return obs

    def _base_info(self) -> dict:
        px, py, _ = self.robot.get_position()
        return {"robot_name": self.robot_name, "x": float(px), "y": float(py), "steps": self._steps}