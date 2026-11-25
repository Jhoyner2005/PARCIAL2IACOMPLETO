

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

try:  # config como módulo del paquete o como módulo suelto (pythonpath).
    from . import config
    from .race_env import CheckpointTracker
except ImportError:  # pragma: no cover - ruta usada bajo pythonpath plano
    import config
    from race_env import CheckpointTracker


def _ordered_from_coords(
    coords: Dict[int, Tuple[float, float]],
    order: Sequence[int],
    goal_y: float,
) -> List[Tuple[float, float]]:
    
    targets: List[Tuple[float, float]] = []
    for index in order:
        if index == 0:
            targets.append((0.0, float(goal_y)))
        elif index in coords:
            targets.append(coords[index])
        elif index in config.CHECKPOINT_COORDS:
            targets.append(config.CHECKPOINT_COORDS[index])
    return targets


class RaceWinnerDetector:
    

    def __init__(self, goal_y: float = config.GOAL_Y,
                 checkpoints: Optional[Sequence[Tuple[float, float]]] = None,
                 order: Sequence[int] = config.CHECKPOINT_ORDER,
                 radius: float = config.CHECKPOINT_RADIUS):
        self.goal_y = float(goal_y)
        self.order = tuple(order)
        self.radius = float(radius)
        if checkpoints is None:
            checkpoints = _ordered_from_coords(
                config.CHECKPOINT_COORDS, self.order, self.goal_y
            )
        self._targets: List[Tuple[float, float]] = [
            (float(x), float(y)) for (x, y) in checkpoints
        ]
        self.winner: Optional[str] = None
        self.finish_order: list = []  # nombres en orden de llegada
        self._trackers: Dict[str, CheckpointTracker] = {}

    def _tracker_for(self, name: str) -> CheckpointTracker:
        tracker = self._trackers.get(name)
        if tracker is None:
            tracker = CheckpointTracker(self._targets, radius=self.radius)
            self._trackers[name] = tracker
        return tracker

    def reset(self) -> None:
        """Reinicia el estado del detector para una nueva carrera."""
        self.winner = None
        self.finish_order = []
        self._trackers = {}

    def update(self, positions: Dict[str, Tuple[float, float]]) -> Optional[str]:
        
        completed: List[str] = []
        for name, xy in positions.items():
            if name in self.finish_order:
                continue
            tracker = self._tracker_for(name)
            result = tracker.update(float(xy[0]), float(xy[1]))
            if result["done"]:
                completed.append(name)
      
        for name in completed:
            if name not in self.finish_order:
                self.finish_order.append(name)
                if self.winner is None:
                    self.winner = name
        return self.winner


def run_race_until_winner(robots: Dict[str, object], max_steps: int,
                          goal_y: float = config.GOAL_Y,
                          action_fn=None) -> Dict[str, object]:
    
    checkpoints: Optional[Sequence[Tuple[float, float]]] = None
    for robot in robots.values():
        getter = getattr(robot, "ordered_checkpoints", None)
        if getter is not None:
            try:
                checkpoints = getter(order=config.CHECKPOINT_ORDER, goal_y=goal_y)
            except Exception:
                checkpoints = None
        break

    detector = RaceWinnerDetector(goal_y=goal_y, checkpoints=checkpoints)
    if action_fn is None:
        def action_fn(_name, robot):  # noqa: ANN001 - helper local
            robot.walk_forward()

    steps = 0
    for steps in range(1, int(max_steps) + 1):
        positions: Dict[str, Tuple[float, float]] = {}
        for name, robot in robots.items():
            action_fn(name, robot)
            robot.tick()
            px, py, _ = robot.get_position()
            positions[name] = (px, py)
        if detector.update(positions) is not None:
            break

    return {
        "winner": detector.winner,
        "finish_order": list(detector.finish_order),
        "steps": steps,
    }
