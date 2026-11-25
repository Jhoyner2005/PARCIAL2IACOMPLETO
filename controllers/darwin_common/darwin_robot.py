from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

try:
    from . import config
except ImportError:
    import config


def _iter_world_nodes(supervisor):
    get_root = getattr(supervisor, "getRoot", None)
    if get_root is None:
        return
    root = get_root()
    if root is None:
        return
    try:
        children = root.getField("children")
        count = children.getCount()
    except Exception:
        return
    for i in range(count):
        try:
            node = children.getMFNode(i)
        except Exception:
            node = None
        if node is not None:
            yield node


def _node_name(node) -> Optional[str]:
    getter = getattr(node, "getNameString", None)
    if getter is not None:
        try:
            return getter()
        except Exception:
            pass
    try:
        name_field = node.getField("name")
    except Exception:
        return None
    if name_field is None:
        return None
    try:
        return name_field.getSFString()
    except Exception:
        value = getattr(name_field, "_value", None)
        if isinstance(value, (list, tuple)) and value:
            return value[0]
    return None


def _node_xy(node) -> Optional[Tuple[float, float]]:
    try:
        trans = node.getField("translation").getSFVec3f()
    except Exception:
        return None
    return float(trans[0]), float(trans[1])


def _clamp_to_limits(limits: Optional[Tuple[float, float]], angle: float) -> float:
    if limits is None:
        return float(angle)
    lo, hi = limits
    margin = config.JOINT_LIMIT_MARGIN
    lo_safe = lo + margin
    hi_safe = hi - margin
    if lo_safe > hi_safe:
        return float((lo + hi) / 2.0)
    return float(max(lo_safe, min(hi_safe, angle)))


def _clamp_joint(name: str, angle: float) -> float:
    return _clamp_to_limits(config.JOINT_LIMITS.get(name), angle)


class _GaitEngine:
    def __init__(self):
        self.phase = 0.0
        self.x_amplitude = 0.0
        self.a_amplitude = 0.0

    def set_amplitudes(self, x: float, a: float) -> None:
        self.x_amplitude = float(x)
        self.a_amplitude = float(a)

    def reset_phase(self) -> None:
        self.phase = 0.0

    def advance(self) -> None:
        self.phase += 2.0 * math.pi * config.GAIT_FREQUENCY
        two_pi = 2.0 * math.pi
        if self.phase >= two_pi:
            self.phase -= two_pi

    def standing_pose(self) -> Dict[str, float]:
        return {name: _clamp_joint(name, angle) for name, angle in config.STANDING_POSE.items()}

    def compute_targets(self, phase: Optional[float] = None) -> Dict[str, float]:
        if phase is None:
            phase = self.phase

        pose = dict(config.STANDING_POSE)

        x = self.x_amplitude
        a = self.a_amplitude
        hip = config.GAIT_HIP_AMPLITUDE
        knee = config.GAIT_KNEE_AMPLITUDE
        ankle = config.GAIT_ANKLE_AMPLITUDE
        sway = config.GAIT_LATERAL_SWAY
        offset = config.GAIT_LEG_PHASE_OFFSET

        phase_r = phase
        phase_l = phase + offset

        sin_r = math.sin(phase_r)
        sin_l = math.sin(phase_l)
        knee_r = abs(math.sin(phase_r))
        knee_l = abs(math.sin(phase_l))
        lateral = math.cos(phase)

        pose["LegUpperR"] += x * hip * sin_r
        pose["LegUpperL"] += x * hip * sin_l
        pose["LegLowerR"] += x * knee * knee_r
        pose["LegLowerL"] += -x * knee * knee_l
        pose["AnkleR"] += x * ankle * sin_r
        pose["AnkleL"] += x * ankle * sin_l

        pose["PelvYR"] += a * 0.3
        pose["PelvYL"] += a * 0.3

        pose["PelvR"] += sway * lateral
        pose["PelvL"] += sway * lateral
        pose["AnkleR"] += -sway * lateral
        pose["AnkleL"] += -sway * lateral

        return {name: _clamp_joint(name, angle) for name, angle in pose.items()}


class _LocomotionBackend:
    def __init__(self, supervisor, config_path: str = ""):
        self.supervisor = supervisor
        self.engine = _GaitEngine()
        self.mode = config.resolve_locomotion_mode()
        self._kin_action: Optional[str] = None
        self.motors: Dict[str, object] = {}
        self.motor_limits: Dict[str, Tuple[float, float]] = {}
        self._fetch_motors()
        self.available = bool(self.motors)
        if not self.available:
            warnings.warn(
                "Ningún motor del DARwIn-OP resolvió; locomoción en modo no-op.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _fetch_motors(self) -> None:
        getter = getattr(self.supervisor, "getDevice", None)
        if getter is None:
            getter = getattr(self.supervisor, "getMotor", None)
        if getter is None:
            return
        for name in config.MOTOR_NAMES:
            try:
                motor = getter(name)
            except Exception:
                motor = None
            if motor is None:
                continue
            set_vel = getattr(motor, "setVelocity", None)
            if set_vel is not None:
                try:
                    set_vel(config.MOTOR_VELOCITY)
                except Exception:
                    pass
            self.motors[name] = motor
            self.motor_limits[name] = self._read_motor_limits(name, motor)

    @staticmethod
    def _read_motor_limits(name: str, motor) -> Optional[Tuple[float, float]]:
        lo = hi = None
        get_min = getattr(motor, "getMinPosition", None)
        get_max = getattr(motor, "getMaxPosition", None)
        if callable(get_min) and callable(get_max):
            try:
                lo = float(get_min())
                hi = float(get_max())
            except Exception:
                lo = hi = None
        if lo is None or hi is None or lo >= hi:
            return config.JOINT_LIMITS.get(name)
        return (lo, hi)

    def clamp(self, name: str, angle: float) -> float:
        if name in self.motor_limits:
            return _clamp_to_limits(self.motor_limits[name], angle)
        return _clamp_joint(name, angle)

    def walk_forward(self) -> None:
        self.engine.set_amplitudes(config.GAIT_X_AMPLITUDE, 0.0)
        if self.mode == "kinematic":
            if self._kinematic_blocked_forward():
                self._kin_action = None
                return
            _queue(self.supervisor, "walk_forward")
            self._kin_action = "walk_forward"
            return
        _queue(self.supervisor, "walk_forward")

    def turn_left(self) -> None:
        self.engine.set_amplitudes(0.0, config.GAIT_A_AMPLITUDE)
        _queue(self.supervisor, "turn_left")
        if self.mode == "kinematic":
            self._kin_action = "turn_left"

    def turn_right(self) -> None:
        self.engine.set_amplitudes(0.0, -config.GAIT_A_AMPLITUDE)
        _queue(self.supervisor, "turn_right")
        if self.mode == "kinematic":
            self._kin_action = "turn_right"

    def stand(self) -> None:
        self.engine.set_amplitudes(0.0, 0.0)
        _queue(self.supervisor, None)
        if self.mode == "kinematic":
            self._kin_action = None

    def apply_targets(self, targets: Dict[str, float]) -> None:
        for name, motor in self.motors.items():
            angle = targets.get(name)
            if angle is None:
                continue
            set_pos = getattr(motor, "setPosition", None)
            if set_pos is None:
                continue
            try:
                set_pos(self.clamp(name, float(angle)))
            except Exception:
                pass

    def _apply_kinematic_motion(self, action: Optional[str]) -> None:
        if action is None or hasattr(self.supervisor, "queue_action"):
            return
        get_self = getattr(self.supervisor, "getSelf", None)
        if get_self is None:
            return
        node = get_self()
        if node is None:
            return
        try:
            trans_field = node.getField("translation")
            rot_field = node.getField("rotation")
            rot = rot_field.getSFRotation()
            theta = rot[3] if rot[2] >= 0 else -rot[3]
            if action == "walk_forward":
                d = config.KINEMATIC_LINEAR_STEP
                x, y, z = trans_field.getSFVec3f()
                x += math.cos(theta) * d
                y += math.sin(theta) * d
                trans_field.setSFVec3f([float(x), float(y), float(z)])
            elif action == "turn_left":
                theta += config.KINEMATIC_ANGULAR_STEP
                rot_field.setSFRotation([0.0, 0.0, 1.0, float(theta)])
                d = config.KINEMATIC_LINEAR_STEP * 0.7
                x, y, z = trans_field.getSFVec3f()
                trans_field.setSFVec3f([float(x + math.cos(theta) * d), float(y + math.sin(theta) * d), float(z)])
            elif action == "turn_right":
                theta -= config.KINEMATIC_ANGULAR_STEP
                rot_field.setSFRotation([0.0, 0.0, 1.0, float(theta)])
                d = config.KINEMATIC_LINEAR_STEP * 0.7
                x, y, z = trans_field.getSFVec3f()
                trans_field.setSFVec3f([float(x + math.cos(theta) * d), float(y + math.sin(theta) * d), float(z)])
        except Exception:
            pass

    def _apply_upright_pose(self) -> None:
        get_self = getattr(self.supervisor, "getSelf", None)
        if get_self is None:
            return
        try:
            node = get_self()
        except Exception:
            return
        if node is None:
            return
        try:
            trans_field = node.getField("translation")
            rot_field = node.getField("rotation")
            rot = rot_field.getSFRotation()
            theta = rot[3] if rot[2] >= 0 else -rot[3]
            rot_field.setSFRotation([0.0, 0.0, 1.0, float(theta)])
            x, y, _z = trans_field.getSFVec3f()
            trans_field.setSFVec3f(
                [float(x), float(y), float(config.KINEMATIC_STAND_Z)]
            )
        except Exception:
            pass

    def _self_xy_theta(self) -> Optional[Tuple[float, float, float]]:
        get_self = getattr(self.supervisor, "getSelf", None)
        if get_self is None:
            return None
        try:
            node = get_self()
        except Exception:
            return None
        if node is None:
            return None
        try:
            x, y, _z = node.getField("translation").getSFVec3f()
            rot = node.getField("rotation").getSFRotation()
        except Exception:
            return None
        theta = rot[3] if rot[2] >= 0 else -rot[3]
        return float(x), float(y), float(theta)

    def _obstacles_xy(self):
        get_self = getattr(self.supervisor, "getSelf", None)
        self_node = None
        if get_self is not None:
            try:
                self_node = get_self()
            except Exception:
                self_node = None
        robot_names = set(config.START_POSITIONS.keys())
        points = []
        for node in _iter_world_nodes(self.supervisor):
            if self_node is not None and node is self_node:
                continue
            name = _node_name(node)
            if name is None:
                continue
            is_barrier = name.startswith("valla")
            is_other_robot = name in robot_names
            if not (is_barrier or is_other_robot):
                continue
            xy = _node_xy(node)
            if xy is not None:
                points.append(xy)
        return points

    def _kinematic_blocked_forward(self) -> bool:
        pose = self._self_xy_theta()
        if pose is None:
            return False
        x, y, theta = pose
        d = config.KINEMATIC_LINEAR_STEP
        nx = x + math.cos(theta) * d
        ny = y + math.sin(theta) * d
        radius = config.KINEMATIC_COLLISION_RADIUS
        for (bx, by) in self._obstacles_xy():
            if math.hypot(nx - bx, ny - by) < radius:
                return True
        return False

    def update(self) -> None:
        if self.mode == "kinematic":
            self.hold_kinematic_standing()
            self._apply_upright_pose()
            self._apply_kinematic_motion(self._kin_action)
            self._reset_self_physics()
            self._kin_action = None
            return
        self.engine.advance()
        self.apply_targets(self.engine.compute_targets())

    def _reset_self_physics(self) -> None:
        get_self = getattr(self.supervisor, "getSelf", None)
        if get_self is None:
            return
        node = get_self()
        reset_physics = getattr(node, "resetPhysics", None)
        if reset_physics is None:
            return
        try:
            reset_physics()
        except Exception:
            pass

    def hold_standing(self) -> None:
        self.engine.set_amplitudes(0.0, 0.0)
        self.apply_targets(self.engine.standing_pose())

    def hold_kinematic_standing(self) -> None:
        self.apply_targets(
            {name: self.clamp(name, angle)
             for name, angle in config.KINEMATIC_STANDING_POSE.items()}
        )


def _queue(supervisor, action: Optional[str]) -> None:
    if hasattr(supervisor, "queue_action"):
        supervisor.queue_action(action)


class DarwinRobot:
    SUPERVISOR_REQUIRED_MSG = (
        "Este robot no es Supervisor: getSelf() devolvió None. Añade "
        "'supervisor TRUE' al nodo del robot en el .wbt."
    )

    def __init__(self, backend, name: str, config_path: str = ""):
        self.backend = backend
        self.name = name
        self.time_step = int(backend.getBasicTimeStep())
        self.locomotion = _LocomotionBackend(backend, config_path)
        try:
            self._self_node = backend.getSelf()
        except Exception:
            self._self_node = None
        self._accel = self._safe_sensor("Accelerometer")
        self._gyro = self._safe_sensor("Gyro")
        self._accel_out_of_band = 0

    def is_supervisor(self) -> bool:
        return self._self_node is not None

    def _require_self_node(self):
        if self._self_node is None:
            raise RuntimeError(self.SUPERVISOR_REQUIRED_MSG)
        return self._self_node

    def _safe_sensor(self, device_name: str):
        getter = getattr(self.backend, "getDevice", None)
        sensor = None
        if getter is not None:
            try:
                sensor = getter(device_name)
            except Exception:
                sensor = None
        if sensor is not None and hasattr(sensor, "enable"):
            try:
                sensor.enable(self.time_step)
            except Exception:
                pass
        return sensor

    def walk_forward(self) -> None:
        self.locomotion.walk_forward()

    def turn_left(self) -> None:
        self.locomotion.turn_left()

    def turn_right(self) -> None:
        self.locomotion.turn_right()

    def stand(self) -> None:
        self.locomotion.stand()

    def tick(self) -> int:
        self.locomotion.update()
        return self.backend.step(self.time_step)

    def stabilize(self) -> None:
        self.locomotion.engine.reset_phase()
        for _ in range(int(config.STABILIZATION_STEPS)):
            self.locomotion.hold_standing()
            self.backend.step(self.time_step)

    def get_position(self) -> Tuple[float, float, float]:
        trans = self._require_self_node().getField("translation").getSFVec3f()
        return float(trans[0]), float(trans[1]), float(trans[2])

    def get_orientation(self) -> float:
        rot = self._require_self_node().getField("rotation").getSFRotation()
        axis_z = rot[2]
        angle = rot[3]
        return float(angle if axis_z >= 0 else -angle)

    def is_fallen(self) -> bool:
        if self.locomotion.mode == "kinematic":
            return False

        if self._accel is not None:
            try:
                values = self._accel.getValues()
            except Exception:
                values = None
            if values is not None:
                deviation = max(abs(float(v) - config.ACCEL_CENTER) for v in values)
                if deviation > config.ACCEL_FALL_TOLERANCE:
                    self._accel_out_of_band += 1
                    if self._accel_out_of_band >= config.FALL_ACCEL_PERSIST_STEPS:
                        return True
                else:
                    self._accel_out_of_band = 0

        try:
            rot = self._self_node.getField("rotation").getSFRotation()
            tilt_angle = abs(float(rot[3]))
            axis_x, axis_y = float(rot[0]), float(rot[1])
            effective_tilt = tilt_angle * math.hypot(axis_x, axis_y)
            if effective_tilt > config.FALL_TILT_THRESHOLD:
                return True
        except Exception:
            pass

        try:
            trans = self._self_node.getField("translation").getSFVec3f()
            if float(trans[2]) < config.FALL_HEIGHT_Z:
                return True
        except Exception:
            pass

        return False

    def reset_pose(self, x: float, y: float, z: Optional[float] = None) -> None:
        self_node = self._require_self_node()
        trans_field = self_node.getField("translation")
        current = trans_field.getSFVec3f()
        new_z = current[2] if z is None else z
        trans_field.setSFVec3f([float(x), float(y), float(new_z)])
        start_yaw = float(getattr(config, "START_YAW", 0.0))
        self_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, start_yaw])

        self.locomotion.engine.reset_phase()
        self.locomotion._kin_action = None
        self.locomotion.hold_standing()

        reset_physics = getattr(self_node, "resetPhysics", None)
        if reset_physics is not None:
            try:
                reset_physics()
            except Exception:
                pass

        self.stabilize()

    def find_others(self) -> Dict[str, Tuple[float, float]]:
        robot_names = set(config.START_POSITIONS.keys())
        others: Dict[str, Tuple[float, float]] = {}
        for node in _iter_world_nodes(self.backend):
            name = _node_name(node)
            if name is None or name == self.name or name not in robot_names:
                continue
            xy = _node_xy(node)
            if xy is not None:
                others[name] = xy
        return others

    def find_barriers(self) -> Dict[str, Tuple[float, float]]:
        barriers: Dict[str, Tuple[float, float]] = {}
        for node in _iter_world_nodes(self.backend):
            name = _node_name(node)
            if name is None or not name.startswith("valla"):
                continue
            xy = _node_xy(node)
            if xy is not None:
                barriers[name] = xy
        return barriers

    def find_checkpoints(self) -> Dict[int, Tuple[float, float]]:
        checkpoints: Dict[int, Tuple[float, float]] = {}
        for node in _iter_world_nodes(self.backend):
            name = _node_name(node)
            if name is None or not name.startswith("checkpoint"):
                continue
            suffix = name[len("checkpoint"):]
            if not suffix.isdigit():
                continue
            index = int(suffix)
            xy = _node_xy(node)
            if xy is not None:
                checkpoints[index] = xy
        return checkpoints

    def ordered_checkpoints(
        self,
        order: Tuple[int, ...] = None,
        goal_y: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        if order is None:
            order = config.CHECKPOINT_ORDER
        if goal_y is None:
            goal_y = config.GOAL_Y

        scanned = {} if len(order) > 8 else self.find_checkpoints()
        fallback = config.CHECKPOINT_COORDS

        resolved: Dict[int, Tuple[float, float]] = {}
        missing: List[int] = []
        for index in set(order):
            if index == 0:
                resolved[0] = (0.0, float(goal_y))
                continue
            if index in scanned:
                resolved[index] = scanned[index]
            elif index in fallback:
                resolved[index] = fallback[index]
                missing.append(index)
            else:
                missing.append(index)
        if missing:
            warnings.warn(
                f"Checkpoints no encontrados en el mundo {sorted(missing)}; "
                "usando config.CHECKPOINT_COORDS como fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
        return [resolved[index] for index in order if index in resolved]