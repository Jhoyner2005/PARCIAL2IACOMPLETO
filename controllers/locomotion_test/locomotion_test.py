
from __future__ import annotations

import os
import sys

# Permite importar el paquete común (config, darwin_robot) tanto bajo pytest
# como al ejecutarse como controlador de Webots.
_COMMON = os.path.join(os.path.dirname(__file__), "..", "darwin_common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

import config  # noqa: E402
from darwin_robot import DarwinRobot  # noqa: E402

# Comando -> método de locomoción de alto nivel.
_COMMANDS = {
    "stand": "stand",
    "forward": "walk_forward",
    "left": "turn_left",
    "right": "turn_right",
}


def _read_command(argv=None) -> str:
    
    if argv is None:
        argv = sys.argv
    raw = None
    if len(argv) > 1 and argv[1].strip():
        raw = argv[1]
    elif os.environ.get("LOCO_CMD", "").strip():
        raw = os.environ.get("LOCO_CMD")
    value = (raw or "stand").strip().lower()
    if value not in _COMMANDS:
        print("[locomotion_test] comando '%s' no reconocido; uso 'stand'." % value)
        value = "stand"
    return value


def _motor_limits(motor):
    """Lee (min, max) reales del motor de forma defensiva; None si no se puede."""
    get_min = getattr(motor, "getMinPosition", None)
    get_max = getattr(motor, "getMaxPosition", None)
    if not callable(get_min) or not callable(get_max):
        return None
    try:
        return float(get_min()), float(get_max())
    except Exception:
        return None


def _print_motor_diagnostics(robot: DarwinRobot) -> None:
    
    print("[locomotion_test] --- límites reales de motores (min, max) ---")
    motors = robot.locomotion.motors
    for name in config.MOTOR_NAMES:
        motor = motors.get(name)
        limits = _motor_limits(motor) if motor is not None else None
        if limits is None:
            limits = config.JOINT_LIMITS.get(name)
        if limits is None:
            print("[locomotion_test]   %-10s min=n/a    max=n/a" % name)
        else:
            print("[locomotion_test]   %-10s min=%.4f max=%.4f" % (name, limits[0], limits[1]))
    print("[locomotion_test] --- fin límites de motores ---")


def _accel_values(robot: DarwinRobot):
   
    accel = getattr(robot, "_accel", None)
    if accel is None:
        return None
    try:
        return accel.getValues()
    except Exception:
        return None


def main() -> None:
    from controller import Supervisor

    supervisor = Supervisor()
    robot_name = os.environ.get("LOCO_ROBOT", "DQN")
    robot = DarwinRobot(supervisor, robot_name)

    if not robot.is_supervisor():
        print("[locomotion_test] ERROR: " + robot.SUPERVISOR_REQUIRED_MSG,
              file=sys.stderr)
        sys.exit(1)

    command = _read_command()
    method = _COMMANDS[command]
    mode = config.resolve_locomotion_mode()
    print("[locomotion_test] robot=%s comando=%s modo=%s" % (robot_name, command, mode))
    if mode == "kinematic":
        print("[locomotion_test] modo kinematic (Plan B): el Supervisor arrastra "
              "al robot de pie; is_fallen() siempre False, sin ajuste de marcha.")
    print("[locomotion_test] motores resueltos: %d/%d" % (
        len(robot.locomotion.motors), len(config.MOTOR_NAMES)))

    
    _print_motor_diagnostics(robot)

   
    robot.reset_pose(*robot.get_position())

    
    _, _, standing_z = robot.get_position()
    print("[locomotion_test] z de pie resuelta tras estabilización: %.4f" % standing_z)

    step = 0
    while True:
        
        getattr(robot, method)()
        if robot.tick() == -1:
            break

        if step % 10 == 0:
            x, y, z = robot.get_position()
            heading = robot.get_orientation()
            accel = _accel_values(robot)
            fallen = robot.is_fallen()
            print(
                "[locomotion_test] step=%d pos=(%.3f, %.3f, %.3f) heading=%.3f "
                "accel=%s fallen=%s" % (step, x, y, z, heading, accel, fallen)
            )
        step += 1


if __name__ == "__main__":
    main()
