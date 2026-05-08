#!/usr/bin/env python3
"""WASD holonomic teleop for the mecanum explorer robot.

Key layout:
  W / S        : forward / backward
  A / D        : strafe left / right  (holonomic)
  Q / E        : rotate left / right
  SPACE        : full stop (immediate)
  + / =        : increase linear speed (+0.05 m/s)
  -            : decrease linear speed (-0.05 m/s)
  ] / [        : increase / decrease turn speed
  X or Ctrl-C  : stop robot and exit

Holding a key moves the robot. Releasing stops it automatically
within ~0.3 s without needing to press SPACE.
"""

import select
import sys
import termios
import tty
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ── Terminal colours ──────────────────────────────────────────────────────────
_CYAN   = '\033[96m'
_YELLOW = '\033[93m'
_GREEN  = '\033[92m'
_RESET  = '\033[0m'
_BOLD   = '\033[1m'

BANNER = f"""{_BOLD}{_CYAN}
╔══════════════════════════════════════════════╗
║   WASD Holonomic Teleop  —  Mecanum Robot   ║
╠══════════════════════════════════════════════╣
║  {_YELLOW}W{_CYAN} / {_YELLOW}S{_CYAN}       forward / backward              ║
║  {_YELLOW}A{_CYAN} / {_YELLOW}D{_CYAN}       strafe left / right  (holonomic)║
║  {_YELLOW}Q{_CYAN} / {_YELLOW}E{_CYAN}       rotate left / right             ║
║  {_YELLOW}SPACE{_CYAN}       full stop (immediate)            ║
║  {_YELLOW}+{_CYAN} / {_YELLOW}-{_CYAN}       linear speed up / down          ║
║  {_YELLOW}]{_CYAN} / {_YELLOW}[{_CYAN}       turn speed up / down            ║
║  {_YELLOW}X{_CYAN} / Ctrl-C  stop and exit                   ║
╚══════════════════════════════════════════════╝
{_RESET}"""

# key → (vx_sign, vy_sign, wz_sign)
KEY_MAP = {
    'w': ( 1.0,  0.0,  0.0),
    's': (-1.0,  0.0,  0.0),
    'a': ( 0.0,  1.0,  0.0),   # strafe left  (positive vy = left for ROS REP-103)
    'd': ( 0.0, -1.0,  0.0),   # strafe right
    'q': ( 0.0,  0.0,  1.0),   # rotate left  (positive wz = CCW)
    'e': ( 0.0,  0.0, -1.0),   # rotate right
}

KEY_LABELS = {
    'w': 'FWD  ', 's': 'BWD  ',
    'a': 'LEFT ', 'd': 'RIGHT',
    'q': 'ROT-L', 'e': 'ROT-R',
}

MIN_LINEAR = 0.05
MAX_LINEAR = 0.50
MIN_TURN   = 0.20
MAX_TURN   = 1.50

AUTO_STOP_TIMEOUT = 0.30   # seconds without a key → publish zero


def _getkey(timeout: float) -> str:
    """Non-blocking read: returns '' if no key arrives within timeout."""
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if r else ''


class WasdTeleop(Node):
    def __init__(self):
        super().__init__('wasd_teleop')
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._linear = 0.30    # m/s
        self._turn   = 0.80    # rad/s

    def _pub_vel(self, vx: float, vy: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x  = vx
        msg.linear.y  = vy
        msg.angular.z = wz
        self._pub.publish(msg)

    def run(self) -> None:
        print(BANNER)
        self._print_status('READY')

        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())

        vx = vy = wz = 0.0
        last_key_time = 0.0

        try:
            while rclpy.ok():
                key = _getkey(timeout=0.05)   # poll at 20 Hz

                if key in ('\x03', 'x', 'X'):  # Ctrl-C or X
                    self._pub_vel(0.0, 0.0, 0.0)
                    break

                elif key == ' ':
                    vx = vy = wz = 0.0
                    last_key_time = 0.0
                    self._pub_vel(0.0, 0.0, 0.0)
                    self._print_status('STOP')

                elif key in ('+', '='):
                    self._linear = min(self._linear + 0.05, MAX_LINEAR)
                    self._turn   = min(self._turn   + 0.10, MAX_TURN)
                    self._print_status('SPEED+')

                elif key == '-':
                    self._linear = max(self._linear - 0.05, MIN_LINEAR)
                    self._turn   = max(self._turn   - 0.10, MIN_TURN)
                    self._print_status('SPEED-')

                elif key == ']':
                    self._turn = min(self._turn + 0.10, MAX_TURN)
                    self._print_status('TURN+ ')

                elif key == '[':
                    self._turn = max(self._turn - 0.10, MIN_TURN)
                    self._print_status('TURN- ')

                elif key.lower() in KEY_MAP:
                    k = key.lower()
                    dx, dy, dw = KEY_MAP[k]
                    vx = dx * self._linear
                    vy = dy * self._linear
                    wz = dw * self._turn
                    last_key_time = time.monotonic()
                    label = KEY_LABELS[k]
                    self._print_status(
                        f'{label}  vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}')

                # Auto-stop when key not held
                if last_key_time and time.monotonic() - last_key_time > AUTO_STOP_TIMEOUT:
                    vx = vy = wz = 0.0
                    last_key_time = 0.0
                    self._print_status('STOP (auto)')

                self._pub_vel(vx, vy, wz)

        except KeyboardInterrupt:
            self._pub_vel(0.0, 0.0, 0.0)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print(f'\n{_GREEN}Teleop exited — robot stopped.{_RESET}\n')

    def _print_status(self, action: str) -> None:
        print(
            f'\r{_BOLD}[{action:<16}]{_RESET}'
            f'  lin={_YELLOW}{self._linear:.2f}{_RESET} m/s'
            f'  turn={_YELLOW}{self._turn:.2f}{_RESET} rad/s'
            f'                ',
            end='', flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = WasdTeleop()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
