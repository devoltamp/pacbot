import json
import math

import paho.mqtt.client as mqtt

MQTT_HOST = "localhost"
MQTT_PORT = 1883
TOPIC_SENSORS = "pacbot/sensors"      # simulator publishes, this file subscribes
TOPIC_WHEEL_VEL = "pacbot/wheel_vel"  # this file publishes, simulator subscribes

# Robot geometry - set these from your task document / robot model.
WHEEL_RADIUS = 0.02       # m
WHEEL_BASE = 0.08         # m, distance between the two wheels

# Speed.
CRUISE_SPEED = 0.40       # m/s on straights (raise once it runs clean)
MIN_SPEED_FRAC = 0.35     # fraction of cruise kept while creeping up to a wall
TURN_RATE = 2.0           # rad/s, max in-place turning rate
TURN_RATE_MIN = 0.35      # rad/s, so the turn never stalls near the target
TURN_TOL = 0.02           # rad, a turn is "done" inside this window (~1.1 deg)

# Wall following.
WALL_DIST = None

K_LATERAL = 25.0
K_HEADING = 4.0
K_INTEGRAL = 2.0

K_TURN = 4.0
MAX_STEER = 2.0

# Geometry of decisions.
OPEN_RATIO = 1.8          # side reading > OPEN_RATIO * WALL_DIST => "opening"
FRONT_STOP_TRIM = 0.0     # m, + stops earlier, - stops later (see notes)
GOAL_OVERRUN = 0.05       # m to keep driving after leaving the last corridor

# Gyro sign: +1 if a left (counter-clockwise) turn gives positive gyro z.
YAW_SIGN = +1

FAR = 10.0
CAL_SAMPLES = 5
CAL_TIMEOUT = 40
DEFAULT_WALL_DIST = 0.05
EXIT_TIME = 0.15
# =====================================================================


def _clean(x):
    """Sensor value -> float, with inf/nan/None mapped to FAR."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return FAR
    return x if math.isfinite(x) else FAR


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class WallFollower:
    """step() takes one sensor message, returns (left_vel, right_vel) rad/s."""

    def __init__(self):
        self.state = "CALIBRATE"
        self.heading = 0.0        # integrated gyro heading (rad, CCW +)
        self.target = 0.0         # heading the current corridor runs along
        self.samples = []
        self.msgs = 0
        self.wall = WALL_DIST or DEFAULT_WALL_DIST
        self.open_thr = OPEN_RATIO * self.wall
        self.front_stop = self.wall
        self.slow_zone = 2.0 * self.wall
        self.exit_time = 0.0
        self.overrun = 0.0
        self.turns = 0

    # ---- helpers ----------------------------------------------------
    @staticmethod
    def _wheels(v, w):
        """Body velocity (m/s, rad/s CCW) -> wheel angular velocities."""
        half = 0.5 * WHEEL_BASE * w
        return (v - half) / WHEEL_RADIUS, (v + half) / WHEEL_RADIUS

    def _calibrate(self, fl, fr):
        self.msgs += 1
        if fl < 1.0 and fr < 1.0:
            self.samples.append(0.5 * (fl + fr))
        if len(self.samples) >= CAL_SAMPLES:
            s = sorted(self.samples)
            measured = s[len(s) // 2]
        elif self.msgs >= CAL_TIMEOUT:
            measured = DEFAULT_WALL_DIST
        else:
            return False
        self.wall = WALL_DIST or measured
        self.open_thr = OPEN_RATIO * self.wall
        # Stop with the robot centre half a cell from the wall ahead.  Half a
        # cell equals the centred side-wall reading, whenever front and side
        # sensors sit at the same offset from the robot centre.
        self.front_stop = self.wall + FRONT_STOP_TRIM
        self.slow_zone = 2.0 * self.wall
        self.target = self.heading   # start pose is aligned with the corridor
        print(f"[calib] wall_dist={self.wall:.3f} m  open>{self.open_thr:.3f} m  "
              f"front_stop={self.front_stop:.3f} m")
        return True

    # ---- main entry -------------------------------------------------
    def step(self, fl, fr, sl, sr, yaw_rate, dt):
        fl, fr, sl, sr = _clean(fl), _clean(fr), _clean(sl), _clean(sr)
        self.heading += YAW_SIGN * yaw_rate * dt
        front = min(sl, sr)

        if self.state == "CALIBRATE":
            if self._calibrate(fl, fr):
                self.state = "DRIVE"
            return 0.0, 0.0

        if self.state == "DONE":
            return 0.0, 0.0

        if self.state == "TURN":
            err = _wrap(self.target - self.heading)
            if abs(err) < TURN_TOL:
                self.state = "DRIVE"
                return 0.0, 0.0
            w = _clamp(K_TURN * err, -TURN_RATE, TURN_RATE)
            if abs(w) < TURN_RATE_MIN:
                w = math.copysign(TURN_RATE_MIN, err)
            return self._wheels(0.0, w)

        left_wall = fl < self.open_thr
        right_wall = fr < self.open_thr

        if self.state == "FINISH":      # coast out of the goal passage
            self.overrun -= CRUISE_SPEED * dt
            if self.overrun <= 0.0:
                self.state = "DONE"
                print("[done] left the maze - stopping")
                return 0.0, 0.0
            return self._drive(fl, fr, left_wall, right_wall, CRUISE_SPEED, dt)

        # ---- DRIVE ----
        # Both sides open for a while = we are out of the maze.
        outside = not (left_wall or right_wall) and front > 4.0 * self.wall
        self.exit_time = self.exit_time + dt if outside else 0.0
        if self.exit_time >= EXIT_TIME:
            self.state = "FINISH"
            self.overrun = GOAL_OVERRUN
            return self._drive(fl, fr, left_wall, right_wall, CRUISE_SPEED, dt)

        # Wall ahead: stop, choose a side, turn in place.
        if front <= self.front_stop:
            if fl > self.open_thr and fl >= fr:
                turn = math.pi / 2          # left
            elif fr > self.open_thr:
                turn = -math.pi / 2         # right
            else:
                turn = math.pi              # dead end
            self.target = _wrap(self.target + turn)
            self.turns += 1
            self.state = "TURN"
            print(f"[turn {self.turns}] {math.degrees(turn):+.0f} deg  "
                  f"(front={front:.3f} fl={fl:.3f} fr={fr:.3f})")
            return 0.0, 0.0

        # Slow down smoothly as the wall ahead gets close.
        frac = _clamp((front - self.front_stop) / self.slow_zone, MIN_SPEED_FRAC, 1.0)
        return self._drive(fl, fr, left_wall, right_wall, CRUISE_SPEED * frac, dt)

    def _drive(self, fl, fr, left_wall, right_wall, v, dt):
        # Lateral error: + means "too far from the left wall" (steer left).
        if left_wall and right_wall:
            e = 0.5 * (fl - fr)
        elif left_wall:
            e = fl - self.wall
        elif right_wall:
            e = self.wall - fr
        else:
            e = 0.0
        e = _clamp(e, -self.wall, self.wall)
        # Walls are the ground truth for "straight": nudge the gyro heading
        # estimate so a persistent lateral error cannot build up.
        if left_wall or right_wall:
            self.heading -= K_INTEGRAL * e * dt
        # Heading error: + means "rotated left of the corridor axis".
        th = _wrap(self.heading - self.target)
        w = _clamp(K_LATERAL * e - K_HEADING * th, -MAX_STEER, MAX_STEER)
        v *= 1.0 - 0.5 * abs(e) / self.wall     # ease off when far off-line
        return self._wheels(v, w)


# =====================================================================
#  MQTT
# =====================================================================
controller = WallFollower()


def _mqtt_client():
    # paho-mqtt >= 2.0 requires picking a callback API version explicitly.
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        return mqtt.Client()


def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())

    fl = data["fl"]            # distance to the left side wall (m)
    fr = data["fr"]            # distance to the right side wall (m)
    sl = data["sl"]            # distance ahead, left of centre (m)
    sr = data["sr"]            # distance ahead, right of centre (m)
    yaw_rate = data["gyro"][2]  # rad/s about z
    dt = data["dt"]            # s, simulator timestep

    left_vel, right_vel = controller.step(fl, fr, sl, sr, yaw_rate, dt)

    client.publish(TOPIC_WHEEL_VEL, json.dumps({
        "left": float(left_vel), "right": float(right_vel),
    }))


def main():
    client = _mqtt_client()
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT)
    client.subscribe(TOPIC_SENSORS)
    client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
