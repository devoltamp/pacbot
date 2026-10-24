#!/usr/bin/env python3
"""PacBot Task 2A - map-independent explorer / solver (v2 - Cleaned & Collision-Free)."""
import heapq
import itertools
import json
import math
import traceback

import paho.mqtt.client as mqtt

# ------------------------------------------------------------------ MQTT
MQTT_HOST = "localhost"
MQTT_PORT = 1883
TOPIC_SENSORS = "pacbot/sensors"
TOPIC_POSES = "pacbot/poses"
TOPIC_PELLETS = "pacbot/pellets"
TOPIC_RESULT = "pacbot/result"
TOPIC_WHEEL_VEL = "pacbot/wheel_vel"

# ------------------------------------------------------------------ maze
N = 15
PITCH = 0.22
HALF = PITCH / 2
FACE = HALF - 0.005                 # cell centre -> wall face
EXIT_CELL = (N - 1, N - 1)
EXIT_DIR = 3
NO_HIT = 2.0

# dir: 0=+x 1=+y 2=-x 3=-y ; STEP = (d_row, d_col)
STEP = ((0, 1), (-1, 0), (0, -1), (1, 0))
UX = (1.0, 0.0, -1.0, 0.0)
UY = (0.0, 1.0, 0.0, -1.0)

# ----------------------------------------------------------------- robot
WHEEL_R = 0.017
TRACK = 0.078
TICKS_PER_REV = 360
M_PER_TICK = 2 * math.pi * WHEEL_R / TICKS_PER_REV
WHEEL_MAX = 30.0
AXLE_BACK = 0.033

FRONT_AX = AXLE_BACK - 0.005        # forward ToF origin, ahead of axle
FRONT_NOM = FACE - FRONT_AX         # forward reading when axle is at cell centre (~0.077m)
SIDE_AX = AXLE_BACK + 0.036         # side ToF origin, ahead of axle
SIDE_Y = 0.005
FRONT_BASE = 0.068                  # lateral spacing of sl / sr

# ---------------------------------------------------------------- tuning
CRUISE = 0.20        # m/s on straights (safe & stable)
ACC = 0.8            # m/s^2 speed ramp-up
DEC = 1.2            # m/s^2 increased braking profile
V_CREEP = 0.03       # m/s final approach
STOP_TOL = 0.003     # m, axle-to-centre tolerance when stopping
SCAN_START = 0.090   # m before cell centre: start sampling walls
SCAN_END = 0.065     # m before cell centre: classify + decide
SCAN_FRAMES = 5      # frames of a stationary scan
SIDE_WALL = 0.16     # side reading below this = wall
FRONT_MARGIN = 0.09  # front wall tolerance above expected reading
EMERG_R = 0.095      # FIXED: Set higher than FRONT_NOM (0.077) to prevent boundary crashes
EXIT_OVER = 0.10     # m axle drives past centre of outside cell
SIDE_GUARD = 0.075   # side reading below this = too close to a wall
BIAS_FRAMES = 40     # stationary frames used to measure gyro bias
LAG_COMP = 0.04      # s, braking anticipation for motor lag
ALIGN_TOL = 0.006    # m, axle must be this close before pivot

KE = 6.0             # lateral error -> heading offset
KH = 6.0             # heading error -> turn rate
TURN_KP = 6.0
TURN_MAX = 2.5       # rad/s
TURN_MIN = 0.30
TURN_TOL = 0.012     # rad
SLEW_W = 15.0        # rad/s^2

TURN_COST = 1.6
TURN_COST_180 = 2.6
UNKNOWN_PENALTY = 0.2
REVISIT_PENALTY = 0.45   # prefer new corridors, never forbid necessary revisit
RETURN_TO_START = True   # pellet -> back to spawn cell -> exit


# --------------------------------------------------------------- helpers
def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def inside(r, c):
    return 0 <= r < N and 0 <= c < N


def cell_centre(cell):
    return cell[1] * PITCH, (N - 1 - cell[0]) * PITCH


def grid_angle(d):
    return wrap(d * math.pi / 2)


def wheels(v, omega):
    left = (v - omega * TRACK / 2) / WHEEL_R
    right = (v + omega * TRACK / 2) / WHEEL_R
    return clamp(left, -WHEEL_MAX, WHEEL_MAX), clamp(right, -WHEEL_MAX, WHEEL_MAX)


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def rd(s, key):
    v = s.get(key)
    if finite(v) and v > 0:
        return min(float(v), NO_HIT)
    return NO_HIT


# ================================================================ solver
class Solver:
    def __init__(self):
        self.t = 0.0
        self.spawn = None
        self.pellet = None
        self.done = False
        self.state = "WAIT"

        # axle pose (world) and mission bookkeeping
        self.x = self.y = 0.0
        self.theta = 0.0
        self.cell = None
        self.facing = 0
        self.start_cell = None
        self.collected = False
        self.stage = 0           # 0: fetch pellet, 1: return to spawn, 2: exit
        self.visits = {}         # cell -> number of arrivals

        # wall knowledge: True wall / False open / missing = unknown
        self.wall = {}
        self.locked = set()
        self.votes = {}          # edge id -> [open votes, wall votes]
        self.traversed = set()   # edge ids robot has driven through
        self._init_walls()

        # odometry / gyro
        self.prev_ticks = None
        self.ds = 0.0
        self.dth_e = 0.0
        self.gyro_mode = "pending"
        self.gyro_sign = 1.0
        self.gyro_bias = 0.0
        self.bias_sum = 0.0
        self.bias_n = 0
        self.acc_e = 0.0
        self.acc_g = 0.0
        self.v_prev = 0.0
        self.w_prev = 0.0

        # leg / scan / turn state
        self.buf = []
        self.leg_dir = 0
        self.target = (0, 0)
        self.scanned = False
        self.stopping = False
        self.continue_ok = False
        self.exit_leg = False
        self.turn_dir = 0
        self.turn_n = 0
        self.quiet = 0
        self.settle_n = 0
        self.cell_turns = 0
        self.align_n = 0

    def log(self, text):
        print(f"[{self.t:7.2f}s] {text}", flush=True)

    @staticmethod
    def edge_id(r, c, d):
        nr, nc = r + STEP[d][0], c + STEP[d][1]
        a, b = (r, c), (nr, nc)
        return (a, b) if a <= b else (b, a)

    def _init_walls(self):
        for i in range(N):
            for key in ((0, i, 1), (i, 0, 2), (i, N - 1, 0), (N - 1, i, 3)):
                self.wall[key] = True
                self.locked.add(key)
        ex = (EXIT_CELL[0], EXIT_CELL[1], EXIT_DIR)
        self.wall[ex] = False
        self.locked.add(ex)

    def observe_edge(self, r, c, d, is_wall, weight=1):
        key = (r, c, d)
        nr, nc = r + STEP[d][0], c + STEP[d][1]
        back = (nr, nc, (d + 2) % 4)
        if key in self.locked or (inside(nr, nc) and back in self.locked):
            return
        v = self.votes.setdefault(self.edge_id(r, c, d), [0, 0])
        v[1 if is_wall else 0] += weight
        state = v[1] >= v[0]
        self.wall[key] = state
        if inside(nr, nc):
            self.wall[back] = state

    def mark_open(self, cell, d):
        r, c = cell
        self.traversed.add(self.edge_id(r, c, d))
        self.wall[(r, c, d)] = False
        self.locked.add((r, c, d))
        nr, nc = r + STEP[d][0], c + STEP[d][1]
        if inside(nr, nc):
            self.wall[(nr, nc, (d + 2) % 4)] = False
            self.locked.add((nr, nc, (d + 2) % 4))

    def forget(self):
        for k in list(self.wall):
            if k not in self.locked:
                del self.wall[k]
        self.votes.clear()

    def advance(self, stage, cell):
        if stage == 0 and cell == self.pellet:
            stage = 1 if RETURN_TO_START else 2
        if stage == 1 and cell == self.start_cell:
            stage = 2
        return stage

    def dead_cells(self, protected):
        dead = set()
        changed = True
        while changed:
            changed = False
            for r in range(N):
                for c in range(N):
                    if (r, c) in dead or (r, c) in protected:
                        continue
                    n = 0
                    for d in range(4):
                        if self.wall.get((r, c, d)) is True:
                            continue
                        nr, nc = r + STEP[d][0], c + STEP[d][1]
                        if inside(nr, nc) and (nr, nc) in dead:
                            continue
                        n += 1
                    if n <= 1:
                        dead.add((r, c))
                        changed = True
        return dead

    def plan(self, cell, d, stage, prune=True):
        protected = {EXIT_CELL, tuple(cell)}
        if stage == 0 and self.pellet is not None:
            protected.add(self.pellet)
        if stage <= 1 and RETURN_TO_START and self.start_cell is not None:
            protected.add(self.start_cell)
        dead = self.dead_cells(protected) if prune else set()

        start = (cell[0], cell[1], d, stage)
        goal = (-1, -1, -1, 3)
        dist = {start: 0.0}
        prev = {}
        cnt = itertools.count()
        pq = [(0.0, next(cnt), start)]
        while pq:
            cst, _, st = heapq.heappop(pq)
            if cst > dist.get(st, 1e18):
                continue
            if st == goal:
                break
            r, c, dd, sg = st
            nbrs = [((r, c, (dd + 1) % 4, sg), TURN_COST),
                    ((r, c, (dd + 3) % 4, sg), TURN_COST),
                    ((r, c, (dd + 2) % 4, sg), TURN_COST_180)]
            w = self.wall.get((r, c, dd))
            if w is not True:
                nr, nc = r + STEP[dd][0], c + STEP[dd][1]
                if inside(nr, nc):
                    if (nr, nc) not in dead:
                        step = 1.0 + (UNKNOWN_PENALTY if w is None else 0.0)
                        if self.edge_id(r, c, dd) in self.traversed:
                            step += REVISIT_PENALTY
                        nbrs.append(((nr, nc, dd, self.advance(sg, (nr, nc))), step))
                elif sg == 2 and (r, c, dd) == (EXIT_CELL[0], EXIT_CELL[1], EXIT_DIR):
                    nbrs.append((goal, 1.0))
            for ns, cost in nbrs:
                nd = cst + cost
                if nd < dist.get(ns, 1e18):
                    dist[ns] = nd
                    prev[ns] = st
                    heapq.heappush(pq, (nd, next(cnt), ns))
        if goal not in dist:
            return None
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()
        if len(path) < 2:
            return None
        nxt = path[1]
        if nxt == goal or (nxt[0], nxt[1]) != tuple(cell):
            return ("forward", d)
        i = 1
        while i + 1 < len(path) and path[i + 1] != goal and (path[i + 1][0], path[i + 1][1]) == tuple(cell):
            i += 1
        return ("turn", path[i][2])

    def best_action(self, cell, d):
        stage = self.advance(self.stage, tuple(cell))
        act = self.plan(cell, d, stage, True)
        if act is None:
            act = self.plan(cell, d, stage, False)
        if act is None:
            self.log("no route on known map - forgetting unconfirmed walls")
            self.forget()
            act = self.plan(cell, d, stage, False)
        return act

    def rem_to(self, cell, d):
        cx, cy = cell_centre(cell)
        return (cx - self.x) * UX[d] + (cy - self.y) * UY[d]

    def front_correct(self, sl, sr, cell, d, a_pos, a_head, stopped=False):
        if abs(sl - sr) > 0.035:
            return False
        r = 0.5 * (sl + sr)
        if r > 0.45:
            return False
        phi = math.atan((sl - sr) / FRONT_BASE)
        rem_est = self.rem_to(cell, d)
        rem_meas = r * math.cos(phi) - FRONT_NOM
        if stopped:
            if abs(rem_meas) > 0.09:
                return False
        elif abs(rem_meas - rem_est) > 0.05:
            return False
        shift = (rem_est - rem_meas) * a_pos
        self.x += UX[d] * shift
        self.y += UY[d] * shift
        herr = wrap(grid_angle(d) + phi - self.theta)
        if abs(herr) < 0.2:
            self.theta = wrap(self.theta + a_head * herr)
        return True

    def lateral_measure(self, fl, fr, d):
        phi = wrap(self.theta - grid_angle(d))
        if abs(phi) > 0.4:
            return None
        cp, sp = math.cos(phi), math.sin(phi)
        es = []
        if fl < SIDE_WALL:
            es.append(FACE - fl * cp - SIDE_AX * sp - SIDE_Y * cp)
        if fr < SIDE_WALL:
            es.append(fr * cp - FACE - SIDE_AX * sp + SIDE_Y * cp)
        if not es:
            return None
        return sum(es) / len(es)

    def side_correct(self, fl, fr, cell, d, alpha):
        e = self.lateral_measure(fl, fr, d)
        if e is None:
            return False
        cx, cy = cell_centre(cell)
        nx, ny = -UY[d], UX[d]
        e_est = (self.x - cx) * nx + (self.y - cy) * ny
        if abs(e - e_est) > 0.06:
            return False
        shift = (e - e_est) * alpha
        self.x += nx * shift
        self.y += ny * shift
        return True

    def apply_scan(self, samples, cell, d, strong):
        msl = median([q[0] for q in samples])
        msr = median([q[1] for q in samples])
        mfl = median([q[2] for q in samples])
        mfr = median([q[3] for q in samples])
        if strong:
            self.front_correct(msl, msr, cell, d, 1.0, 1.0, True)
            self.side_correct(mfl, mfr, cell, d, 1.0)
        rem = self.rem_to(cell, d)
        wall_f = min(msl, msr) < rem + FRONT_NOM + FRONT_MARGIN
        wall_l = mfl < SIDE_WALL
        wall_r = mfr < SIDE_WALL
        r, c = cell
        w = 2 if strong else 1
        self.observe_edge(r, c, d, wall_f, w)
        self.observe_edge(r, c, (d + 1) % 4, wall_l, w)
        self.observe_edge(r, c, (d + 3) % 4, wall_r, w)
        return wall_f, wall_l, wall_r

    def on_poses(self, msg):
        bot, d = msg.get("bot"), msg.get("dir")
        if isinstance(bot, (list, tuple)) and d is not None and self.cell is None:
            self.spawn = (int(d), (int(bot[0]), int(bot[1])))
            self.log(f"spawn {self.spawn[1]} facing {self.spawn[0]}")

    def on_pellets(self, msg):
        p = msg.get("pellet")
        if p and self.cell is None:
            self.pellet = (int(p[0]), int(p[1]))
            self.log(f"pellet {self.pellet}")

    def stop(self):
        self.v_prev = 0.0
        self.w_prev = 0.0
        return 0.0, 0.0

    def step(self, s):
        dt = s.get("dt", 0.01)
        if not finite(dt) or dt <= 0:
            dt = 0.01
        self.t += dt

        ticks = (s.get("ticks_l", 0), s.get("ticks_r", 0))
        if self.prev_ticks is None:
            self.prev_ticks = ticks
        dl = (ticks[0] - self.prev_ticks[0]) * M_PER_TICK
        dr = (ticks[1] - self.prev_ticks[1]) * M_PER_TICK
        self.prev_ticks = ticks
        ds = 0.5 * (dl + dr)
        dth_e = (dr - dl) / TRACK
        self.ds, self.dth_e = ds, dth_e

        g = None
        gy = s.get("gyro")
        if isinstance(gy, (list, tuple)) and len(gy) >= 3 and finite(gy[2]):
            g = float(gy[2])

        if self.state == "INIT_BIAS":
            if g is not None:
                self.bias_sum += g
                self.bias_n += 1
            if self.bias_n >= BIAS_FRAMES:
                self.gyro_bias = self.bias_sum / self.bias_n
                self.state = "INIT_SCAN"
                self.buf = []
            return self.stop()

        dth = dth_e
        if g is not None and self.gyro_mode != "enc":
            gc = g - self.gyro_bias
            dth = self.gyro_sign * gc * dt
            if self.gyro_mode == "pending" and self.state == "TURN":
                self.acc_e += dth_e
                self.acc_g += dth
                if abs(self.acc_e) > 0.6:
                    ratio = abs(self.acc_g) / abs(self.acc_e)
                    if self.acc_g * self.acc_e > 0 and 0.6 < ratio < 1.7:
                        self.gyro_mode = "use"
                    elif self.acc_g * self.acc_e < 0 and 0.6 < ratio < 1.7:
                        self.gyro_sign = -self.gyro_sign
                        self.theta = wrap(self.theta - 2.0 * self.acc_g)
                        dth = -dth
                        self.gyro_mode = "use"
                    else:
                        self.theta = wrap(self.theta - self.acc_g + self.acc_e)
                        dth = dth_e
                        self.gyro_mode = "enc"
                self.log(f"heading source: {self.gyro_mode} (gyro sign {self.gyro_sign:+.0f})")

        mid = self.theta + dth / 2
        self.x += ds * math.cos(mid)
        self.y += ds * math.sin(mid)
        self.theta = wrap(self.theta + dth)

        if self.done:
            return self.stop()
        if self.cell is None:
            if self.spawn is None or self.pellet is None:
                return self.stop()
            self.init_pose()
            return self.stop()

        sens = (rd(s, "sl"), rd(s, "sr"), rd(s, "fl"), rd(s, "fr"))
        st = self.state
        if st in ("INIT_SCAN", "SCAN_STATIC"):
            return self.st_scan(sens)
        if st == "DRIVE":
            return self.st_drive(sens, dt)
        if st == "TURN":
            return self.st_turn(dt)
        if st == "ALIGN":
            return self.st_align(dt, min(sens[0], sens[1]))
        if st == "SETTLE":
            return self.st_settle()
        return self.stop()

    def init_pose(self):
        d, cell = self.spawn
        self.facing = d
        self.cell = cell
        self.start_cell = cell
        self.theta = grid_angle(d)
        cx, cy = cell_centre(cell)
        self.x = cx - AXLE_BACK * UX[d]
        self.y = cy - AXLE_BACK * UY[d]
        self.buf = []
        self.state = "INIT_BIAS"

    def st_scan(self, sens):
        self.buf.append(sens)
        if len(self.buf) < SCAN_FRAMES:
            return self.stop()
        wf, wl, wr = self.apply_scan(self.buf, self.cell, self.facing, True)
        self.buf = []
        self.log(f"cell {self.cell} dir {self.facing} walls F{int(wf)} L{int(wl)} R{int(wr)}")
        if self.state == "INIT_SCAN":
            self.target = self.cell
            self.leg_dir = self.facing
            self.scanned = True
            self.stopping = True
            self.continue_ok = False
            self.exit_leg = False
            self.state = "DRIVE"
        else:
            self.decide()
        return self.stop()

    def decide(self):
        act = self.best_action(self.cell, self.facing)
        if act is None:
            self.buf = []
            self.state = "SCAN_STATIC"
            return
        if act[0] == "turn":
            self.cell_turns += 1
            self.turn_dir = act[1]
            self.turn_n = 0
            self.align_n = 0
            if abs(self.rem_to(self.cell, self.facing)) > ALIGN_TOL:
                self.state = "ALIGN"
            else:
                self.state = "TURN"
        else:
            self.start_leg(self.facing)

    def start_leg(self, d):
        r, c = self.cell
        nr, nc = r + STEP[d][0], c + STEP[d][1]
        self.leg_dir = d
        self.target = (nr, nc)
        self.exit_leg = not inside(nr, nc)
        self.scanned = self.exit_leg
        self.continue_ok = self.exit_leg
        self.stopping = False
        self.buf = []
        self.cell_turns = 0
        self.state = "DRIVE"

    def arrive(self, cell):
        self.cell = cell
        self.visits[cell] = self.visits.get(cell, 0) + 1
        if cell == self.pellet and not self.collected:
            self.collected = True
            self.log(f"pellet cell {cell} reached")
        new_stage = self.advance(self.stage, cell)
        if new_stage != self.stage:
            self.stage = new_stage
            self.log(f"stage {new_stage} at {cell}")

    def finish_drive_scan(self):
        tgt, d = self.target, self.leg_dir
        self.apply_scan(self.buf, tgt, d, False)
        self.buf = []
        self.scanned = True
        act = self.best_action(tgt, d)
        if act is not None and act[0] == "forward":
            self.continue_ok = True
            self.stopping = False
        else:
            self.continue_ok = False
            self.stopping = True

    def do_pass(self):
        d = self.leg_dir
        previous = self.cell
        r, c = self.target
        if self.wall.get((r, c, d)) is True:
            self.stopping = True
            self.continue_ok = False
            return
        self.mark_open(previous, d)
        self.arrive(self.target)
        nr, nc = r + STEP[d][0], c + STEP[d][1]
        self.target = (nr, nc)
        self.exit_leg = not inside(nr, nc)
        self.scanned = self.exit_leg
        self.continue_ok = self.exit_leg
        self.stopping = False
        self.buf = []

    def st_drive(self, sens, dt):
        sl, sr, fl, fr = sens
        d = self.leg_dir
        tgt = self.target
        self.front_correct(sl, sr, tgt, d, 0.3, 0.05)
        self.side_correct(fl, fr, tgt, d, 0.25)
        rem = self.rem_to(tgt, d)
        rmin = min(sl, sr)

        # ABSOLUTE HARD BOUNDARY / CORNER SAFETY OVERRIDE
        if not self.exit_leg and (rmin < EMERG_R or rmin <= FRONT_NOM + 0.005):
            self.stopping = True
            self.continue_ok = False
            if rmin <= FRONT_NOM + 0.003:
                if inside(tgt[0], tgt[1]) and tgt != self.cell:
                    self.mark_open(self.cell, d)
                self.arrive(tgt)
                self.state = "SETTLE"
                self.quiet = 0
                self.settle_n = 0
                return self.stop()

        if not self.scanned:
            if rem <= SCAN_START:
                self.buf.append(sens)
            if rem <= SCAN_END:
                if not self.buf:
                    self.buf.append(sens)
                self.finish_drive_scan()

        if not self.exit_leg and not self.stopping and rmin < EMERG_R:
            self.stopping = True
            self.continue_ok = False

        if self.stopping:
            rem_eff = rem - LAG_COMP * self.v_prev
            if not self.exit_leg and rmin < FRONT_NOM:
                rem_eff = min(rem, rmin - FRONT_NOM)
            if rem_eff <= STOP_TOL:
                if inside(tgt[0], tgt[1]) and tgt != self.cell:
                    self.mark_open(self.cell, d)
                self.arrive(tgt)
                self.state = "SETTLE"
                self.quiet = 0
                self.settle_n = 0
                return self.stop()
            v = min(CRUISE, max(V_CREEP, math.sqrt(2 * DEC * rem_eff)))
        else:
            if self.exit_leg:
                if rem <= -EXIT_OVER:
                    self.log("left the maze")
                    self.done = True
                    return self.stop()
            elif rem <= 0.0:
                if self.continue_ok:
                    self.do_pass()
                else:
                    self.stopping = True
                d = self.leg_dir
                tgt = self.target
                rem = self.rem_to(tgt, d)
            v = CRUISE
            if self.stopping:
                v = V_CREEP

        cx, cy = cell_centre(tgt)
        lat = (self.x - cx) * (-UY[d]) + (self.y - cy) * UX[d]
        e_meas = self.lateral_measure(fl, fr, d)
        if e_meas is not None and abs(e_meas - lat) < 0.06:
            lat = e_meas
        ga = grid_angle(d)
        des = ga + clamp(-KE * lat, -0.25, 0.25)
        herr = wrap(des - self.theta)
        omega = clamp(KH * herr, -2.5, 2.5)

        guard = 0.0
        if fl < SIDE_GUARD:
            guard = max(guard, SIDE_GUARD - fl)
            omega -= 25.0 * (SIDE_GUARD - fl)
        if fr < SIDE_GUARD:
            guard = max(guard, SIDE_GUARD - fr)
            omega += 25.0 * (SIDE_GUARD - fr)
        if abs(wrap(ga - self.theta)) > 0.5:
            v = 0.0
        v *= max(0.3, 1.0 - 2.0 * abs(herr))
        if guard > 0.0:
            v = min(v, max(0.08, 0.2 - 4.0 * guard))
        if not self.exit_leg and rmin < 0.25:
            v = min(v, max(0.0, 6.0 * (rmin - 0.05)))
        v = clamp(min(v, self.v_prev + ACC * dt), 0.0, CRUISE)
        self.v_prev = v
        self.w_prev = 0.0
        return wheels(v, clamp(omega, -3.0, 3.0))

    def st_align(self, dt, rmin=NO_HIT):
        self.align_n += 1
        d = self.facing
        rem = self.rem_to(self.cell, d)
        if abs(rem) < 0.003 or self.align_n > 400:
            self.state = "SETTLE"
            self.quiet = 0
            self.settle_n = 0
            return self.stop()
        v = math.copysign(clamp(abs(2.5 * rem), 0.02, 0.06), rem)
        if v > 0 and rmin < FRONT_NOM + 0.004:
            self.state = "SETTLE"
            self.quiet = 0
            self.settle_n = 0
            return self.stop()
        omega = clamp(KH * wrap(grid_angle(d) - self.theta), -1.5, 1.5)
        self.v_prev = 0.0
        return wheels(v, omega)

    def st_turn(self, dt):
        self.turn_n += 1
        err = wrap(grid_angle(self.turn_dir) - self.theta)
        if abs(err) < TURN_TOL or self.turn_n > 900:
            self.facing = self.turn_dir
            self.state = "SETTLE"
            self.quiet = 0
            self.settle_n = 0
            return self.stop()
        w = clamp(TURN_KP * err, -TURN_MAX, TURN_MAX)
        if abs(w) < TURN_MIN:
            w = math.copysign(TURN_MIN, err)
        lim = abs(self.w_prev) + SLEW_W * dt
        w = clamp(w, -lim, lim)
        self.w_prev = w
        self.v_prev = 0.0
        return wheels(0.0, w)

    def st_settle(self):
        self.settle_n += 1
        if abs(self.ds) < 4e-4 and abs(self.dth_e) < 0.004:
            self.quiet += 1
        else:
            self.quiet = 0
        if self.quiet >= 3 or self.settle_n > 60:
            err = wrap(grid_angle(self.facing) - self.theta)
            if abs(err) > 0.04 and self.turn_n <= 900:
                self.turn_dir = self.facing
                self.turn_n = 0
                self.state = "TURN"
            else:
                self.buf = []
                self.state = "SCAN_STATIC"
        return self.stop()


def _client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        return mqtt.Client()


def main():
    solver = Solver()
    client = _client()

    def send(cli, left, right):
        cli.publish(TOPIC_WHEEL_VEL, json.dumps({"left": float(left), "right": float(right)}))

    def on_message(cli, _userdata, msg):
        if msg.topic == TOPIC_SENSORS:
            try:
                data = json.loads(msg.payload)
                cmd = solver.step(data)
            except Exception:
                traceback.print_exc()
                cmd = (0.0, 0.0)
            send(cli, *cmd)
        elif msg.topic == TOPIC_POSES:
            if msg.payload:
                solver.on_poses(json.loads(msg.payload))
        elif msg.topic == TOPIC_PELLETS:
            if msg.payload:
                solver.on_pellets(json.loads(msg.payload))
        elif msg.topic == TOPIC_RESULT:
            solver.log(f"result: {msg.payload.decode()}")
            solver.done = True
            send(cli, 0.0, 0.0)
            cli.disconnect()

    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT)
    for topic in (TOPIC_POSES, TOPIC_PELLETS, TOPIC_RESULT, TOPIC_SENSORS):
        client.subscribe(topic)
    print("connected, waiting for the simulator", flush=True)
    client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
