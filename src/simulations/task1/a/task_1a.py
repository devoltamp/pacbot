#!/usr/bin/env python3
"""
"""

import json
import time

import paho.mqtt.client as mqtt
from collections import deque

MAZE_ROWS = 13
MAZE_COLS = 13
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
POSE_TOPIC = "robot/pose"

# wall bit per side, OR'd together
WALL_N, WALL_E, WALL_S, WALL_W = 0x1, 0x2, 0x4, 0x8

WALLS = [
    [12, 6, 12, 6, 13, 4, 0, 4, 5, 6, 12, 5, 6],
    [10, 11, 10, 10, 12, 3, 8, 2, 12, 1, 1, 6, 10],
    [8, 5, 3, 8, 1, 6, 9, 2, 9, 6, 13, 2, 10],
    [10, 12, 4, 3, 12, 1, 4, 0, 6, 9, 4, 2, 10],
    [10, 10, 8, 5, 3, 13, 2, 10, 9, 6, 10, 11, 10],
    [8, 3, 9, 4, 5, 6, 8, 1, 6, 10, 9, 5, 2],
    [10, 12, 4, 1, 6, 10, 9, 6, 10, 8, 5, 5, 2],
    [8, 1, 2, 12, 3, 10, 12, 3, 9, 2, 12, 6, 10],
    [9, 6, 10, 10, 12, 1, 2, 12, 5, 1, 0, 1, 3],
    [14, 8, 1, 1, 3, 12, 1, 3, 12, 4, 2, 12, 6],
    [8, 0, 4, 7, 12, 3, 12, 6, 10, 9, 1, 2, 10],
    [10, 10, 9, 6, 10, 12, 2, 8, 1, 7, 12, 0, 2],
    [9, 1, 5, 1, 1, 3, 8, 1, 5, 5, 3, 9, 3],
]

# the 2 known exits: (row, col, facing)
EXIT_CELLS = [
    (0, 6, 'south'),
    (MAZE_ROWS - 1, 6, 'north'),
]

BOT_CMD_TOPIC = "bot/cmd"
PELLETS_TOPIC = "pellets/pose"
CMD_VEL_TOPIC = "robot/cmd_vel"

# yaw -> dr, dc, wall bit. 0=EAST, 90=NORTH, 180=WEST, 270=SOUTH
HEADING_DELTA = {
    0.0:   (0, 1, WALL_E),
    90.0:  (1, 0, WALL_N),
    180.0: (0, -1, WALL_W),
    270.0: (-1, 0, WALL_S),
}


# ============================================================================
# YOUR ALGORITHM GOES HERE. Everything above and below is plumbing.
# ============================================================================

# --- One-time precomputation (done at import, not on every MQTT message) ---

EXIT_SET = {(r, c) for r, c, _ in EXIT_CELLS}
EXIT_FACING = {(r, c): facing for r, c, facing in EXIT_CELLS}
DELTA_TO_YAW = {(dr, dc): yaw for yaw, (dr, dc, _) in HEADING_DELTA.items()}
# 0=EAST, 90=NORTH, 180=WEST, 270=SOUTH (matches HEADING_DELTA's convention)
FACING_TO_YAW = {'east': 0.0, 'north': 90.0, 'west': 180.0, 'south': 270.0}
RELATIVE_CMD = {0: "FRONT", 90: "LEFT", 180: "BACK", 270: "RIGHT"}


def _build_graph():
    """Precompute open neighbors for every cell once, instead of re-deriving
    them from wall bits on every single planning call."""
    graph = {}
    for r in range(MAZE_ROWS):
        for c in range(MAZE_COLS):
            neighbors = []
            for dr, dc, wall_bit in HEADING_DELTA.values():
                if WALLS[r][c] & wall_bit:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < MAZE_ROWS and 0 <= nc < MAZE_COLS:
                    neighbors.append((nr, nc))
            graph[(r, c)] = neighbors
    return graph


GRAPH = _build_graph()

_last_cell = None


def _bfs_first_step(start, targets):
    """Shortest-path BFS from `start` to the nearest cell in `targets`.
    Returns the first cell to move to, or None if start has no reachable
    target other than itself."""
    if not targets:
        return None

    visited = {start}
    # queue holds (cell, first_step_taken_to_get_there)
    queue = deque((n, n) for n in GRAPH[start])
    visited.update(n for n, _ in queue)

    for cell, first_step in queue:
        if cell in targets:
            return first_step

    while queue:
        cell, first_step = queue.popleft()
        for neighbor in GRAPH[cell]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            if neighbor in targets:
                return first_step
            queue.append((neighbor, first_step))

    return None


def _emergency_step(cell, avoid):
    """No BFS path found (shouldn't normally happen) - just pick any open
    neighbor, preferring one that isn't where we just came from."""
    neighbors = GRAPH[cell]
    if not neighbors:
        return None
    preferred = [n for n in neighbors if n != avoid]
    return preferred[0] if preferred else neighbors[0]


def _command_for_move(cell, next_cell, current_yaw):
    dr = next_cell[0] - cell[0]
    dc = next_cell[1] - cell[1]
    target_yaw = DELTA_TO_YAW.get((dr, dc))
    if target_yaw is None:
        return None

    diff = int((target_yaw - current_yaw) % 360)
    return RELATIVE_CMD.get(diff)


def choose_command(pacbot_cell, pacbot_yaw, pellets_remaining):
    """FRONT/LEFT/RIGHT/BACK to send now, or None. pacbot_cell=(row,col),
    pacbot_yaw one of HEADING_DELTA's keys, pellets_remaining=set of
    (row,col)."""
    global _last_cell

    # Normalize yaw to nearest 90 degrees to fix floating-point drift issues
    pacbot_yaw = round(pacbot_yaw / 90.0) * 90.0 % 360

    if pellets_remaining:
        targets = set(pellets_remaining)
    else:
        # No pellets left: head for an exit. If we're ALREADY standing on
        # one, don't pathfind to the other one - drive straight out through
        # this cell's known exit facing instead. That direction leads off
        # the grid, so it can never appear in GRAPH/BFS; it has to be
        # handled as a special case, separate from normal cell-to-cell travel.
        if pacbot_cell in EXIT_FACING:
            desired_yaw = FACING_TO_YAW[EXIT_FACING[pacbot_cell]]
            diff = int((desired_yaw - pacbot_yaw) % 360)
            _last_cell = pacbot_cell
            return RELATIVE_CMD.get(diff)
        targets = EXIT_SET

    next_cell = _bfs_first_step(pacbot_cell, targets)

    if next_cell is None:
        next_cell = _emergency_step(pacbot_cell, _last_cell)

    if next_cell is None:
        return None

    _last_cell = pacbot_cell
    return _command_for_move(pacbot_cell, next_cell, pacbot_yaw)
# ============================================================================


def parse_pellets(payload):
    return {tuple(cell) for cell in json.loads(payload)}


def main():
    state = {
        "running": False,
        "pellets": set(),
        "cell": (MAZE_ROWS // 2, MAZE_COLS // 2),
        "yaw": 0.0,
        "in_flight": False,
        "got_pose": False,
        "got_pellets": False,
    }

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="Controller")

    def decide_and_send():
        if (not state["running"] or state["in_flight"]
                or not state["got_pose"] or not state["got_pellets"]):
            return
        print(f"[debug] pose={state['cell']} yaw={state['yaw']} pellets={state['pellets']}")
        cmd = choose_command(state["cell"], state["yaw"], set(state["pellets"]))
        if cmd is not None:
            state["in_flight"] = True
            client.publish(CMD_VEL_TOPIC, cmd)
            print(f"[controller] {state['cell']} yaw={state['yaw']} -> {cmd}, "
                  f"pellets_left={len(state['pellets'])}")

    def on_message(client, userdata, msg):
        try:
            if msg.topic == BOT_CMD_TOPIC:
                running = msg.payload.decode().startswith("1")
                was_running = state["running"]
                state["running"] = running
                if running and not was_running:
                    decide_and_send()   # kick off the reactive loop on Start
            elif msg.topic == PELLETS_TOPIC:
                state["pellets"] = parse_pellets(msg.payload.decode())
                state["got_pellets"] = True
                decide_and_send()
            elif msg.topic == POSE_TOPIC:
                data = json.loads(msg.payload.decode())
                state["cell"] = (int(data["col"]), int(data["row"]))
                state["got_pose"] = True
                state["yaw"] = float(data.get("yaw", 0.0))
                state["in_flight"] = False
                decide_and_send()
        except Exception as e:
            print("[controller] mqtt parse error:", e)

    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe([(BOT_CMD_TOPIC, 0), (PELLETS_TOPIC, 0), (POSE_TOPIC, 0)])
    client.loop_start()

    print(f"[controller] ready; sending one '{CMD_VEL_TOPIC}' command at a time, "
          f"reacting to '{POSE_TOPIC}'/'{PELLETS_TOPIC}' feedback")

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
