import json
import sys
import math
import time
import heapq
from itertools import permutations

# ============================================================
# INPUT
# ============================================================

input_data = json.loads(sys.stdin.read())

map_size = input_data["map_size"]
warehouse = (map_size[0] / 2, map_size[1] / 2)

drones = input_data["drones"]
deliveries = input_data["deliveries"]

no_fly_zones = input_data.get("no_fly_zones", [])
charging_stations = input_data.get("charging_stations", [])

# ============================================================
# CONSTANTS
# ============================================================

BATTERY = 500.0
CHARGE_RATE = 2.0

MAX_TRIP_SIZE = 6
MAX_CAND_PER_TRIP = 50

TIME_BUDGET = 8.73

START_TIME = time.time()

_hypot = math.hypot

# scoring weights
W_DELIVERY = 100.0
W_ENERGY = 0.1
W_TIME = 0.05

# ============================================================
# UTILITIES
# ============================================================

def time_left():
    return TIME_BUDGET - (time.time() - START_TIME)

def dist(a, b):
    return _hypot(a[0] - b[0], a[1] - b[1])

def make_step(
        x,
        y,
        t,
        action,
        delivery_id=None,
        delivery_ids=None):

    s = {
        "x": float(x),
        "y": float(y),
        "t": float(t),
        "action": action
    }

    if delivery_id is not None:
        s["delivery_id"] = delivery_id

    if delivery_ids is not None:
        s["delivery_ids"] = delivery_ids

    return s

def path_phys_dist(start_pos, steps):

    total = 0.0
    cur = start_pos

    for s in steps:

        nxt = (s["x"], s["y"])

        if s["action"] != "WAIT":
            total += dist(cur, nxt)

        cur = nxt

    return total

# ============================================================
# NFZ GEOMETRY
# ============================================================

def line_intersects_rect(a, b, rect):

    x1, y1 = a
    x2, y2 = b

    c = rect["corners"]

    xmin = min(c[0][0], c[1][0])
    xmax = max(c[0][0], c[1][0])

    ymin = min(c[0][1], c[1][1])
    ymax = max(c[0][1], c[1][1])

    dx = x2 - x1
    dy = y2 - y1

    p = [-dx, dx, -dy, dy]

    q = [
        x1 - xmin,
        xmax - x1,
        y1 - ymin,
        ymax - y1
    ]

    u1 = 0.0
    u2 = 1.0

    for pi, qi in zip(p, q):

        if abs(pi) < 1e-9:

            if qi < 0:
                return False

        else:

            t = qi / pi

            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)

    return u1 < u2

def line_intersects_circle(a, b, circle):

    cx, cy = circle["center"]
    r = circle["radius"]

    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:

        return _hypot(ax - cx, ay - cy) <= r

    t = (
        ((cx - ax) * dx + (cy - ay) * dy)
        / (dx * dx + dy * dy)
    )

    t = max(0.0, min(1.0, t))

    px = ax + t * dx
    py = ay + t * dy

    return _hypot(px - cx, py - cy) <= r

def zone_geo_hit(a, b, z):

    if z["shape"] == "circle":
        return line_intersects_circle(a, b, z)

    return line_intersects_rect(a, b, z)

def segment_blocked(a, b, z, start_t):

    if not zone_geo_hit(a, b, z):
        return False

    seg_len = dist(a, b)

    z_s = z.get("T_start", -1e18)
    z_e = z.get("T_end", 1e18)

    return not (
        start_t + seg_len < z_s
        or
        start_t > z_e
    )

def blocked(a, b, zones, start_t=0.0):

    if dist(a, b) < 1e-9:
        return False

    return any(
        segment_blocked(a, b, z, start_t)
        for z in zones
    )

def earliest_clear_departure(
        a,
        b,
        zones,
        start_t):

    wait_t = start_t

    for _ in range(60):

        any_block = False

        for z in zones:

            if segment_blocked(
                    a,
                    b,
                    z,
                    wait_t):

                wait_t = max(
                    wait_t,
                    z.get("T_end", wait_t + 1.0)
                )

                any_block = True

        if not any_block:
            return wait_t

    return None

# ============================================================
# DETOURS
# ============================================================

def detour_candidates(a, b, z):

    if z["shape"] == "circle":

        cx, cy = z["center"]

        r = z["radius"] + 40

        dx = b[0] - a[0]
        dy = b[1] - a[1]

        length = _hypot(dx, dy)

        if length < 1e-9:

            return [
                (cx + r, cy),
                (cx - r, cy)
            ]

        px = -dy / length
        py = dx / length

        return [

            (cx + px * r, cy + py * r),
            (cx - px * r, cy - py * r),

            (cx + px * 1.5 * r, cy + py * 1.5 * r),
            (cx - px * 1.5 * r, cy - py * 1.5 * r),

            (cx + px * 2.0 * r, cy + py * 2.0 * r),
            (cx - px * 2.0 * r, cy - py * 2.0 * r),
        ]

    c = z["corners"]

    xmin = min(c[0][0], c[1][0])
    xmax = max(c[0][0], c[1][0])

    ymin = min(c[0][1], c[1][1])
    ymax = max(c[0][1], c[1][1])

    m = 25

    mx = (xmin + xmax) / 2
    my = (ymin + ymax) / 2

    return [

        (xmax + m, my),
        (xmin - m, my),

        (mx, ymax + m),
        (mx, ymin - m),

        (xmax + m, ymax + m),
        (xmin - m, ymin - m),

        (xmax + m, ymin - m),
        (xmin - m, ymax + m),
    ]

# ============================================================
# PATH PLANNER
# ============================================================

def plan_segment(a, b, start_t, zones):

    if not blocked(a, b, zones, start_t):

        return [
            make_step(
                b[0],
                b[1],
                start_t + dist(a, b),
                "WAYPOINT"
            )
        ]

    best_path = None
    best_arrive = float("inf")

    clear_t = earliest_clear_departure(
        a,
        b,
        zones,
        start_t
    )

    if clear_t is not None:

        arrive = clear_t + dist(a, b)

        if arrive < best_arrive:

            segs = []

            if clear_t > start_t + 1e-9:

                segs.append(
                    make_step(
                        a[0],
                        a[1],
                        clear_t,
                        "WAIT"
                    )
                )

            segs.append(
                make_step(
                    b[0],
                    b[1],
                    arrive,
                    "WAYPOINT"
                )
            )

            best_path = segs
            best_arrive = arrive

    for z in zones:

        if not segment_blocked(a, b, z, start_t):
            continue

        for mid in detour_candidates(a, b, z):

            d1 = dist(a, mid)

            if blocked(a, mid, zones, start_t):
                continue

            t_mid = start_t + d1

            d2 = dist(mid, b)

            if blocked(mid, b, zones, t_mid):
                continue

            arrive = t_mid + d2

            if arrive < best_arrive:

                best_arrive = arrive

                best_path = [

                    make_step(
                        mid[0],
                        mid[1],
                        t_mid,
                        "WAYPOINT"
                    ),

                    make_step(
                        b[0],
                        b[1],
                        arrive,
                        "WAYPOINT"
                    )
                ]

    return best_path

def preprocess_deliveries(raw, wh):

    out = []

    for d in raw:

        dd = dict(d)

        dd["_point"] = (d["x"], d["y"])

        dd["_warehouse_dist"] = dist(
            wh,
            dd["_point"]
        )

        out.append(dd)

    return out

def route_combined_cost(
        wh,
        items,
        start_t):

    cur = wh
    t = start_t

    payload = sum(
        d["weight"]
        for d in items
    )

    energy = 0.0
    penalty = 0.0

    for d in items:

        d_dist = dist(cur, d["_point"])

        t += d_dist

        energy += d_dist * (
            1.0 + payload
        )

        payload -= d["weight"]

        if t > d["deadline"]:

            penalty += (
                t - d["deadline"]
            ) * 1e9

        cur = d["_point"]

    energy += dist(cur, wh)

    return (
        penalty
        + energy * W_ENERGY
        + t * W_TIME
    )

def best_order(
        wh,
        items,
        start_t=0.0):

    n = len(items)

    if n <= 1:
        return list(items)

    if n <= 8:

        best_perm = None
        best_cost = float("inf")

        for perm in permutations(items):

            c = route_combined_cost(
                wh,
                perm,
                start_t
            )

            if c < best_cost:

                best_cost = c
                best_perm = list(perm)

        return best_perm

    return list(items)

def simulate_route(
        wh,
        drone,
        order,
        start_t,
        zones):

    if not order:
        return None

    total_payload = sum(
        d["weight"]
        for d in order
    )

    if total_payload > drone["max_payload"]:
        return None

    cur = wh
    cur_t = start_t

    payload = total_payload

    path = []

    battery = BATTERY

    total_nrg = 0.0

    for d in order:

        target = d["_point"]

        plan = plan_segment(
            cur,
            target,
            cur_t,
            zones
        )

        if plan is None:
            return None

        leg_dist = path_phys_dist(
            cur,
            plan
        )

        leg_e = leg_dist * (
            1.0 + payload
        )

        if leg_e > battery:
            return None

        arrive_t = plan[-1]["t"]

        if arrive_t > d["deadline"]:
            return None

        battery -= leg_e

        total_nrg += leg_e

        plan[-1]["action"] = "DELIVER"

        plan[-1]["delivery_id"] = d["id"]

        path.extend(plan)

        payload -= d["weight"]

        cur = target
        cur_t = arrive_t

    back = plan_segment(
        cur,
        wh,
        cur_t,
        zones
    )

    if back is None:
        return None

    back_dist = path_phys_dist(
        cur,
        back
    )

    back_e = back_dist

    if back_e > battery:
        return None

    battery -= back_e

    total_nrg += back_e

    back[-1]["action"] = "RETURN"

    path.extend(back)

    return {
        "path": path,
        "finish_t": back[-1]["t"],
        "energy": total_nrg,
        "deliveries": [
            d["id"] for d in order
        ]
    }

def trip_score(trip):

    n = len(trip["deliveries"])

    return (
        n * W_DELIVERY
        - trip["energy"] * W_ENERGY
        - trip["finish_t"] * W_TIME
    )

def filter_candidates(candidates):

    if len(candidates) <= MAX_CAND_PER_TRIP:
        return candidates

    urgent = heapq.nsmallest(
        MAX_CAND_PER_TRIP // 2,
        candidates,
        key=lambda d: d["deadline"]
    )

    nearby = heapq.nsmallest(
        MAX_CAND_PER_TRIP // 2,
        candidates,
        key=lambda d: d["_warehouse_dist"]
    )

    seen = set()
    out = []

    for d in urgent + nearby:

        if d["id"] in seen:
            continue

        seen.add(d["id"])
        out.append(d)

    return out

def try_build_trip(
        wh,
        drone,
        candidates,
        start_t,
        zones):

    candidates = filter_candidates(
        candidates
    )

    best_trip = None
    best_score = float("-inf")

    orderings = [

        sorted(
            candidates,
            key=lambda d: (
                d["deadline"],
                d["_warehouse_dist"]
            )
        ),

        sorted(
            candidates,
            key=lambda d: (
                -d["weight"],
                d["deadline"]
            )
        ),
    ]

    for cand_order in orderings:

        chosen = []
        weight = 0.0

        for cand in cand_order:

            if time_left() < 0.1:
                break

            if (
                weight + cand["weight"]
                > drone["max_payload"]
            ):
                continue

            if len(chosen) >= MAX_TRIP_SIZE:
                break

            trial = chosen + [cand]

            order = best_order(
                wh,
                trial,
                start_t
            )

            sim = simulate_route(
                wh,
                drone,
                order,
                start_t,
                zones
            )

            if sim is None:
                continue

            sc = trip_score(sim)

            if sc > best_score:

                best_score = sc
                best_trip = sim

            chosen = trial
            weight += cand["weight"]

    return best_trip

def solve(
        wh,
        drones,
        deliveries,
        zones):

    deliveries = preprocess_deliveries(
        deliveries,
        wh
    )

    states = {
        d["id"]: {
            "available": 0.0,
            "path": []
        }
        for d in drones
    }

    pending = {
        d["id"]: d
        for d in deliveries
    }

    while pending and time_left() > 0.2:

        any_progress = False

        for drone in sorted(
                drones,
                key=lambda dr:
                states[dr["id"]]["available"]):

            if time_left() < 0.1:
                break

            did = drone["id"]

            start_t = states[did]["available"]

            feasible = [

                d for d in pending.values()

                if (
                    d["weight"]
                    <= drone["max_payload"]
                )

                and (
                    start_t
                    + d["_warehouse_dist"]
                    <= d["deadline"]
                )
            ]

            if not feasible:
                continue

            trip = try_build_trip(
                wh,
                drone,
                feasible,
                start_t,
                zones
            )

            if trip is None:
                continue

            pickup = make_step(
                wh[0],
                wh[1],
                start_t,
                "PICKUP",
                delivery_ids=trip["deliveries"]
            )

            states[did]["path"].append(
                pickup
            )

            states[did]["path"].extend(
                trip["path"]
            )

            states[did]["available"] = (
                trip["finish_t"]
            )

            for did2 in trip["deliveries"]:
                pending.pop(did2, None)

            any_progress = True

        if not any_progress:
            break

    return [

        {
            "drone_id": drone["id"],
            "path": states[drone["id"]]["path"]
        }

        for drone in drones

        if states[drone["id"]]["path"]
    ]

result = solve(
    warehouse,
    drones,
    deliveries,
    no_fly_zones
)

print(json.dumps({
    "flight_manifest": result
}))
