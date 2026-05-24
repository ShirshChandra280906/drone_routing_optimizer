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

# permutation enum cap for best_order; n!>this falls back to NN+2-opt
PERM_ENUM_CAP = 7

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

    for _ in range(80):

        any_block = False

        for z in zones:

            if segment_blocked(
                    a,
                    b,
                    z,
                    wait_t):

                # Bump comfortably past T_end to avoid any
                # boundary ambiguity in the checker.
                wait_t = max(
                    wait_t,
                    z.get("T_end", wait_t + 1.0)
                    + 1e-3
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

        r_safe = z["radius"] + 2.0  # tight detour
        r_med = z["radius"] + 12.0
        r_far = z["radius"] + 30.0

        dx = b[0] - a[0]
        dy = b[1] - a[1]

        length = _hypot(dx, dy)

        if length < 1e-9:

            return [
                (cx + r_far, cy),
                (cx - r_far, cy),
                (cx, cy + r_far),
                (cx, cy - r_far),
            ]

        # perpendicular unit vector to AB
        px = -dy / length
        py = dx / length

        # foot of perpendicular from circle center on AB
        # used to anchor detour midpoints on the line
        ab_dx = dx / length
        ab_dy = dy / length

        t_foot = (
            (cx - a[0]) * ab_dx
            + (cy - a[1]) * ab_dy
        )
        t_foot = max(0.0, min(length, t_foot))

        fx = a[0] + ab_dx * t_foot
        fy = a[1] + ab_dy * t_foot

        cands = []

        # offsets perpendicular from foot point: tight to far
        for r in (r_safe, r_med, r_far):
            cands.append((fx + px * r, fy + py * r))
            cands.append((fx - px * r, fy - py * r))

        # also offsets from circle center (for cases where
        # foot is outside the segment span)
        for r in (r_med, r_far):
            cands.append((cx + px * r, cy + py * r))
            cands.append((cx - px * r, cy - py * r))

        # tangent-style points: along AB direction past the
        # circle on the far side, useful for short approach
        cands.append(
            (cx + ab_dx * r_far, cy + ab_dy * r_far)
        )
        cands.append(
            (cx - ab_dx * r_far, cy - ab_dy * r_far)
        )

        return cands

    c = z["corners"]

    xmin = min(c[0][0], c[1][0])
    xmax = max(c[0][0], c[1][0])

    ymin = min(c[0][1], c[1][1])
    ymax = max(c[0][1], c[1][1])

    m_close = 4.0
    m_far = 25.0

    mx = (xmin + xmax) / 2
    my = (ymin + ymax) / 2

    return [

        # tight cardinal exits
        (xmax + m_close, my),
        (xmin - m_close, my),
        (mx, ymax + m_close),
        (mx, ymin - m_close),

        # wider cardinal exits
        (xmax + m_far, my),
        (xmin - m_far, my),
        (mx, ymax + m_far),
        (mx, ymin - m_far),

        # corner exits
        (xmax + m_close, ymax + m_close),
        (xmin - m_close, ymin - m_close),
        (xmax + m_close, ymin - m_close),
        (xmin - m_close, ymax + m_close),

        (xmax + m_far, ymax + m_far),
        (xmin - m_far, ymin - m_far),
        (xmax + m_far, ymin - m_far),
        (xmin - m_far, ymax + m_far),
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

def fly_with_charging(
        cur,
        target,
        cur_t,
        battery,
        payload,
        zones,
        charge_pts):
    """
    Plan flight cur -> target. If direct flight lacks battery,
    insert a single charge stop at the best charging station.

    Returns (steps, arrival_t, end_battery, energy_used)
    or None if no feasible plan exists.

    steps is a list of dict-step entries; the last step's coords
    are at target with action WAYPOINT (caller may rewrite to
    DELIVER / RETURN). For charge insertions, intermediate steps
    include WAYPOINTs to the station, a CHARGE step at arrival,
    a CHARGE_COMPLETE step at departure, then WAYPOINTs to target.
    """

    # 1. Try direct (existing detour-aware planner)
    direct = plan_segment(cur, target, cur_t, zones)

    if direct is not None:

        d_dist = path_phys_dist(cur, direct)
        d_e = d_dist * (1.0 + payload)

        if d_e <= battery + 1e-9:
            return (
                [dict(s) for s in direct],
                direct[-1]["t"],
                battery - d_e,
                d_e,
            )

    if not charge_pts:
        return None

    best = None
    best_finish = float("inf")

    for cs in charge_pts:

        # Hop 1: reach charging station
        if dist(cur, cs) < 1e-6:
            plan1 = []
            t_at_cs = cur_t
            bat_at_cs = battery
            e1 = 0.0
        else:
            plan1 = plan_segment(cur, cs, cur_t, zones)
            if plan1 is None:
                continue
            d1 = path_phys_dist(cur, plan1)
            e1 = d1 * (1.0 + payload)
            if e1 > battery + 1e-9:
                continue
            bat_at_cs = battery - e1
            t_at_cs = plan1[-1]["t"]

        # Hop 2: estimate cs -> target to size the charge
        plan2_est = plan_segment(cs, target, t_at_cs, zones)
        if plan2_est is None:
            continue

        d2_est = path_phys_dist(cs, plan2_est)
        e2_est = d2_est * (1.0 + payload)

        # full charge cap check
        if e2_est > BATTERY + 1e-9:
            continue

        deficit = max(0.0, e2_est - bat_at_cs)
        charge_time = deficit / CHARGE_RATE
        bat_after = min(
            BATTERY,
            bat_at_cs + charge_time * CHARGE_RATE
        )
        t_after = t_at_cs + charge_time

        # Replan hop 2 with possibly-different departure time
        # (NFZ window may shift the route or wait)
        plan2 = plan_segment(cs, target, t_after, zones)
        if plan2 is None:
            continue

        d2 = path_phys_dist(cs, plan2)
        e2 = d2 * (1.0 + payload)

        # If new route is heavier than estimate, charge longer.
        if e2 > bat_after + 1e-9:
            extra = e2 - bat_after
            extra_t = extra / CHARGE_RATE
            charge_time += extra_t
            bat_after = bat_at_cs + charge_time * CHARGE_RATE
            if bat_after > BATTERY + 1e-9:
                continue
            t_after = t_at_cs + charge_time
            plan2 = plan_segment(cs, target, t_after, zones)
            if plan2 is None:
                continue
            d2 = path_phys_dist(cs, plan2)
            e2 = d2 * (1.0 + payload)
            if e2 > bat_after + 1e-9:
                continue

        arrive_t = plan2[-1]["t"]

        # Build step list
        if plan1:
            steps = [dict(s) for s in plan1]
            steps[-1]["action"] = "CHARGE"
            steps[-1].pop("delivery_id", None)
            steps[-1].pop("delivery_ids", None)
        else:
            steps = [
                make_step(
                    cs[0],
                    cs[1],
                    t_at_cs,
                    "CHARGE"
                )
            ]

        steps.append(
            make_step(
                cs[0],
                cs[1],
                t_after,
                "CHARGE_COMPLETE"
            )
        )

        for s in plan2:
            steps.append(dict(s))

        if arrive_t < best_finish:
            best_finish = arrive_t
            best = (
                steps,
                arrive_t,
                bat_after - e2,
                e1 + e2,
            )

    return best

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

def _nn_order(wh, items):

    if not items:
        return []

    rem = list(items)
    cur = wh
    out = []

    while rem:

        nxt = min(
            rem,
            key=lambda d: dist(cur, d["_point"])
        )

        out.append(nxt)
        rem.remove(nxt)
        cur = nxt["_point"]

    return out

def _two_opt(wh, order, start_t, max_passes=8):

    n = len(order)

    if n < 4:
        return list(order)

    cur = list(order)
    cur_cost = route_combined_cost(wh, cur, start_t)

    for _ in range(max_passes):

        if time_left() < 0.05:
            break

        improved = False

        for i in range(n - 1):

            for j in range(i + 1, n):

                # reverse segment [i:j+1]
                new = (
                    cur[:i]
                    + cur[i:j + 1][::-1]
                    + cur[j + 1:]
                )

                c = route_combined_cost(
                    wh,
                    new,
                    start_t
                )

                if c < cur_cost - 1e-9:
                    cur = new
                    cur_cost = c
                    improved = True

        if not improved:
            break

    return cur

def best_order(
        wh,
        items,
        start_t=0.0):

    n = len(items)

    if n <= 1:
        return list(items)

    if n <= PERM_ENUM_CAP:

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

    # n > PERM_ENUM_CAP: seed several heuristics, refine via 2-opt.
    seeds = [
        _nn_order(wh, items),
        sorted(items, key=lambda d: d["deadline"]),
        sorted(
            items,
            key=lambda d: (
                d["deadline"],
                d["_warehouse_dist"]
            )
        ),
        sorted(items, key=lambda d: d["_warehouse_dist"]),
        sorted(items, key=lambda d: -d["weight"]),
    ]

    best = None
    best_cost = float("inf")

    seen_keys = set()

    for seed in seeds:

        key = tuple(d["id"] for d in seed)

        if key in seen_keys:
            continue

        seen_keys.add(key)

        refined = _two_opt(wh, seed, start_t)

        c = route_combined_cost(
            wh,
            refined,
            start_t
        )

        if c < best_cost:
            best_cost = c
            best = refined

    return best if best is not None else list(items)

def simulate_route(
        wh,
        drone,
        order,
        start_t,
        zones,
        charge_pts):

    if not order:
        return None

    total_payload = sum(
        d["weight"]
        for d in order
    )

    if total_payload > drone["max_payload"] + 1e-9:
        return None

    cur = wh
    cur_t = start_t

    payload = total_payload

    path = []

    battery = BATTERY

    total_nrg = 0.0

    for d in order:

        target = d["_point"]

        leg = fly_with_charging(
            cur,
            target,
            cur_t,
            battery,
            payload,
            zones,
            charge_pts,
        )

        if leg is None:
            return None

        steps, arrive_t, new_battery, leg_e = leg

        if arrive_t > d["deadline"] + 1e-9:
            return None

        # Final waypoint at target becomes the DELIVER step
        steps[-1]["action"] = "DELIVER"
        steps[-1]["delivery_id"] = d["id"]
        steps[-1].pop("delivery_ids", None)

        path.extend(steps)

        battery = new_battery
        total_nrg += leg_e

        payload -= d["weight"]
        if payload < 0.0:
            payload = 0.0

        cur = target
        cur_t = arrive_t

    # Return leg (with optional charging stop)
    back = fly_with_charging(
        cur,
        wh,
        cur_t,
        battery,
        0.0,
        zones,
        charge_pts,
    )

    if back is None:
        return None

    back_steps, back_arrival, _, back_e = back

    back_steps[-1]["action"] = "RETURN"
    back_steps[-1].pop("delivery_id", None)
    back_steps[-1].pop("delivery_ids", None)

    path.extend(back_steps)
    total_nrg += back_e

    return {
        "path": path,
        "finish_t": back_arrival,
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

    quota = max(1, MAX_CAND_PER_TRIP // 3)

    urgent = heapq.nsmallest(
        quota,
        candidates,
        key=lambda d: d["deadline"]
    )

    nearby = heapq.nsmallest(
        quota,
        candidates,
        key=lambda d: d["_warehouse_dist"]
    )

    # "slack" = deadline - dist; small slack = pressing.
    pressing = heapq.nsmallest(
        quota,
        candidates,
        key=lambda d: (
            d["deadline"]
            - d["_warehouse_dist"]
        )
    )

    seen = set()
    out = []

    for d in urgent + nearby + pressing:

        if d["id"] in seen:
            continue

        seen.add(d["id"])
        out.append(d)

        if len(out) >= MAX_CAND_PER_TRIP:
            break

    return out

def try_build_trip(
        wh,
        drone,
        candidates,
        start_t,
        zones,
        charge_pts):

    candidates = filter_candidates(
        candidates
    )

    if not candidates:
        return None

    best_trip = None
    best_score = float("-inf")

    orderings = [

        # Tightest deadline first, near as tiebreak
        sorted(
            candidates,
            key=lambda d: (
                d["deadline"],
                d["_warehouse_dist"]
            )
        ),

        # Heaviest first (so heavy items get a slot)
        sorted(
            candidates,
            key=lambda d: (
                -d["weight"],
                d["deadline"]
            )
        ),

        # Closest first (cheap, fast trips)
        sorted(
            candidates,
            key=lambda d: d["_warehouse_dist"]
        ),

        # Pressing: smallest slack (deadline-dist) first
        sorted(
            candidates,
            key=lambda d: (
                d["deadline"]
                - d["_warehouse_dist"]
            )
        ),
    ]

    for cand_order in orderings:

        if time_left() < 0.1:
            break

        chosen = []
        weight = 0.0
        cur_score = float("-inf")
        cur_trip = None

        for cand in cand_order:

            if time_left() < 0.05:
                break

            if (
                weight + cand["weight"]
                > drone["max_payload"] + 1e-9
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
                zones,
                charge_pts,
            )

            if sim is None:
                continue

            sc = trip_score(sim)

            # Only commit if extension actually improves;
            # otherwise this candidate is dead weight.
            if sc > cur_score + 1e-9:

                cur_score = sc
                cur_trip = sim
                chosen = trial
                weight += cand["weight"]

        if (
            cur_trip is not None
            and cur_score > best_score
        ):
            best_score = cur_score
            best_trip = cur_trip

    return best_trip

def solve(
        wh,
        drones,
        deliveries,
        zones,
        charge_pts):

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

        # Tie-break by larger payload first so the high-capacity
        # drone gets first crack at heavy / dense trips.
        for drone in sorted(
                drones,
                key=lambda dr: (
                    states[dr["id"]]["available"],
                    -dr["max_payload"],
                )):

            if time_left() < 0.1:
                break

            did = drone["id"]

            start_t = states[did]["available"]

            feasible = [

                d for d in pending.values()

                if (
                    d["weight"]
                    <= drone["max_payload"] + 1e-9
                )

                and (
                    start_t
                    + d["_warehouse_dist"]
                    <= d["deadline"] + 1e-9
                )
            ]

            if not feasible:
                continue

            trip = try_build_trip(
                wh,
                drone,
                feasible,
                start_t,
                zones,
                charge_pts,
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
    no_fly_zones,
    [(s["x"], s["y"]) for s in charging_stations],
)

print(json.dumps({
    "flight_manifest": result
}))
