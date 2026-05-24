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

MAX_TRIP_SIZE = 8
MAX_CAND_PER_TRIP = 60

TIME_BUDGET = 8.73

START_TIME = time.time()

_hypot = math.hypot

# scoring weights
W_DELIVERY = 100.0
W_ENERGY = 0.1
W_TIME = 0.05

# safety margins
BATTERY_SAFETY = 1.02      # need 2% slack on every leg
DEADLINE_SAFETY = 1e-6

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

def _cs_point(cs):
    # Tolerate either dict or list/tuple form
    if isinstance(cs, dict):
        return (cs.get("x", cs.get(0)), cs.get("y", cs.get(1)))
    return (cs[0], cs[1])

# Pre-compute charging-station points once
_CS_POINTS = [_cs_point(c) for c in charging_stations]

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

            (cx + px * 1.3 * r, cy + py * 1.3 * r),
            (cx - px * 1.3 * r, cy - py * 1.3 * r),

            (cx + px * 1.7 * r, cy + py * 1.7 * r),
            (cx - px * 1.7 * r, cy - py * 1.7 * r),

            (cx + px * 2.2 * r, cy + py * 2.2 * r),
            (cx - px * 2.2 * r, cy - py * 2.2 * r),
        ]

    c = z["corners"]

    xmin = min(c[0][0], c[1][0])
    xmax = max(c[0][0], c[1][0])

    ymin = min(c[0][1], c[1][1])
    ymax = max(c[0][1], c[1][1])

    m = 25
    m2 = 60

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

        (xmax + m2, my),
        (xmin - m2, my),
        (mx, ymax + m2),
        (mx, ymin - m2),
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

# ============================================================
# CHARGING STATION HELPERS  (NEW)
# ============================================================

def find_charging_detour(
        a,
        b,
        battery,
        payload,
        start_t,
        zones,
        cs_points):
    """
    Pick the charging-station that lets us go a -> CS -> b
    with the smallest extra distance, while:
      - leg a -> CS is feasible with current battery
      - leg CS -> b is feasible with full battery
      - both legs respect NFZs
    Returns (cs_point, plan_a_cs, plan_cs_b, arrive_t) or None.
    """
    if not cs_points:
        return None

    best = None
    best_extra = float("inf")
    direct = dist(a, b)
    factor = 1.0 + payload

    # Cheap geometric pre-filter; expensive plan_segment only on top few.
    geo_ranked = []
    for cs_pt in cs_points:
        d1 = dist(a, cs_pt)
        d2 = dist(cs_pt, b)
        # quick battery feasibility checks (Euclidean lower bound)
        if d1 * factor * BATTERY_SAFETY > battery:
            continue
        if d2 * factor * BATTERY_SAFETY > BATTERY:
            continue
        geo_ranked.append((d1 + d2 - direct, d1, d2, cs_pt))

    if not geo_ranked:
        return None

    geo_ranked.sort(key=lambda x: x[0])

    # Plan only the most promising candidates (NFZ-aware)
    for extra, _, _, cs_pt in geo_ranked[:6]:
        plan_a_cs = plan_segment(a, cs_pt, start_t, zones)
        if plan_a_cs is None:
            continue

        leg1_dist = path_phys_dist(a, plan_a_cs)
        leg1_e = leg1_dist * factor
        if leg1_e * BATTERY_SAFETY > battery:
            continue

        arrive_cs_t = plan_a_cs[-1]["t"]
        # Time to recharge to full from where we'll be after leg1.
        residual = battery - leg1_e
        charge_time = (BATTERY - residual) / CHARGE_RATE
        depart_t = arrive_cs_t + charge_time

        plan_cs_b = plan_segment(cs_pt, b, depart_t, zones)
        if plan_cs_b is None:
            continue

        leg2_dist = path_phys_dist(cs_pt, plan_cs_b)
        leg2_e = leg2_dist * factor
        if leg2_e * BATTERY_SAFETY > BATTERY:
            continue

        if extra < best_extra:
            best_extra = extra
            best = (cs_pt, plan_a_cs, plan_cs_b,
                    arrive_cs_t, depart_t, leg1_e, leg2_e)

    return best

# ============================================================
# DELIVERY PREP / COSTING
# ============================================================

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

# ============================================================
# ORDERING (overhauled: seeds + 2-opt + Or-opt)
# ============================================================

def _two_opt(perm, wh, start_t, deadline_check):
    n = len(perm)
    if n < 4:
        return perm

    best = list(perm)
    best_cost = route_combined_cost(wh, best, start_t)

    improved = True
    while improved:
        if not deadline_check():
            break
        improved = False
        for i in range(n - 1):
            if not deadline_check():
                break
            bi = best[i]
            for j in range(i + 1, n):
                # reverse segment best[i:j+1]
                new_perm = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cost = route_combined_cost(wh, new_perm, start_t)
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best = new_perm
                    improved = True
                    bi = best[i]
    return best

def _or_opt(perm, wh, start_t, deadline_check):
    n = len(perm)
    if n < 3:
        return perm

    best = list(perm)
    best_cost = route_combined_cost(wh, best, start_t)

    improved = True
    while improved:
        if not deadline_check():
            break
        improved = False
        # Try moving a single element from i to position j
        for seg_len in (1, 2):
            if seg_len >= n:
                break
            for i in range(n - seg_len + 1):
                if not deadline_check():
                    break
                seg = best[i:i + seg_len]
                rest = best[:i] + best[i + seg_len:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    new_perm = rest[:j] + seg + rest[j:]
                    if new_perm == best:
                        continue
                    cost = route_combined_cost(wh, new_perm, start_t)
                    if cost < best_cost - 1e-9:
                        best_cost = cost
                        best = new_perm
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best

def best_order(
        wh,
        items,
        start_t=0.0):

    n = len(items)

    if n <= 1:
        return list(items)

    # Exhaustive when small enough.
    if n <= 7:

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

    # Heuristic seeds + local search for larger trips.
    seeds = []

    # 1. Nearest-neighbour from warehouse
    remaining = list(items)
    cur = wh
    nn = []
    while remaining:
        nxt = min(
            remaining,
            key=lambda d: dist(cur, d["_point"])
        )
        nn.append(nxt)
        remaining.remove(nxt)
        cur = nxt["_point"]
    seeds.append(nn)

    # 2. Deadline order (urgent first)
    seeds.append(
        sorted(items, key=lambda d: d["deadline"])
    )

    # 3. Sweep (polar angle around warehouse)
    seeds.append(
        sorted(
            items,
            key=lambda d: math.atan2(
                d["_point"][1] - wh[1],
                d["_point"][0] - wh[0]
            )
        )
    )

    # 4. Distance from warehouse ascending (drop heavy near first)
    seeds.append(
        sorted(items, key=lambda d: d["_warehouse_dist"])
    )

    # Pick best seed
    best = seeds[0]
    best_cost = route_combined_cost(wh, best, start_t)
    for s in seeds[1:]:
        c = route_combined_cost(wh, s, start_t)
        if c < best_cost:
            best_cost = c
            best = s

    deadline_check = lambda: time_left() > 0.12

    # Refine: 2-opt -> Or-opt -> 2-opt
    best = _two_opt(best, wh, start_t, deadline_check)
    best = _or_opt(best, wh, start_t, deadline_check)
    best = _two_opt(best, wh, start_t, deadline_check)

    return best

# ============================================================
# SIMULATE A TRIP (now charging-station aware)
# ============================================================

def simulate_route(
        wh,
        drone,
        order,
        start_t,
        zones,
        cs_points,
        battery_start=BATTERY):

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

    battery = battery_start
    total_nrg = 0.0
    charges = 0

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

        leg_dist = path_phys_dist(cur, plan)
        leg_e = leg_dist * (1.0 + payload)

        if leg_e * BATTERY_SAFETY > battery:

            # Try to detour through a charging station
            det = find_charging_detour(
                cur,
                target,
                battery,
                payload,
                cur_t,
                zones,
                cs_points
            )

            if det is None:
                return None

            (cs_pt, plan_a_cs, plan_cs_b,
             arrive_cs_t, depart_t,
             leg1_e, leg2_e) = det

            arrive_t = plan_cs_b[-1]["t"]
            if arrive_t > d["deadline"] + DEADLINE_SAFETY:
                return None

            battery -= leg1_e
            total_nrg += leg1_e

            # CHARGE step at the station
            plan_a_cs[-1]["action"] = "CHARGE"
            path.extend(plan_a_cs)
            charges += 1

            # After charging, battery is full
            battery = BATTERY - leg2_e
            total_nrg += leg2_e

            plan_cs_b[-1]["action"] = "DELIVER"
            plan_cs_b[-1]["delivery_id"] = d["id"]
            path.extend(plan_cs_b)

            payload -= d["weight"]
            cur = target
            cur_t = arrive_t
            continue

        arrive_t = plan[-1]["t"]
        if arrive_t > d["deadline"] + DEADLINE_SAFETY:
            return None

        battery -= leg_e
        total_nrg += leg_e

        plan[-1]["action"] = "DELIVER"
        plan[-1]["delivery_id"] = d["id"]
        path.extend(plan)

        payload -= d["weight"]
        cur = target
        cur_t = arrive_t

    # Return-to-warehouse leg (also charging-aware)
    back = plan_segment(cur, wh, cur_t, zones)
    if back is None:
        return None

    back_dist = path_phys_dist(cur, back)
    back_e = back_dist  # payload is 0 by here

    if back_e * BATTERY_SAFETY > battery:

        det = find_charging_detour(
            cur,
            wh,
            battery,
            0.0,
            cur_t,
            zones,
            cs_points
        )

        if det is None:
            return None

        (cs_pt, plan_a_cs, plan_cs_b,
         arrive_cs_t, depart_t,
         leg1_e, leg2_e) = det

        battery -= leg1_e
        total_nrg += leg1_e

        plan_a_cs[-1]["action"] = "CHARGE"
        path.extend(plan_a_cs)
        charges += 1

        battery = BATTERY - leg2_e
        total_nrg += leg2_e

        plan_cs_b[-1]["action"] = "RETURN"
        path.extend(plan_cs_b)
        finish_t = plan_cs_b[-1]["t"]
    else:
        battery -= back_e
        total_nrg += back_e
        back[-1]["action"] = "RETURN"
        path.extend(back)
        finish_t = back[-1]["t"]

    return {
        "path": path,
        "finish_t": finish_t,
        "energy": total_nrg,
        "deliveries": [d["id"] for d in order],
        "battery_left": battery,
        "charges": charges
    }

# ============================================================
# TRIP SCORING + CANDIDATE FILTERING
# ============================================================

def trip_score(trip):

    n = len(trip["deliveries"])

    # Light penalty per mid-trip charge so we prefer non-detoured
    # trips when delivery count is equal.
    return (
        n * W_DELIVERY
        - trip["energy"] * W_ENERGY
        - trip["finish_t"] * W_TIME
        - trip.get("charges", 0) * 0.5
    )

def filter_candidates(candidates, wh, start_t):
    """
    Cluster-aware shortlist:
      - keep most urgent items
      - keep nearest-warehouse items
      - keep items geographically close to the urgent seed
    Deduplicated, capped at MAX_CAND_PER_TRIP.
    """
    if len(candidates) <= MAX_CAND_PER_TRIP:
        return candidates

    k = MAX_CAND_PER_TRIP

    # The urgent seed = item with earliest deadline among the truly
    # reachable ones (already filtered upstream).
    seed = min(candidates, key=lambda d: d["deadline"])
    seed_pt = seed["_point"]

    quota_urgent = max(8, k // 3)
    quota_nearby_wh = max(8, k // 3)
    quota_nearby_seed = k - quota_urgent - quota_nearby_wh

    urgent = heapq.nsmallest(
        quota_urgent,
        candidates,
        key=lambda d: d["deadline"]
    )

    nearby_wh = heapq.nsmallest(
        quota_nearby_wh,
        candidates,
        key=lambda d: d["_warehouse_dist"]
    )

    nearby_seed = heapq.nsmallest(
        quota_nearby_seed,
        candidates,
        key=lambda d: dist(seed_pt, d["_point"])
    )

    seen = set()
    out = []
    for group in (urgent, nearby_seed, nearby_wh):
        for d in group:
            if d["id"] in seen:
                continue
            seen.add(d["id"])
            out.append(d)
            if len(out) >= k:
                return out
    return out

# ============================================================
# TRIP BUILDER  (more orderings + cheapest-insertion + refine)
# ============================================================

def _seed_orderings(candidates, wh):
    """Return a list of distinct candidate orderings to try."""
    orderings = []

    orderings.append(sorted(
        candidates,
        key=lambda d: (d["deadline"], d["_warehouse_dist"])
    ))

    orderings.append(sorted(
        candidates,
        key=lambda d: (-d["weight"], d["deadline"])
    ))

    orderings.append(sorted(
        candidates,
        key=lambda d: d["_warehouse_dist"]
    ))

    orderings.append(sorted(
        candidates,
        key=lambda d: math.atan2(
            d["_point"][1] - wh[1],
            d["_point"][0] - wh[0]
        )
    ))

    # density-ish: favour heavy-but-near items (high value per unit cost)
    orderings.append(sorted(
        candidates,
        key=lambda d: -(d["weight"] / (d["_warehouse_dist"] + 1.0))
    ))

    # urgency / distance ratio  (must-do soon AND cheap)
    orderings.append(sorted(
        candidates,
        key=lambda d: (
            (d["deadline"] - d["_warehouse_dist"]) +
            d["_warehouse_dist"] * 0.1
        )
    ))

    # dedupe by id-tuple
    seen = set()
    uniq = []
    for o in orderings:
        key = tuple(d["id"] for d in o)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(o)
    return uniq

def try_build_trip(
        wh,
        drone,
        candidates,
        start_t,
        zones,
        cs_points,
        battery_start):

    candidates = filter_candidates(candidates, wh, start_t)

    if not candidates:
        return None

    best_trip = None
    best_score = float("-inf")

    orderings = _seed_orderings(candidates, wh)

    for cand_order in orderings:

        if time_left() < 0.18:
            break

        chosen = []
        weight = 0.0
        consecutive_failures = 0

        for cand in cand_order:

            if time_left() < 0.12:
                break

            if (
                weight + cand["weight"]
                > drone["max_payload"]
            ):
                continue

            if len(chosen) >= MAX_TRIP_SIZE:
                break

            trial = chosen + [cand]

            order = best_order(wh, trial, start_t)

            sim = simulate_route(
                wh,
                drone,
                order,
                start_t,
                zones,
                cs_points,
                battery_start
            )

            if sim is None:
                consecutive_failures += 1
                # If we keep failing, give up early on this ordering
                if consecutive_failures >= 4 and len(chosen) >= 2:
                    break
                continue

            consecutive_failures = 0

            sc = trip_score(sim)

            if sc > best_score:
                best_score = sc
                best_trip = sim

            chosen = trial
            weight += cand["weight"]

    # Final polish: try removing each delivery from the best trip
    # in case dropping one yields a higher score (lower energy/time).
    if best_trip is not None and len(best_trip["deliveries"]) >= 3 \
            and time_left() > 0.4:

        ids = best_trip["deliveries"]
        id_to_item = {d["id"]: d for d in candidates}
        items_full = [id_to_item[i] for i in ids if i in id_to_item]

        if len(items_full) == len(ids):

            for skip_idx in range(len(items_full)):

                if time_left() < 0.2:
                    break

                trial = [
                    it for k, it in enumerate(items_full)
                    if k != skip_idx
                ]

                order = best_order(wh, trial, start_t)
                sim = simulate_route(
                    wh,
                    drone,
                    order,
                    start_t,
                    zones,
                    cs_points,
                    battery_start
                )
                if sim is None:
                    continue

                sc = trip_score(sim)
                if sc > best_score:
                    best_score = sc
                    best_trip = sim

    return best_trip

# ============================================================
# SOLVER  (now tracks per-drone battery + charge time)
# ============================================================

def solve(
        wh,
        drones,
        deliveries,
        zones,
        cs_points):

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

    # Sort drones once by capacity desc — heavy loaders try first when
    # they tie on availability, which lets them claim heavy clusters.
    drones_by_cap = sorted(
        drones,
        key=lambda dr: -dr["max_payload"]
    )

    while pending and time_left() > 0.25:

        any_progress = False

        # Round-robin by earliest-available drone (then capacity).
        ordered_drones = sorted(
            drones_by_cap,
            key=lambda dr: (
                states[dr["id"]]["available"],
                -dr["max_payload"]
            )
        )

        for drone in ordered_drones:

            if time_left() < 0.15:
                break

            did = drone["id"]
            st = states[did]
            start_t = st["available"]

            feasible = [
                d for d in pending.values()
                if d["weight"] <= drone["max_payload"]
                and start_t + d["_warehouse_dist"] <= d["deadline"]
            ]

            if not feasible:
                continue

            trip = try_build_trip(
                wh,
                drone,
                feasible,
                start_t,
                zones,
                cs_points,
                BATTERY
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

            st["path"].append(pickup)
            st["path"].extend(trip["path"])

            # Match original semantic: drone is ready immediately for
            # the next trip — battery is replenished at warehouse during
            # pickup/loading.  Mid-trip charging stations are still used
            # whenever a leg can't be made on the current battery.
            st["available"] = trip["finish_t"]

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
    _CS_POINTS
)

print(json.dumps({
    "flight_manifest": result
}))
