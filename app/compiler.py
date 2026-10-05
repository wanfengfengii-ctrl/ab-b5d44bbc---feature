"""Integer delay-plan compiler.

Compiles per-element target delays into a small number of integer *ramps*
that the probe firmware can apply.

Definitions (all arithmetic is exact integer arithmetic)
--------------------------------------------------------
A sequence ``x[0..n-1]`` is feasible when:

* every ``x[i]`` lies in the closed global delay interval ``[lo, hi]``;
* every anchor ``x[index] == value`` holds exactly;
* ``|x[i+1] - x[i]| <= max_step``;
* the number of maximal constant runs (ramps) of the adjacent-difference
  sequence ``d[i] = x[i+1] - x[i]`` does not exceed ``max_ramps``.

Optional reset seam (``reset_after = s``): elements ``0..s`` and ``s+1..n-1``
form two independently sloped sub-arrays, each with at least two elements.

* the closed interval, the step limit and the anchors still apply inside each
  sub-array (anchor cones never cross the seam);
* the seam edge ``(s, s+1)`` is exempt from the step limit;
* ramps are tallied per sub-array: runs over edges ``0..s-1`` plus runs over
  edges ``s+1..n-2``; equal slopes across the seam are never merged, so the
  last left ramp ends at element ``s`` and the first right ramp starts at
  element ``s+1`` (the seam edge itself belongs to no ramp);
* ``max_ramps`` bounds the total number of ramps on both sides.

Optimization order, lexicographic (the first differing key decides):

1. minimize the maximum absolute error ``max_i |x[i] - target[i]|``;
2. then minimize the total absolute error ``sum_i |x[i] - target[i]|``;
3. then minimize the number of ramps actually used;
4. then minimize the delay sequence itself in lexicographic order.

Algorithms
----------
* Structural feasibility (global bounds, anchor cones under the step limit)
  is propagated as per-position integer bands; a violated anchor pair is
  reported as a localized conflict interval.
* The minimax error is found with exponential search followed by binary
  search; each feasibility test is a dynamic program over
  ``(position, value, last delta)`` states minimizing the ramp count.
  Transition cost is obtained in O(1) per predecessor value using the
  smallest / second-smallest predecessor ramp count, so each layer costs
  O(W * (2*max_step + 1)).
* With the optimal error budget fixed, a backward DP computes the best
  suffix cost ``(sum abs error, new ramps)`` for every state and the plan is
  recovered greedily, which yields the lexicographically smallest optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_ELEMENTS = 12
MAX_ELEMENTS = 48
MIN_ANCHORS = 2
MAX_ANCHORS = 8

# Safety valve for pathological integer domains (firmware delay values are
# bounded in practice).  A DP working window wider than this many integer
# values, or a layer transition more expensive than this many predecessor
# checks, is rejected as unsupported rather than stalling the service.
MAX_WINDOW = 8_192
MAX_LAYER_OPS = 250_000


class CompileError(ValueError):
    """Invalid request or infeasible compilation.

    ``status`` is the suggested HTTP status: 400 for malformed input,
    422 for well-formed but infeasible instances, 500 for internal errors.
    """

    def __init__(self, message: str, status: int = 400, details=None):
        super().__init__(message)
        self.status = status
        self.details = details or {}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_int(name, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompileError(f"'{name}' must be an integer", 400, {"field": name})
    return value


@dataclass(frozen=True)
class Conflict:
    """An infeasibility interval.

    ``start``/``end`` are element indices.  For a conflict between two
    consecutive anchors they are the two anchor indices; for an anchor that
    violates the global delay interval, ``start == end``.
    """

    kind: str
    start: int
    end: int
    detail: dict

    def to_dict(self) -> dict:
        out = {"kind": self.kind, "start": self.start, "end": self.end}
        out.update(self.detail)
        return out


def validate_request(payload) -> dict:
    """Validate and normalize a compile request payload."""
    if not isinstance(payload, dict):
        raise CompileError("request body must be a JSON object")

    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise CompileError("'targets' must be an array of integers", 400,
                           {"field": "targets"})
    n = len(targets)
    if not (MIN_ELEMENTS <= n <= MAX_ELEMENTS):
        raise CompileError(
            f"'targets' must contain between {MIN_ELEMENTS} and {MAX_ELEMENTS} "
            f"elements (got {n})", 400, {"field": "targets", "length": n})
    int_targets = [_as_int(f"targets[{i}]", v) for i, v in enumerate(targets)]

    if "delay_min" not in payload or "delay_max" not in payload:
        raise CompileError("'delay_min' and 'delay_max' are required", 400)
    lo = _as_int("delay_min", payload["delay_min"])
    hi = _as_int("delay_max", payload["delay_max"])
    if lo > hi:
        raise CompileError("'delay_min' must not exceed 'delay_max'", 400,
                           {"field": "delay_min"})

    max_step = _as_int("max_step", payload.get("max_step"))
    if max_step < 0:
        raise CompileError("'max_step' must be non-negative", 400,
                           {"field": "max_step"})

    max_ramps = _as_int("max_ramps", payload.get("max_ramps"))
    if not (1 <= max_ramps <= n - 1):
        raise CompileError(
            f"'max_ramps' must be between 1 and {n - 1} for {n} elements",
            400, {"field": "max_ramps"})

    # Optional reset seam: zero-based element after which the two independently
    # sloped sub-arrays begin; each side must contain at least two elements.
    reset_after = None
    if "reset_after" in payload and payload["reset_after"] is not None:
        reset_after = _as_int("reset_after", payload["reset_after"])
        if not (1 <= reset_after <= n - 3):
            raise CompileError(
                f"'reset_after' must leave at least two elements on each side "
                f"(an integer in [1, {n - 3}] for {n} elements, got "
                f"{reset_after})", 400, {"field": "reset_after"})

    anchors_raw = payload.get("anchors")
    if not isinstance(anchors_raw, list):
        raise CompileError("'anchors' must be an array", 400,
                           {"field": "anchors"})
    if not (MIN_ANCHORS <= len(anchors_raw) <= MAX_ANCHORS):
        raise CompileError(
            f"'anchors' must contain between {MIN_ANCHORS} and {MAX_ANCHORS} "
            f"entries (got {len(anchors_raw)})", 400, {"field": "anchors"})

    anchors = {}
    for k, a in enumerate(anchors_raw):
        if not isinstance(a, dict):
            raise CompileError(f"anchors[{k}] must be an object", 400,
                               {"field": f"anchors[{k}]"})
        if "index" not in a or "value" not in a:
            raise CompileError(
                f"anchors[{k}] requires 'index' and 'value'", 400,
                {"field": f"anchors[{k}]"})
        idx = _as_int(f"anchors[{k}].index", a["index"])
        val = _as_int(f"anchors[{k}].value", a["value"])
        if not (0 <= idx < n):
            raise CompileError(
                f"anchors[{k}].index={idx} out of range [0,{n - 1}]", 400,
                {"field": f"anchors[{k}].index"})
        if idx in anchors and anchors[idx] != val:
            raise CompileError(
                f"conflicting anchor values at index {idx}", 400,
                {"field": f"anchors[{k}].index", "index": idx})
        anchors[idx] = val

    return {
        "n": n,
        "targets": int_targets,
        "lo": lo,
        "hi": hi,
        "max_step": max_step,
        "max_ramps": max_ramps,
        "anchors": anchors,
        "reset_after": reset_after,
    }


# ---------------------------------------------------------------------------
# Structural feasibility: bands from global bounds + anchor cones
# ---------------------------------------------------------------------------


@dataclass
class StructuralResult:
    feasible: bool
    conflicts: list
    bands: list  # [(lo_i, hi_i), ...] feasible integer value per position


def structural_check(req: dict) -> StructuralResult:
    n = req["n"]
    lo, hi = req["lo"], req["hi"]
    step = req["max_step"]
    seam = req["reset_after"]
    anchors = req["anchors"]
    anchor_items = sorted(anchors.items())
    conflicts: list[Conflict] = []

    for idx, val in anchor_items:
        if val < lo or val > hi:
            conflicts.append(Conflict(
                "anchor_out_of_bounds", idx, idx,
                {"anchor_value": val, "delay_min": lo, "delay_max": hi}))
    if conflicts:
        return StructuralResult(False, conflicts, [(lo, hi)] * n)

    # NOTE: bands are deliberately *not* clipped to the extrema of targets and
    # anchor values: an optimal ramp fit can legitimately take values beyond
    # both (e.g. a shallow V between equal anchors whose targets sit at/above
    # the anchor value).  The error tube used by the DPs keeps every working
    # domain to width <= 2*E + 1 regardless.
    blo = [lo] * n
    bhi = [hi] * n

    # With a reset seam the two sub-arrays are structurally independent:
    # anchor cones never cross the seam and the seam edge (s, s+1) is exempt
    # from the step limit.  Without a seam this is a single [0, n) range.
    if seam is None:
        sides = [(0, n - 1, anchor_items)]
    else:
        left = [(i, v) for i, v in anchor_items if i <= seam]
        right = [(i, v) for i, v in anchor_items if i > seam]
        sides = [(0, seam, left), (seam + 1, n - 1, right)]

    unreachable_spans = []
    for side_lo_i, side_hi_i, items in sides:
        if not items:
            continue  # no anchors on this side: only global bounds + step

        first_idx, first_val = items[0]
        for i in range(side_lo_i, first_idx):
            dist = first_idx - i
            blo[i] = max(blo[i], first_val - step * dist)
            bhi[i] = min(bhi[i], first_val + step * dist)
        blo[first_idx] = bhi[first_idx] = first_val

        for (i0, v0), (i1, v1) in zip(items, items[1:]):
            gap = i1 - i0
            need = abs(v1 - v0)
            if need > step * gap:
                conflicts.append(Conflict(
                    "step_unreachable", i0, i1,
                    {"from_value": v0, "to_value": v1, "steps": gap,
                     "required_min_step": -(-need // gap),
                     "max_step": step, "min_total_change": need,
                     "max_total_change": step * gap}))
                unreachable_spans.append((i0, i1))
            for j in range(i0, i1 + 1):
                d1 = j - i0
                d2 = i1 - j
                blo[j] = max(blo[j], v0 - step * d1, v1 - step * d2)
                bhi[j] = min(bhi[j], v0 + step * d1, v1 + step * d2)

        last_idx, last_val = items[-1]
        for i in range(last_idx + 1, side_hi_i + 1):
            dist = i - last_idx
            blo[i] = max(blo[i], last_val - step * dist)
            bhi[i] = min(bhi[i], last_val + step * dist)

    bands = []
    for i in range(n):
        a, b = blo[i], bhi[i]
        if a > b and not any(s <= i <= e for s, e in unreachable_spans):
            conflicts.append(Conflict(
                "empty_band", i, i,
                {"delay_min": lo, "delay_max": hi, "max_step": step}))
        bands.append((a, b))

    return StructuralResult(not conflicts, conflicts, bands)


# ---------------------------------------------------------------------------
# Feasibility DP (minimize ramp count inside an absolute-error tube)
# ---------------------------------------------------------------------------


def _widths_under_budget(req, bands, budget):
    """Per-position integer ranges inside both the band and the error tube."""
    widths = []
    t = req["targets"]
    anchors = req["anchors"]
    for i, (a, b) in enumerate(bands):
        l = max(a, t[i] - budget)
        h = min(b, t[i] + budget)
        if i in anchors:
            v = anchors[i]
            if not (l <= v <= h):
                return None
            l = h = v
        if l > h:
            return None
        if h - l + 1 > MAX_WINDOW:
            raise CompileError(
                "integer delay domain too large to compile exactly "
                f"(more than {MAX_WINDOW} admissible values at element {i}); "
                "tighten 'delay_min'/'delay_max' or reduce the target spread",
                400, {"field": "delay_min", "element": i,
                      "window_width": h - l + 1})
        widths.append((l, h))
    return widths


def _best_two(p: dict):
    """Smallest value + its key, and the smallest value at another key."""
    m1 = None
    k1 = None
    m2 = None
    for k, v in p.items():
        if m1 is None or v < m1:
            m2 = m1
            m1, k1 = v, k
        elif m2 is None or v < m2:
            m2 = v
    return m1, k1, m2


def feasible_with_widths(req, widths) -> bool:
    """True iff some sequence fits ``widths`` while using <= max_ramps ramps.

    With a reset seam the step limit is skipped on the seam edge and the ramp
    run restarts there: the left ramp tally carries over, but the first right
    edge always starts a fresh ramp even when its slope matches the left one.
    """
    step = req["max_step"]
    cap = req["max_ramps"]
    seam = req["reset_after"]

    # prev[v] maps last-delta -> ramps used so far; None is the sentinel for
    # "no edge yet on this side" (position 0, or the first element after the
    # seam) so the first concrete edge of each side starts ramp number one.
    prev = {v: {None: 0} for v in range(widths[0][0], widths[0][1] + 1)}
    stats = {u: _best_two(p) for u, p in prev.items()}

    for i in range(1, req["n"]):
        lo_w, hi_w = widths[i]
        plo, phi = widths[i - 1]
        on_seam_edge = seam is not None and i == seam + 1
        if not on_seam_edge and (
                (hi_w - lo_w + 1) * min(2 * step + 1, phi - plo + 1)
                > MAX_LAYER_OPS):
            raise CompileError(
                "integer delay domain too large to compile exactly at "
                f"element {i}; tighten 'delay_min'/'delay_max' or reduce "
                "the target spread", 400, {"element": i})
        if on_seam_edge and (hi_w - lo_w + 1) * (phi - plo + 1) > MAX_LAYER_OPS:
            raise CompileError(
                "integer delay domain too large to compile exactly at the "
                f"reset seam (element {i}); tighten 'delay_min'/'delay_max' "
                "or reduce the target spread", 400, {"element": i})
        cur = {}
        if on_seam_edge:
            # The seam edge is unconstrained: any left value can jump to any
            # right value, it costs no ramp, and the delta history restarts.
            best_left = None
            for u in range(plo, phi + 1):
                for count in prev.get(u, {}).values():
                    if best_left is None or count < best_left:
                        best_left = count
            if best_left is None or best_left > cap:
                return False
            cur = {v: {None: best_left} for v in range(lo_w, hi_w + 1)}
        else:
            for v in range(lo_w, hi_w + 1):
                entry = {}
                ua = max(plo, v - step)
                ub = min(phi, v + step)
                for u in range(ua, ub + 1):
                    st = stats.get(u)
                    if st is None:
                        continue
                    m1, k1, m2 = st
                    d = v - u
                    cont = prev[u].get(d)              # keep the current ramp
                    brk = m1 if k1 != d else m2         # start a new ramp here
                    if brk is not None:
                        brk += 1
                    best = None
                    if cont is not None:
                        best = cont
                    if brk is not None and (best is None or brk < best):
                        best = brk
                    if best is not None and best <= cap:
                        old = entry.get(d)
                        if old is None or best < old:
                            entry[d] = best
                if entry:
                    cur[v] = entry
            if not cur:
                return False
        prev = cur
        stats = {u: _best_two(p) for u, p in prev.items()}
    return True


def ramp_feasible_on_bands(req, bands):
    """Decide ramp-budget feasibility with no error tube.

    Returns True/False when the exact band DP fits the configured domain
    guard, or ``None`` when the integer domain is too large to decide here.
    """
    widths = []
    for i, (a, b) in enumerate(bands):
        if b - a + 1 > MAX_WINDOW:
            return None
        widths.append((a, b))
    try:
        return feasible_with_widths(req, widths)
    except CompileError:
        return None


# ---------------------------------------------------------------------------
# Backward optimization DP: (sum abs error, new ramps) suffix costs
# ---------------------------------------------------------------------------


def _incoming_keys(widths, i, step, seam=None):
    """Map each value at position i to the possible deltas of edge (i-1).

    At the first element of a side (position 0, or the element right after the
    reset seam) the sole key is the None sentinel: the entering seam edge is
    exempt from the step limit and belongs to no ramp.
    """
    if i == 0 or (seam is not None and i == seam + 1):
        return {v: (None,) for v in range(widths[i][0], widths[i][1] + 1)}
    plo, phi = widths[i - 1]
    keys = {}
    for v in range(widths[i][0], widths[i][1] + 1):
        lo_u = max(plo, v - step)
        hi_u = min(phi, v + step)
        keys[v] = tuple(v - u for u in range(lo_u, hi_u + 1))
    return keys


def optimal_plan(req, widths) -> list:
    """Recover the lexicographically smallest optimal sequence.

    Assumes the feasibility DP has already proved that ``widths`` admits a
    sequence within the ramp budget.

    Without a reset seam this is one backward DP over positions 0..n-1.  With
    a seam at ``s`` the right side (s+1..n-1) gets its own backward DP; the
    seam-element cell at s joins both sides over the unconstrained seam edge,
    keyed by the ramp budget handed to the right side, so reconstruction can
    reserve ramps globally while recovering each side lexicographically.
    """
    n = req["n"]
    step = req["max_step"]
    cap = req["max_ramps"]
    targets = req["targets"]
    seam = req["reset_after"]

    def vals(i):
        return range(widths[i][0], widths[i][1] + 1)

    # G[i][v] maps state (din, b) -> (S, T), where ``din`` is the delta of
    # the edge entering position i (None at the first element of a side) and
    # ``b`` is the number of *new* ramps still allowed on this side's
    # remaining edges.  S is the minimum sum of absolute errors over the
    # remaining positions of continuations using T <= b new ramps; ties on S
    # are broken by smaller T.  At the seam element S spans both sides and
    # ``b`` is the budget reserved for the whole right side.
    G = [None] * n

    def normal_layer(i):
        nlo, nhi = widths[i + 1]
        keys_here = _incoming_keys(widths, i, step, seam)
        layer = {}
        for v in vals(i):
            ev = abs(v - targets[i])
            wa = max(nlo, v - step)
            wb = min(nhi, v + step)
            feasible_w = list(range(wa, wb + 1))

            # For every remaining budget b >= 1, precompute the best and the
            # second-best successor (distinct values) for starting a new ramp
            # at edge i, compared lexicographically on (S, T, w).
            best_new = {}
            for b in range(1, cap + 1):
                w1 = None
                p1 = None
                w2 = None
                p2 = None
                for w in feasible_w:
                    st = G[i + 1].get(w, {}).get((w - v, b - 1))
                    if st is None:
                        continue
                    cand = (st[0], st[1], w)
                    if p1 is None or cand < p1:
                        w2, p2 = w1, p1
                        w1, p1 = w, cand
                    elif w != w1 and (p2 is None or cand < p2):
                        w2, p2 = w, cand
                if p1 is not None:
                    best_new[b] = (p1, w1, p2, w2)

            cell = {}
            for din in keys_here[v]:
                cont_w = None if din is None else v + din
                for b in range(cap + 1):
                    best = None
                    # Continue the incoming ramp (edge delta == din).
                    if cont_w is not None:
                        st = G[i + 1].get(cont_w, {}).get((din, b))
                        if st is not None:
                            best = (ev + st[0], st[1])
                    # Start a new ramp at edge i.
                    if b >= 1:
                        pre = best_new.get(b)
                        if pre is not None:
                            p1, w1, p2, w2 = pre
                            s0, s1, bw = p1 if w1 != cont_w else (
                                p2 if p2 is not None else (None, None, None))
                            if s0 is not None:
                                cand = (ev + s0, 1 + s1)
                                if best is None or cand < best:
                                    best = cand
                    if best is not None:
                        cell[(din, b)] = best
            if cell:
                layer[v] = cell
        return layer

    def seam_layer(i):
        """Join cell for the left element of the seam (position i == s).

        The edge to i+1 is unconstrained and belongs to no ramp; the state
        budget ``b`` is handed wholesale to the right side (b >= 1, since it
        has at least one edge).  The value pair is (errors on i..n-1, ramps
        used on the right side); the incoming ``din`` still describes the
        ordinary left-side edge (i-1, i), so left ramp decisions at i-1 work
        exactly like every other normal transition.
        """
        nlo, nhi = widths[i + 1]
        keys_here = _incoming_keys(widths, i, step, seam)
        layer = {}
        for v in vals(i):
            ev = abs(v - targets[i])
            by_budget = {}
            for b in range(1, cap + 1):
                best = None
                for w in range(nlo, nhi + 1):
                    st = G[i + 1].get(w, {}).get((None, b))
                    if st is None:
                        continue
                    cand = (st[0], st[1], w)
                    if best is None or cand < best:
                        best = cand
                if best is not None:
                    by_budget[b] = (ev + best[0], best[1])
            if by_budget:
                layer[v] = {(din, b): pair
                            for din in keys_here[v]
                            for b, pair in by_budget.items()}
        return layer

    # Base layer at the final element.
    base = {}
    keys_last = _incoming_keys(widths, n - 1, step, seam)
    for v in vals(n - 1):
        e = abs(v - targets[n - 1])
        base[v] = {(d, b): (e, 0) for d in keys_last[v]
                   for b in range(cap + 1)}
    G[n - 1] = base

    if seam is None:
        fill_from, fill_to = n - 2, -1
    else:
        # Right-side interior first (n-2 .. s+1), then the seam cell, then
        # the left side (s-1 .. 0).
        for i in range(n - 2, seam, -1):
            G[i] = normal_layer(i)
        G[seam] = seam_layer(seam)
        fill_from, fill_to = seam - 1, -1
    for i in range(fill_from, fill_to, -1):
        G[i] = normal_layer(i)

    def reconstruct(lo, hi, budget):
        """Greedy lex-min recovery over one side; returns ramps used."""
        choices = []
        for v in vals(lo):
            st = G[lo].get(v, {}).get((None, budget))
            if st is not None:
                choices.append((st[0], st[1], v))
        if not choices:  # pragma: no cover - guarded by feasibility DP
            raise CompileError("internal reconstruction failure", 500)
        _, _, vi = min(choices, key=lambda z: (z[0], z[1], z[2]))
        x[lo] = vi
        prev_din = None
        used = 0
        for i in range(lo + 1, hi + 1):
            choices = []
            for v in vals(i):
                d = v - x[i - 1]
                if abs(d) > step:
                    continue
                extra = 0 if d == prev_din else 1
                b_remaining = budget - used - extra
                if b_remaining < 0:
                    continue
                st = G[i].get(v, {}).get((d, b_remaining))
                if st is None:
                    continue
                total_ramps = used + extra + st[1]
                choices.append((st[0], total_ramps, v, d))
            if not choices:  # pragma: no cover - guarded by feasibility DP
                raise CompileError("internal reconstruction failure", 500)
            _, _, vi, di = min(choices, key=lambda z: (z[0], z[1], z[2]))
            if di != prev_din:
                used += 1
            prev_din = di
            x[i] = vi
        return used

    x = [0] * n
    if seam is None:
        reconstruct(0, n - 1, cap)
    else:
        used_left = reconstruct(0, seam, cap)
        # The seam cell was reached with exactly the unspent left budget,
        # which it reserved for the right side.
        reconstruct(seam + 1, n - 1, cap - used_left)
    return x


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _ramp_budget_conflict(req):
    items = sorted(req["anchors"].items())
    seam = req["reset_after"]
    pairs = [(a, b) for a, b in zip(items, items[1:])
             if seam is None or not (a[0] <= seam < b[0])]
    segments = [
        {"start": i0, "end": i1, "total_change": v1 - v0, "steps": i1 - i0}
        for (i0, v0), (i1, v1) in pairs
    ]
    return {"kind": "ramp_budget", "start": 0, "end": req["n"] - 1,
            "max_ramps": req["max_ramps"], "anchor_segments": segments}


def ramp_boundaries(x: list, reset_after=None) -> list:
    """Maximal equal-difference runs as [{start, end, delta}, ...].

    ``start``/``end`` are the spanned element indices (both inclusive).

    With ``reset_after = s`` the two sides are tallied independently: the
    last left ramp ends exactly at element ``s`` and the first right ramp
    starts exactly at element ``s+1``, even when both slopes are identical;
    the seam edge (s, s+1) belongs to no ramp.
    """
    if len(x) < 2:
        return []

    def side(lo, hi):
        if hi - lo < 1:
            return []
        out = []
        run_start = lo
        d = x[lo + 1] - x[lo]
        for i in range(lo, hi):
            nd = x[i + 1] - x[i]
            if nd != d:
                out.append({"start": run_start, "end": i, "delta": d})
                run_start = i
                d = nd
        out.append({"start": run_start, "end": hi, "delta": d})
        return out

    if reset_after is None:
        return side(0, len(x) - 1)
    return (side(0, reset_after)
            + side(reset_after + 1, len(x) - 1))


def _saturation_budget(req, bands):
    """Smallest E for which the error tube contains every structural band."""
    t = req["targets"]
    e = 0
    for i, (a, b) in enumerate(bands):
        e = max(e, abs(a - t[i]), abs(b - t[i]))
    return e


def _probe(req, bands, budget):
    """Feasibility at ``budget``.

    Returns True (feasible), False (proven infeasible within the exact
    domain), or None (the exact DP domain exceeds the configured guard).
    """
    widths = _widths_under_budget(req, bands, budget)
    if widths is None:
        return False
    try:
        return feasible_with_widths(req, widths)
    except CompileError:
        return None


def compile_plan(payload) -> dict:
    """Compile a request payload into an optimal integer delay plan.

    Raises :class:`CompileError` for malformed input (400) or infeasible
    instances (422).  Infeasible responses never contain a delay table.
    """
    req = validate_request(payload)
    sr = structural_check(req)
    if not sr.feasible:
        raise CompileError(
            "no feasible delay plan: anchor/step conflict", 422,
            {"conflicts": [c.to_dict() for c in sr.conflicts]})
    bands = sr.bands
    t = req["targets"]
    esat = _saturation_budget(req, bands)

    # Ramp-budget feasibility without any error tube: once bands are fully
    # covered, ramp count is the only remaining restriction.
    if ramp_feasible_on_bands(req, bands) is False:
        raise CompileError(
            "no feasible delay plan within the ramp budget", 422,
            {"conflicts": [_ramp_budget_conflict(req)]})

    # Binary search the optimum budget within the exactly decidable domain.
    # Probe outcomes: True feasible, False infeasible, None means the exact
    # DP domain exceeds the configured guard.  The saturation budget is
    # feasible (ramp feasibility was established above); if probing it raises
    # the guard, locate the largest decidable budget and search below it.
    if _probe(req, bands, esat) is None:
        if _probe(req, bands, 0) is None:  # pragma: no cover - defensive
            raise _guard_refusal(req)
        safe, hi_guard = 0, esat
        while safe + 1 < hi_guard:
            mid = (safe + hi_guard) // 2
            if _probe(req, bands, mid) is None:
                hi_guard = mid
            else:
                safe = mid
        hi_e = safe
    else:
        hi_e = esat
    lo_e = 0

    while lo_e < hi_e:
        mid = (lo_e + hi_e) // 2
        verdict = _probe(req, bands, mid)
        if verdict is True:
            hi_e = mid
        else:
            # False: budget too small.  None cannot occur here because both
            # ends of the active interval are decidable (guard lies above).
            lo_e = mid + 1
    e_star = lo_e

    widths = _widths_under_budget(req, bands, e_star)
    if widths is None or not feasible_with_widths(req, widths):
        # Happens only when the optimum sits above every decidable budget.
        raise _guard_refusal(req)

    x = optimal_plan(req, widths)
    errors = [x[i] - t[i] for i in range(req["n"])]
    seam = req["reset_after"]
    ramps = ramp_boundaries(x, seam)
    plan = {
        "n": req["n"],
        "delays": x,
        "errors": errors,
        "ramps": ramps,
        "ramp_count": len(ramps),
        "max_abs_error": max(abs(e) for e in errors),
        "total_abs_error": sum(abs(e) for e in errors),
    }
    # Echo the seam only when one was requested; omitting it keeps the
    # response byte-for-byte compatible with the seam-less API.
    if seam is not None:
        plan["reset_after"] = seam
    return plan


def _guard_refusal(req):
    return CompileError(
        "integer delay domain too large to compile exactly; tighten "
        "'delay_min'/'delay_max' or reduce the target spread", 400,
        {"field": "delay_min"})
