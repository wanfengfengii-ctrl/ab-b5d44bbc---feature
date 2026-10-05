"""One-shot verification entry point (used by the ``verify`` compose service).

Aggregates three independent checks and exits non-zero if any of them fails:

1. **code tests**      -- the full unittest suite (compiler + HTTP layer);
2. **build artifacts** -- the manifest baked at image build time is compared
                          against the files actually present in the image;
3. **API smoke**       -- health probe plus compile requests against the
                          running service (``API_BASE_URL``): a typical
                          request with the reset seam omitted, the same
                          instance with ``reset_after`` enabled (ramp
                          boundaries must stop at the seam and equal slopes
                          must not merge), an infeasible request whose
                          conflict is confined to one sub-array, and a
                          seam-less infeasible request; if no service URL is
                          given, an in-process server is started for the
                          smoke checks.

The final line is a machine-readable summary and the process exit code is the
number of failed check groups (0 == all green).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MANIFEST_PATH = os.path.join(ROOT, "build-artifacts", "manifest.json")
SMOKE_TIMEOUT = 10.0

TYPICAL_REQUEST = {
    "targets": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
    "delay_min": 0,
    "delay_max": 100,
    "max_step": 4,
    "max_ramps": 4,
    "anchors": [
        {"index": 0, "value": 10},
        {"index": 8, "value": 26},
        {"index": 15, "value": 40},
    ],
}

CONFLICT_REQUEST = dict(
    TYPICAL_REQUEST,
    max_step=1,
    anchors=[{"index": 0, "value": 10}, {"index": 15, "value": 40}],
)

# Reset seam after element 7: the two sub-arrays independently hold the same
# +2 slope, which must be tallied as two ramps (never merged across the seam)
# with ramp boundaries stopping exactly at elements 7 and 8.
SEAM_REQUEST = dict(TYPICAL_REQUEST, reset_after=7)

# A step/anchor conflict confined to the left sub-array (elements 0..7); the
# seam must not make it reachable, and the 422 must leak no delay table.
SEAM_CONFLICT_REQUEST = dict(
    TYPICAL_REQUEST,
    reset_after=7,
    max_step=1,
    anchors=[{"index": 0, "value": 10},
             {"index": 5, "value": 40},
             {"index": 15, "value": 40}],
)


# ---------------------------------------------------------------------------
# Check 1: code tests
# ---------------------------------------------------------------------------


def check_code_tests():
    print("== [1/3] code tests ==")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
    )
    ok = proc.returncode == 0
    print(f"   code tests: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Check 2: build artifacts
# ---------------------------------------------------------------------------


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_build_artifacts():
    print("== [2/3] build artifacts ==")
    if not os.path.exists(MANIFEST_PATH):
        print(f"   manifest missing: {MANIFEST_PATH}")
        return False
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    problems = []
    for rel, expected in sorted(manifest["files"].items()):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            problems.append(f"missing: {rel}")
            continue
        actual = _sha256(path)
        if actual != expected:
            problems.append(f"hash mismatch: {rel}")
    if problems:
        for p in problems[:20]:
            print(f"   {p}")
        print("   build artifacts: FAIL")
        return False
    print(f"   verified {manifest['file_count']} artifact files: PASS")
    return True


# ---------------------------------------------------------------------------
# Check 3: API smoke
# ---------------------------------------------------------------------------


def _request(method, url, payload=None, timeout=SMOKE_TIMEOUT):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


class _InProcessServer:
    def __init__(self):
        from app.server import build_server
        self.httpd = build_server("127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"


def _wait_healthy(base_url, attempts=30):
    for _ in range(attempts):
        try:
            status, _ = _request("GET", f"{base_url}/healthz", timeout=2)
            if status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1.0)
    return False


def _validate_plan(body):
    plan = body["plan"]
    n = len(plan["delays"])
    req = TYPICAL_REQUEST
    x = plan["delays"]
    assert n == len(req["targets"]), "plan must cover every element"
    assert len(plan["errors"]) == n, "one error per element"
    for a in req["anchors"]:
        assert x[a["index"]] == a["value"], "anchor must be hit exactly"
    for i, v in enumerate(x):
        assert req["delay_min"] <= v <= req["delay_max"], "global interval"
        assert plan["errors"][i] == v - req["targets"][i], "error values"
    for i in range(n - 1):
        assert abs(x[i + 1] - x[i]) <= req["max_step"], "step limit"
    ramps = plan["ramps"]
    assert ramps and ramps[0]["start"] == 0 and ramps[-1]["end"] == n - 1
    for r, s in zip(ramps, ramps[1:]):
        assert r["end"] == s["start"] and r["delta"] != s["delta"]
    assert plan["ramp_count"] == len(ramps) <= req["max_ramps"]
    assert plan["max_abs_error"] == max(abs(e) for e in plan["errors"])
    assert plan["total_abs_error"] == sum(abs(e) for e in plan["errors"])


def _validate_seam_plan(body):
    plan = body["plan"]
    n = len(plan["delays"])
    req = SEAM_REQUEST
    seam = req["reset_after"]
    x = plan["delays"]
    assert plan.get("reset_after") == seam, "seam must be echoed"
    assert n == len(req["targets"]), "plan must cover every element"
    assert len(plan["errors"]) == n, "one error per element"
    for a in req["anchors"]:
        assert x[a["index"]] == a["value"], "anchor must be hit exactly"
    for i, v in enumerate(x):
        assert req["delay_min"] <= v <= req["delay_max"], "global interval"
        assert plan["errors"][i] == v - req["targets"][i], "error values"
    for i in range(n - 1):
        if i == seam:
            continue  # the seam edge is exempt from the step limit
        assert abs(x[i + 1] - x[i]) <= req["max_step"], "step limit in sub-array"
    ramps = plan["ramps"]
    left = [r for r in ramps if r["end"] <= seam]
    right = [r for r in ramps if r["start"] >= seam + 1]
    assert left and right, "both sub-arrays report ramps"
    assert left[0]["start"] == 0 and left[-1]["end"] == seam, "left boundary"
    assert (right[0]["start"] == seam + 1
            and right[-1]["end"] == n - 1), "right boundary"
    for group in (left, right):
        for r, s in zip(group, group[1:]):
            assert r["end"] == s["start"] and r["delta"] != s["delta"]
        for r in group:
            for i in range(r["start"], r["end"]):
                assert x[i + 1] - x[i] == r["delta"]
    assert plan["ramp_count"] == len(ramps) <= req["max_ramps"]
    # identical slopes on both sides must never merge
    assert left[-1]["delta"] == right[0]["delta"] == 2
    assert len(ramps) == 2


def check_api_smoke():
    print("== [3/3] API smoke ==")
    base_url = os.environ.get("API_BASE_URL")
    inproc = None
    if not base_url:
        inproc = _InProcessServer()
        inproc.__enter__()
        base_url = inproc.base_url
        print(f"   (no API_BASE_URL set; started in-process server at {base_url})")
    base_url = base_url.rstrip("/")
    try:
        if not _wait_healthy(base_url):
            print(f"   health check failed for {base_url}")
            return False
        print("   healthz: PASS")

        status, body = _request("POST",
                                f"{base_url}/api/delay-plans/compile",
                                TYPICAL_REQUEST)
        if status != 200:
            print(f"   typical compile returned {status}: {body}")
            return False
        try:
            _validate_plan(body)
        except AssertionError as e:
            print(f"   plan validation failed: {e}")
            print(json.dumps(body, indent=2))
            return False
        plan = body["plan"]
        assert "reset_after" not in plan, "omitted seam stays out of response"
        print(f"   typical compile (seam omitted): PASS "
              f"(E_max={plan['max_abs_error']}, E_sum={plan['total_abs_error']}, "
              f"ramps={plan['ramp_count']})")

        status, body = _request("POST",
                                f"{base_url}/api/delay-plans/compile",
                                SEAM_REQUEST)
        if status != 200:
            print(f"   seam compile returned {status}: {body}")
            return False
        try:
            _validate_seam_plan(body)
        except AssertionError as e:
            print(f"   seam plan validation failed: {e}")
            print(json.dumps(body, indent=2))
            return False
        plan = body["plan"]
        print(f"   seam compile (reset_after={plan['reset_after']}): PASS "
              f"(E_max={plan['max_abs_error']}, E_sum={plan['total_abs_error']}, "
              f"ramps={plan['ramp_count']}, boundaries "
              f"{plan['ramps'][0]['start']}-{plan['ramps'][0]['end']}/"
              f"{plan['ramps'][1]['start']}-{plan['ramps'][1]['end']})")

        status, body = _request("POST",
                                f"{base_url}/api/delay-plans/compile",
                                SEAM_CONFLICT_REQUEST)
        if status != 422:
            print(f"   seam conflict request expected 422, got {status}: {body}")
            return False
        if "conflicts" not in body or not body["conflicts"]:
            print("   422 body must list conflict intervals")
            return False
        if "delays" in body or "plan" in body:
            print("   infeasible response leaked a partial delay table")
            return False
        c0 = body["conflicts"][0]
        assert c0["end"] <= SEAM_CONFLICT_REQUEST["reset_after"], (
            "conflict interval must stay inside one sub-array")
        print(f"   seam conflict request: PASS (422 {c0['kind']} "
              f"[{c0['start']},{c0['end']}], no partial table)")

        status, body = _request("POST",
                                f"{base_url}/api/delay-plans/compile",
                                CONFLICT_REQUEST)
        if status != 422:
            print(f"   conflict request expected 422, got {status}: {body}")
            return False
        if "conflicts" not in body or not body["conflicts"]:
            print("   422 body must list conflict intervals")
            return False
        if "delays" in body or "plan" in body:
            print("   infeasible response leaked a partial delay table")
            return False
        c0 = body["conflicts"][0]
        print(f"   conflict request: PASS (422 {c0['kind']} "
              f"[{c0['start']},{c0['end']}], no partial table)")
        return True
    finally:
        if inproc is not None:
            inproc.__exit__(None, None, None)


def main():
    results = {
        "code_tests": check_code_tests(),
        "build_artifacts": check_build_artifacts(),
        "api_smoke": check_api_smoke(),
    }
    print()
    print("== verify summary ==")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    failed = sum(1 for ok in results.values() if not ok)
    print(json.dumps({"results": results, "failed": failed}))
    print("VERIFY:", "ALL PASS" if failed == 0 else f"{failed} GROUP(S) FAILED")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
