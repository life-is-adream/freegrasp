"""Stage A: zero-cost offline paired re-analysis of PathB B1 vs EGHC3_v3 (split-0).

Read-only. No API, no GPU, no writes. Runs on server with data in place.
Output is pure ASCII to avoid remote locale issues.
"""
import json
import math
import os
import random
from collections import Counter

BASE = "/home/public/zb/FreeGrasp_code-freegrasp_codex/data/output/ablation_matrix"
ARM_A = "EGHC3_v3"                 # baseline (highest deployable)
ARM_B = "PATHB_FULL_B1_blk_arb"    # B1
SPLIT = "split_0"
SUBSETS = [
    "easy_no_ambi", "easy_yes_ambi",
    "medium_no_ambi", "medium_yes_ambi",
    "hard_no_ambi", "hard_yes_ambi",
]
ROUTE_KEYS = ["route_mode", "pre_guard_route_mode", "recommended_route", "frozen_route_used"]
VALID_ROUTES = {"direct", "target_first", "blocker_first"}
SEED = 20260822
B_BOOT = 2000


def load_arm(arm):
    rows = []
    for s in SUBSETS:
        p = os.path.join(BASE, arm, SPLIT, "details_%s.json" % s)
        if not os.path.exists(p):
            print("MISSING FILE: %s" % p)
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for it in data:
            rows.append((s, it))
    return rows


def getval(d, key):
    """Top-level first, then recursive; skip None/empty."""
    if isinstance(d, dict):
        if key in d and d[key] not in (None, ""):
            return d[key]
        for v in d.values():
            if isinstance(v, dict):
                r = getval(v, key)
                if r not in (None, ""):
                    return r
    return None


def route_of(item):
    for k in ROUTE_KEYS:
        v = getval(item, k)
        if v in VALID_ROUTES:
            return k, v
    return None, None


def mcnemar_exact(disc_a, disc_b):
    """Two-sided exact McNemar. disc_a = A-win discordant, disc_b = B-win discordant."""
    n = disc_a + disc_b
    if n == 0:
        return 1.0
    k = min(disc_a, disc_b)
    tot = sum(math.comb(n, i) for i in range(0, k + 1))
    p = 2.0 * tot * (0.5 ** n)
    return min(1.0, p)


def boot_ci(deltas, seed=SEED, B=B_BOOT):
    """Paired bootstrap CI for mean delta, expressed in percentage points."""
    n = len(deltas)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        s = 0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        vals.append(s / float(n) * 100.0)
    vals.sort()
    lo = vals[int(0.025 * B)]
    hi = vals[int(0.975 * B)]
    return (lo, hi)


def paired(a_list, b_list, label):
    """a = baseline, b = B1. delta = b - a (positive => B1 better)."""
    n = len(a_list)
    if n == 0:
        print("  %-28s n=0 (skip)" % label)
        return
    sum_a = sum(a_list)
    sum_b = sum(b_list)
    disc_a = sum(1 for i in range(n) if a_list[i] == 1 and b_list[i] == 0)  # B1 lost
    disc_b = sum(1 for i in range(n) if a_list[i] == 0 and b_list[i] == 1)  # B1 won
    deltas = [b_list[i] - a_list[i] for i in range(n)]
    delta = (sum_b - sum_a) / float(n) * 100.0
    lo, hi = boot_ci(deltas)
    p = mcnemar_exact(disc_a, disc_b)
    verdict = "CI CROSSES 0 -> NULL (no significant change)" if (lo <= 0 <= hi) else "CI excludes 0"
    print("  %-28s n=%-4d A=%d B=%d  delta=%+.2f pp  95%%CI [%+.2f, %+.2f]" %
          (label, n, sum_a, sum_b, delta, lo, hi))
    print("  %-28s discordant: B1_won=%d B1_lost=%d  McNemar exact p=%.4f  => %s" %
          ("", disc_b, disc_a, p, verdict))


def main():
    print("=" * 78)
    print("STAGE A - PAIRED OFFLINE RE-ANALYSIS (split-0, read-only)")
    print("baseline A = %s   |   treatment B = %s (pathb_blk_arb_v1)" % (ARM_A, ARM_B))
    print("=" * 78)

    rows_a = load_arm(ARM_A)
    rows_b = load_arm(ARM_B)
    print("loaded: A rows=%d  B rows=%d" % (len(rows_a), len(rows_b)))

    def index(rows):
        out = {}
        miss = 0
        for s, it in rows:
            sid = getval(it, "scene_id")
            oid = getval(it, "object_id")
            if sid is None or oid is None:
                miss += 1
                continue
            out[(sid, oid)] = (s, it)
        return out, miss

    ia, miss_a = index(rows_a)
    ib, miss_b = index(rows_b)
    print("keyed: A=%d (missing_id=%d)  B=%d (missing_id=%d)" % (len(ia), miss_a, len(ib), miss_b))

    keys_a = set(ia)
    keys_b = set(ib)
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    common = sorted(keys_a & keys_b)
    print("PAIRING: common=%d  only_in_A=%d  only_in_B=%d" % (len(common), len(only_a), len(only_b)))
    if only_a:
        print("  sample only_in_A: %s" % sorted(only_a)[:5])
    if only_b:
        print("  sample only_in_B: %s" % sorted(only_b)[:5])
    if len(common) == 0:
        print("FATAL: no common keys, cannot pair")
        return

    # sanity vs published summary
    def tot(idx, keys, field):
        return sum(int(getval(idx[k][1], field) or 0) for k in keys)

    print("")
    print("-- sanity check against full_protocol_summary --")
    print("  A over common: RSR=%d SSR=%d (summary split-0: RSR=186 SSR=172)" %
          (tot(ia, common, "rsr_correct"), tot(ia, common, "ssr_correct")))
    print("  B over common: RSR=%d SSR=%d (B1 split-0:          RSR=186 SSR=171)" %
          (tot(ib, common, "rsr_correct"), tot(ib, common, "ssr_correct")))

    # route field probing
    print("")
    print("-- route field probe (which key holds the route, and value distribution) --")
    used_key = None
    for k in ROUTE_KEYS:
        ca = Counter()
        cb = Counter()
        for key in common:
            va = getval(ia[key][1], k)
            vb = getval(ib[key][1], k)
            if va not in (None, ""):
                ca[va] += 1
            if vb not in (None, ""):
                cb[vb] += 1
        print("  %-22s A=%s" % (k, dict(ca)))
        print("  %-22s B=%s" % ("", dict(cb)))
        if used_key is None and ca:
            used_key = k
    print("  -> using route key: %s" % used_key)

    # build paired vectors
    a_rsr, b_rsr, a_ssr, b_ssr = [], [], [], []
    route_a, subsets_of = [], []
    for key in common:
        it_a = ia[key][1]
        it_b = ib[key][1]
        a_rsr.append(int(getval(it_a, "rsr_correct") or 0))
        b_rsr.append(int(getval(it_b, "rsr_correct") or 0))
        a_ssr.append(int(getval(it_a, "ssr_correct") or 0))
        b_ssr.append(int(getval(it_b, "ssr_correct") or 0))
        _, rv = route_of(it_a)   # subset defined by BASELINE route (treatment-independent)
        route_a.append(rv)
        subsets_of.append(ia[key][0])

    print("")
    print("=" * 78)
    print("PRIMARY ENDPOINT - full paired sample (n=%d)" % len(common))
    print("=" * 78)
    paired(a_rsr, b_rsr, "RSR")
    paired(a_ssr, b_ssr, "SSR")

    # mechanism: blocker_first subset, defined by BASELINE arm route
    print("")
    print("=" * 78)
    print("MECHANISM - stratified by BASELINE route (exploratory, pre-registered)")
    print("=" * 78)
    groups = {}
    for i, rv in enumerate(route_a):
        groups.setdefault(rv, []).append(i)
    for rv in sorted(groups, key=lambda x: str(x)):
        idxs = groups[rv]
        print("")
        print("route = %s  (baseline-defined), n=%d" % (rv, len(idxs)))
        paired([a_rsr[i] for i in idxs], [b_rsr[i] for i in idxs], "  RSR")
        paired([a_ssr[i] for i in idxs], [b_ssr[i] for i in idxs], "  SSR")

    # difficulty strata
    print("")
    print("=" * 78)
    print("EXPLORATORY - by difficulty subset")
    print("=" * 78)
    groups = {}
    for i, s in enumerate(subsets_of):
        groups.setdefault(s, []).append(i)
    for s in SUBSETS:
        if s not in groups:
            continue
        idxs = groups[s]
        print("")
        print("subset = %s, n=%d" % (s, len(idxs)))
        paired([a_rsr[i] for i in idxs], [b_rsr[i] for i in idxs], "  RSR")
        paired([a_ssr[i] for i in idxs], [b_ssr[i] for i in idxs], "  SSR")

    print("")
    print("=" * 78)
    print("bootstrap: %d resamples, seed=%d, cluster = per sample" % (B_BOOT, SEED))
    print("=" * 78)


main()
