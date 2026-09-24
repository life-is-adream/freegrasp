"""Failure-mode diagnosis on FreeGrasp split-0 details (EGHC3_v3 + PATHB_FULL_B1_blk_arb).

Goal: classify each FAILED case (rsr_correct=='0') into a failure bucket to test the
hypothesis that the bottleneck is perception (mask/visibility), not reasoning (selection/
routing). Directly informs whether UNOGrasp-style perception improvements are worth pursuing.

Heuristic classification (priority order):
  D  INFRA   -> top-level error != 'None' (API/system, not model)
  A1 PARSE   -> parse_status != 'valid_id' (model output unusable)
  A2 SEL     -> selected_gt_id not in gt_ids (wrong target chosen) -> reasoning
  B  PERC    -> correct target, but mask pipeline failed
               (langsam_failure_reason != 'None' OR langsam_mask_contains_point=='False')
               -> perception
  C  POSE    -> correct target, mask ok, but still failed -> grasp pose / sim physics
  E  ROUTE   -> (secondary) route_mode contradicts scene complexity evidence
               (visibility/blocking suggests blocker_first but route==direct) -> reasoning

Read-only. No API, no GPU. Output ASCII + markdown.
"""
import json
import os
from collections import Counter, defaultdict

BASE = "E:/FreeGrasp_code-main/output/failure_mode_diag"
ARMS = ["EGHC3_v3", "PATHB_FULL_B1_blk_arb"]
SUBSETS = [
    "easy_no_ambi", "easy_yes_ambi",
    "medium_no_ambi", "medium_yes_ambi",
    "hard_no_ambi", "hard_yes_ambi",
]


def load(arm):
    rows = []
    for s in SUBSETS:
        p = os.path.join(BASE, arm, "split_0", "details_%s.json" % s)
        data = json.load(open(p, encoding="utf-8"))
        for it in data:
            rows.append((s, it))
    return rows


def g(d, *keys, default=None):
    """Read first present key (top-level or nested under reasoning)."""
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in (None, ""):
            return d[k]
    r = d.get("reasoning") if isinstance(d, dict) else None
    if isinstance(r, dict):
        for k in keys:
            if k in r and r[k] not in (None, ""):
                return r[k]
    return default


def flatten_ids(v):
    out = set()
    if v is None:
        return out
    if isinstance(v, (list, tuple)):
        for x in v:
            out |= flatten_ids(x)
    else:
        out.add(str(v))
    return out


def classify(it):
    """Return (bucket, reason_str)."""
    rsr = str(g(it, "rsr_correct") or "0")
    if rsr == "1":
        return ("OK", "")
    err = str(g(it, "error") or "None")
    if err not in ("None", "null", ""):
        return ("D_INFRA", "error=%s" % err[:60])
    parse = str(g(it, "parse_status") or "None")
    if parse != "valid_id":
        return ("A1_PARSE", "parse_status=%s" % parse)
    gt = flatten_ids(it.get("gt_ids"))
    sel = str(g(it, "selected_gt_id") or "None")
    if sel == "None" or (gt and sel not in gt):
        return ("A2_SEL", "selected_gt_id=%s not in gt_ids=%s" % (sel, sorted(gt)))
    # correct target -> check mask/perception
    lf = str(g(it, "langsam_failure_reason") or "None")
    contains = str(g(it, "langsam_mask_contains_point") or "None")
    if lf not in ("None", "null", "") or contains == "False":
        return ("B_PERC", "langsam_failure=%s contains_point=%s" % (lf[:40], contains))
    # correct target, mask ok -> pose/sim
    return ("C_POSE", "target ok, mask ok, rsr=0")


def visibility_of(it):
    v = g(it, "target_visibility")
    return str(v or "unk")


def route_of(it):
    for k in ("route_mode", "pre_guard_route_mode", "recommended_route"):
        v = g(it, k)
        if v in ("direct", "target_first", "blocker_first"):
            return v
    return "unk"


def run():
    for arm in ARMS:
        rows = load(arm)
        total = len(rows)
        failed = [r for r in rows if str(g(r[1], "rsr_correct") or "0") == "0"]
        buckets = Counter()
        by_subset = defaultdict(Counter)
        by_vis = defaultdict(Counter)
        by_route = defaultdict(Counter)
        examples = defaultdict(list)
        for s, it in rows:
            b, _ = classify(it)
            buckets[b] += 1
            if b != "OK":
                by_subset[s][b] += 1
                by_vis[visibility_of(it)][b] += 1
                by_route[route_of(it)][b] += 1
                if len(examples[b]) < 3:
                    examples[b].append((s, g(it, "scene_id"), g(it, "object_id"),
                                        visibility_of(it), route_of(it),
                                        str(g(it, "langsam_failure_reason") or "None")[:30],
                                        str(g(it, "iou") or "?")[:8]))
        nf = len(failed)
        print("=" * 78)
        print("ARM %s   total=%d  failed(rsr=0)=%d (%.1f%%)" % (arm, total, nf, 100.0*nf/total))
        print("=" * 78)
        print("Bucket composition (all cases):")
        for b in ("OK", "A1_PARSE", "A2_SEL", "B_PERC", "C_POSE", "D_INFRA"):
            c = buckets.get(b, 0)
            print("  %-10s %4d  (%5.1f%% of all, %5.1f%% of failed)" %
                  (b, c, 100.0*c/total, 100.0*c/nf if nf else 0))
        # reasoning vs perception vs other (failed only)
        reasoning = buckets.get("A1_PARSE",0)+buckets.get("A2_SEL",0)
        perception = buckets.get("B_PERC",0)
        pose = buckets.get("C_POSE",0)
        infra = buckets.get("D_INFRA",0)
        print("")
        print("FAILED-case diagnosis (n=%d):" % nf)
        print("  reasoning (parse+sel) = %d (%.1f%%)" % (reasoning, 100.0*reasoning/nf))
        print("  perception (mask)      = %d (%.1f%%)" % (perception, 100.0*perception/nf))
        print("  pose/sim               = %d (%.1f%%)" % (pose, 100.0*pose/nf))
        print("  infra/error            = %d (%.1f%%)" % (infra, 100.0*infra/nf))
        print("")
        print("By difficulty subset (failed counts):")
        for s in SUBSETS:
            cc = by_subset.get(s, Counter())
            tot_s = sum(buckets[("OK")] for _ in [0])  # placeholder
            line = "  %-16s " % s
            for b in ("A1_PARSE","A2_SEL","B_PERC","C_POSE","D_INFRA"):
                if cc.get(b):
                    line += "%s=%d " % (b.replace("_","")[:5], cc[b])
            print(line)
        print("")
        print("By target_visibility (failed counts):")
        for v in sorted(by_vis):
            cc = by_vis[v]
            print("  %-14s total_fail=%d  perc=%d sel=%d pose=%d" %
                  (v, sum(cc.values()), cc.get("B_PERC",0), cc.get("A2_SEL",0), cc.get("C_POSE",0)))
        print("")
        print("Examples per bucket:")
        for b in ("A2_SEL","B_PERC","C_POSE","A1_PARSE","D_INFRA"):
            for ex in examples.get(b, []):
                print("  %-8s %s" % (b, ex))
        print("")


if __name__ == "__main__":
    run()
