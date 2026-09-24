# -*- coding: utf-8 -*-
"""Stage 0 细化：S2 子分类 + S1-direct 误路由归因。只读、本地。"""
import json
import glob
import os
from collections import Counter

BASE = "E:/FreeGrasp_code-main/output/failure_mode_diag/EGHC3_v3/split_0"
FILES = sorted(glob.glob(os.path.join(BASE, "details_*.json")))


def load_all():
    rows = []
    for fp in FILES:
        subset = os.path.basename(fp)[len("details_"):-len(".json")]
        for it in json.load(open(fp, encoding="utf-8")):
            it["_subset"] = subset
            rows.append(it)
    return rows


def parse_router_raw(r):
    out = {"candidates": [], "blocking": [], "visibility": None}
    raw = r.get("router_raw_output") or ""
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:]
    try:
        j = json.loads(txt)
    except Exception:
        return out
    out["candidates"] = j.get("target_candidates") or []
    out["blocking"] = j.get("blocking_relations") or []
    out["visibility"] = j.get("target_visibility")
    return out


def norm(s):
    return (s or "").strip().lower()


def blocker_labels(blocking):
    """blocking_relations 里所有 blocker 侧的 label/id 文本。"""
    labs = set()
    for b in blocking if isinstance(blocking, list) else []:
        if isinstance(b, dict):
            for key in ("blocker", "blocker_label", "obstructor", "obstructor_label",
                        "blocking_object", "blocker_name", "object_label"):
                if b.get(key):
                    labs.add(norm(str(b[key])))
            # 任意嵌套字符串字段兜底
            for v in b.values():
                if isinstance(v, str):
                    labs.add(norm(v))
    return labs


def main():
    rows = load_all()
    fails = [it for it in rows if it.get("rsr_correct") == 0]

    s2_sub = Counter()
    s1_direct_sub = Counter()
    examples = {"S2_hit_reported_blocker": [], "S2_miss": [],
                "S1direct_blocking_empty": [], "S1direct_blocking_nonempty": []}

    for it in fails:
        r = it.get("reasoning") or {}
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except Exception:
                r = {}
        sel = r.get("selected_gt_id")
        query_id = it.get("object_id")
        gt_ids = set(it.get("gt_ids") or [])
        route = r.get("route_mode")
        ro = parse_router_raw(r)
        blk = ro["blocking"]
        sel_class = norm(r.get("class_name"))
        blabs = blocker_labels(blk)
        hit_blocker = bool(sel_class) and sel_class in blabs

        if sel == query_id and query_id not in gt_ids:
            if route == "direct":
                key = "blocking_empty" if not blk else "blocking_nonempty"
                s1_direct_sub[key] += 1
                ex = examples["S1direct_" + key]
                if len(ex) < 5:
                    ex.append({"scene": it.get("scene_id"), "query": it.get("object_id"),
                               "gt": sorted(gt_ids), "vis": ro["visibility"],
                               "blk": blk[:2], "ann": it.get("annotation")})
        elif sel != query_id and isinstance(sel, int) and sel > 0 and blk:
            if hit_blocker:
                s2_sub["hit_reported_blocker"] += 1
                if len(examples["S2_hit_reported_blocker"]) < 5:
                    examples["S2_hit_reported_blocker"].append(
                        {"scene": it.get("scene_id"), "query": it.get("object_id"),
                         "sel": sel, "sel_class": sel_class, "gt": sorted(gt_ids),
                         "blk": blk[:2], "ann": it.get("annotation")})
            else:
                s2_sub["not_reported_blocker"] += 1
                if len(examples["S2_miss"]) < 5:
                    examples["S2_miss"].append(
                        {"scene": it.get("scene_id"), "query": it.get("object_id"),
                         "sel": sel, "sel_class": sel_class, "gt": sorted(gt_ids),
                         "blk": blk[:2], "ann": it.get("annotation")})

    print("=== S2 (n=53) 子分类：选中物体是否是 router 报告过的 blocker ===")
    for k, v in s2_sub.most_common():
        print(f"  {k:24s} {v}")
    print("\n  示例 hit_reported_blocker:")
    for e in examples["S2_hit_reported_blocker"]:
        print("   ", json.dumps(e, ensure_ascii=False)[:220])
    print("\n  示例 not_reported_blocker:")
    for e in examples["S2_miss"]:
        print("   ", json.dumps(e, ensure_ascii=False)[:220])

    print("\n=== S1-direct (n=23) 子分类：遮挡证据状态 ===")
    for k, v in s1_direct_sub.most_common():
        print(f"  {k:24s} {v}")
    print("\n  示例 blocking_empty（遮挡漏检嫌疑）:")
    for e in examples["S1direct_blocking_empty"]:
        print("   ", json.dumps(e, ensure_ascii=False)[:220])
    print("\n  示例 blocking_nonempty（有证据仍 direct＝误放行）:")
    for e in examples["S1direct_blocking_nonempty"]:
        print("   ", json.dumps(e, ensure_ascii=False)[:220])

    # blocking_relations 的结构样例（了解字段名）
    print("\n=== blocking_relations 原始结构样例 ===")
    seen = 0
    for it in fails:
        r = it.get("reasoning") or {}
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except Exception:
                r = {}
        ro = parse_router_raw(r)
        if ro["blocking"]:
            print("   ", json.dumps(ro["blocking"][:1], ensure_ascii=False)[:300])
            seen += 1
            if seen >= 3:
                break


if __name__ == "__main__":
    main()
