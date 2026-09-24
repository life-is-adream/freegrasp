# -*- coding: utf-8 -*-
"""
UNOGrasp x PathB1 融合可行性 · Stage 0 定性错误分析（2026-09-22）
目的：对 EGHC3_v3 split-0 的 114 个失败 case 做错误形态分类，
      量化「UNOGrasp 式遮挡推理」理论上可修复的比例。
分类判据（预注册于本文件 docstring，跑前写死）：
  S1 早抓目标   : selected_gt_id == 查询目标 object_id，但 gt_ids 不含它
                  → 目标被遮挡、应先移除障碍，模型却直接抓目标。UNOGrasp 核心场景。
  S2 错误障碍   : selected_gt_id != 查询目标，且 router 报告了 blocking_relations（有遮挡证据），
                  且 selected 不在 gt_ids → 遮挡方向对但选错节点（或把非障碍当障碍）。
  S3 无关物体   : selected_gt_id != 查询目标，且无 blocking_relations 证据
                  → 纯 grounding/识别错误，遮挡推理救不了。
  S4 无效输出   : selected_gt_id <= 0 或 parse_status != valid_id。
辅助信号：ambiguity_evidence 非空 / target_candidates 多候选 / route_mode。
"""
import json
import glob
import os
from collections import Counter, defaultdict

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


def get_reasoning(it):
    r = it.get("reasoning")
    if isinstance(r, str):
        try:
            r = json.loads(r)
        except Exception:
            r = {}
    return r if isinstance(r, dict) else {}


def parse_router_raw(r):
    """从 router_raw_output 提取结构化字段（容错）。"""
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


def classify(it):
    r = get_reasoning(it)
    sel = r.get("selected_gt_id")
    query_id = it.get("object_id")
    gt_ids = set(it.get("gt_ids") or [])
    parse_status = r.get("parse_status")
    ro = parse_router_raw(r)
    amb = r.get("ambiguity_evidence") or []
    n_cand = len(ro["candidates"])

    rec = {
        "scene_id": it.get("scene_id"),
        "object_id": query_id,
        "selected_gt_id": sel,
        "gt_ids": sorted(gt_ids),
        "route_mode": r.get("route_mode"),
        "target_label": r.get("target_label"),
        "annotation": it.get("annotation"),
        "target_visibility": ro["visibility"] or r.get("pre_guard_route_assessment", {}).get("target_visibility") if isinstance(r.get("pre_guard_route_assessment"), dict) else ro["visibility"],
        "has_blocking": len(ro["blocking"]) > 0,
        "blocking_relations": ro["blocking"],
        "n_candidates": n_cand,
        "ambiguity": len(amb) > 0,
        "parse_status": parse_status,
        "subset": it["_subset"],
    }

    if (not isinstance(sel, int)) or sel <= 0 or parse_status not in (None, "valid_id"):
        rec["cls"] = "S4_invalid"
    elif sel == query_id and query_id not in gt_ids:
        rec["cls"] = "S1_premature_target"
    elif sel == query_id and query_id in gt_ids:
        # 失败但选中目标且目标在 gt_ids：矛盾，人工核对
        rec["cls"] = "X_anomaly"
    elif len(ro["blocking"]) > 0:
        rec["cls"] = "S2_wrong_blocker"
    else:
        rec["cls"] = "S3_unrelated_object"
    return rec


def main():
    rows = load_all()
    print(f"total samples: {len(rows)}")
    fails = [it for it in rows if it.get("rsr_correct") == 0]
    print(f"failures: {len(fails)}")

    recs = [classify(it) for it in fails]
    by_cls = Counter(rec["cls"] for rec in recs)
    print("\n=== 失败分类（全部 114 例） ===")
    for c, n in by_cls.most_common():
        print(f"  {c:26s} {n:4d}  ({n/len(recs)*100:.1f}%)")

    # 按 route_mode 交叉
    print("\n=== 分类 x route_mode ===")
    cross = defaultdict(Counter)
    for rec in recs:
        cross[rec["route_mode"]][rec["cls"]] += 1
    for rm in sorted(cross, key=lambda x: str(x)):
        print(f"  {rm}: {dict(cross[rm])}")

    # blocker_first 深挖
    bf = [rec for rec in recs if rec["route_mode"] == "blocker_first"]
    print(f"\n=== blocker_first 失败 {len(bf)} 例 · 分类 ===")
    by_cls_bf = Counter(rec["cls"] for rec in bf)
    for c, n in by_cls_bf.most_common():
        print(f"  {c:26s} {n:4d}  ({n/len(bf)*100:.1f}%)")

    # 语言歧义信号
    n_amb = sum(1 for rec in recs if rec["ambiguity"])
    n_multicand = sum(1 for rec in recs if rec["n_candidates"] > 1)
    print(f"\nambiguity_evidence 非空: {n_amb}/{len(recs)}; 多候选(>1): {n_multicand}/{len(recs)}")

    # S1 明细（UNOGrasp 核心可修集）按 route_mode
    s1 = [rec for rec in recs if rec["cls"] == "S1_premature_target"]
    print(f"\n=== S1 早抓目标（n={len(s1)}）按 route_mode ===")
    print("  ", dict(Counter(rec["route_mode"] for rec in s1)))

    # 输出 S1/S2 明细供人工复核（前 30 条）
    out = []
    for rec in recs:
        out.append({k: rec[k] for k in ("scene_id", "object_id", "selected_gt_id",
                                        "gt_ids", "route_mode", "cls", "target_label",
                                        "annotation", "n_candidates", "ambiguity",
                                        "has_blocking", "subset")})
    outp = "E:/FreeGrasp_code-main/output/failure_mode_diag/stage0_error_taxonomy.json"
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n明细已写: {outp}")

    # 成功 case 里 S1 型（选中目标且目标在 gt_ids 且有遮挡）对照
    succ = [it for it in rows if it.get("rsr_correct") == 1]
    succ_rec = [classify(it) for it in succ]
    s1_like_succ = [rec for rec in succ_rec
                    if rec["cls"] == "S2_wrong_blocker" and rec["selected_gt_id"] == rec["object_id"]]
    print(f"\n对照：成功 case 中 selected==query 且有遮挡证据: {len(s1_like_succ)}")


if __name__ == "__main__":
    main()
