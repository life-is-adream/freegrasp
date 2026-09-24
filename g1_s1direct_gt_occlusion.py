# -*- coding: utf-8 -*-
"""G1 v2: GT 遮挡核实 · S1-direct 23 例（简化判据版）

依据 g1_probe_occlusion.py 的探测结论 [E]：
  - instances_objects: instance id = Molmo/visible id（identity），0 = 背景
  - 单通道 occlusion: 像素级被遮挡图（量级大，覆盖各被挡实例）
  - occlusion_objects: 层数≠实例数、含自遮挡伪影 → 弃用
判据（预注册）：
  occ_ratio(query) = |(occlusion>0) ∧ (instances==query)| / |instances==query|
  ≥ 0.15 → GT_CONFIRMS_OCCLUSION（router free_visible = 漏检，理论上可修）
  < 0.05 → GT_NO_OCCLUSION（GT 认为目标未被挡；首抓集合不含目标属他因 → prompt 改造不可修）
  其余   → EDGE
"""
import json
import os
import numpy as np

NPZ_DIR = "/home/public/zb/FreeGrasp_code-freegrasp_codex/data/npz_file"
OUT = "/home/public/zb/freegrasp_unograsp_fusion_20260922_v1/g1_s1direct_result.json"

CASES = [
    (2274, 3, [2]), (3576, 2, [3]), (3576, 4, [3]), (6784, 8, [9]),
    (6837, 5, [2]), (6949, 10, [4]), (7108, 11, [3]), (7228, 4, [1]),
    (59, 1, [3]), (1961, 9, [8]), (4156, 3, [7]), (206, 6, [3]),
    (988, 1, [4]), (1104, 8, [7]), (1670, 11, [3]), (1860, 2, [4]),
    (2052, 6, [1, 3]), (6897, 4, [6]), (501, 4, [3]), (774, 4, [3]),
    (1357, 5, [3]), (1978, 4, [3]), (2275, 2, [1]),
]

# 对照组：blocker_first 失败 5 例（GT 应显示目标被挡）+ 成功 direct 3 例（GT 应显示无遮挡）
CHECK_BLOCKED = [(5636, 2, [2]), (2276, 2, [2]), (1206, 5, [5]), (1670, 6, [6]), (564, 5, [5])]
CHECK_FREE = [(0, 3, [3])]


def occ_ratio(d, oid):
    inst = d["instances_objects"]
    occ = d["occlusion"]
    mask = (inst == float(oid))
    denom = int(mask.sum())
    if denom == 0:
        return None, 0
    return round(float(np.logical_and(occ > 0, mask).sum()) / denom, 4), denom


def run(case_list, tag, results):
    for scene, query, gt_ids in case_list:
        p = os.path.join(NPZ_DIR, f"{scene}.npz")
        if not os.path.exists(p):
            results.append({"tag": tag, "scene": scene, "query": query,
                            "gt_ids": gt_ids, "status": "NO_NPZ"})
            continue
        d = np.load(p, allow_pickle=True)
        inst_vals = set(int(v) for v in np.unique(d["instances_objects"]))
        r, denom = occ_ratio(d, query)
        if r is None:
            results.append({"tag": tag, "scene": scene, "query": query,
                            "gt_ids": gt_ids, "status": "NO_INSTANCE",
                            "inst_vals": sorted(inst_vals)})
            continue
        if tag == "S1_direct":
            verdict = ("GT_CONFIRMS_OCCLUSION" if r >= 0.15
                       else "GT_NO_OCCLUSION" if r < 0.05 else "EDGE")
        else:
            verdict = "CONTROL"
        results.append({"tag": tag, "scene": scene, "query": query,
                        "gt_ids": gt_ids, "status": "OK", "occ_ratio": r,
                        "mask_px": denom, "verdict": verdict})


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = []
    run(CASES, "S1_direct", results)
    run(CHECK_BLOCKED, "CTRL_blocked_first_fail", results)
    run(CHECK_FREE, "CTRL_direct_success", results)
    from collections import Counter
    s1 = [r for r in results if r["tag"] == "S1_direct" and r.get("status") == "OK"]
    summary = Counter(r["verdict"] for r in s1)
    out = {"summary_s1direct": dict(summary), "results": results}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("== S1-direct verdict summary ==", dict(summary))
    print("== detail ==")
    for r in results:
        print(r["tag"], r["scene"], "q", r["query"], "gt", r["gt_ids"],
              r.get("status"), r.get("occ_ratio"), r.get("verdict", ""))
    print("== written ==", OUT)


if __name__ == "__main__":
    main()
