# -*- coding: utf-8 -*-
"""探测 occlusion / occlusion_objects 的像素语义（只读）。"""
import numpy as np

NPZ = "/home/public/zb/FreeGrasp_code-freegrasp_codex/data/npz_file"

for scene in [0, 2276]:
    d = np.load(f"{NPZ}/{scene}.npz", allow_pickle=True)
    inst = d["instances_objects"]
    occ = d["occlusion"]
    occ_o = d["occlusion_objects"]
    print(f"=== scene {scene} ===")
    print("inst dtype", inst.dtype, "unique:", [float(v) for v in np.unique(inst)])
    print("occ>0 total:", int((occ > 0).sum()), " occ_o>0 per-layer:", [int((occ_o[i] > 0).sum()) for i in range(occ_o.shape[0])])
    # 逐层:与各 instance mask 的交
    for i in range(occ_o.shape[0]):
        m = occ_o[i] > 0
        if m.sum() == 0:
            continue
        inter = {int(v): int(np.logical_and(m, inst == v).sum()) for v in np.unique(inst)}
        inter = {k: v for k, v in inter.items() if v > 0}
        top = sorted(inter.items(), key=lambda kv: -kv[1])[:4]
        own = inter.get(i if i in inter else float(i), None)
        print(f" layer {i}: top-overlap instances {top}")
    # occlusion 单通道 与 occlusion_objects 的关系
    union_o = np.logical_or.reduce([occ_o[i] > 0 for i in range(occ_o.shape[0])])
    print(" occ>0 == union(occ_o>0)?", bool(np.array_equal(occ > 0, union_o)))
    # occlusion 像素落在哪些 instance 上
    m = occ > 0
    inter = {int(v): int(np.logical_and(m, inst == v).sum()) for v in np.unique(inst)}
    print(" occlusion>0 by instance:", {k: v for k, v in sorted(inter.items(), key=lambda kv: -kv[1])[:6]})
