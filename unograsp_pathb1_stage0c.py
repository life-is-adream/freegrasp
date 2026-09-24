# -*- coding: utf-8 -*-
"""Stage 0 终判：量化「同类多实例混淆」在 114 失败中的占比。只读、本地。"""
import json
import glob
import os
import re
from collections import Counter

BASE = "E:/FreeGrasp_code-main/output/failure_mode_diag/EGHC3_v3/split_0"
FILES = sorted(glob.glob(os.path.join(BASE, "details_*.json")))

STOP = {"the", "a", "an", "on", "of", "in", "at", "top", "bottom", "left", "right",
        "thing", "one", "other", "little", "big", "bigger", "small", "smaller",
        "green", "yellow", "red", "blue", "black", "white", "box", "leaning"}


def words(s):
    return set(w for w in re.findall(r"[a-z]+", (s or "").lower()) if w not in STOP and len(w) > 2)


def main():
    rows = []
    for fp in FILES:
        subset = os.path.basename(fp)[len("details_"):-len(".json")]
        for it in json.load(open(fp, encoding="utf-8")):
            it["_subset"] = subset
            rows.append(it)

    fails = [it for it in rows if it.get("rsr_correct") == 0]
    succ = [it for it in rows if it.get("rsr_correct") == 1]

    def same_class(it):
        r = it.get("reasoning") or {}
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except Exception:
                r = {}
        sel_class = (r.get("class_name") or "").lower().strip()
        ann = (it.get("annotation") or "").lower()
        if not sel_class:
            return None, r
        # sel_class 的核心词是否出现在 annotation 中（如 'knife' ⊂ 'the knife on top of the box'）
        cw = [w for w in words(sel_class)]
        hit = any(w in ann for w in cw) if cw else None
        return hit, r

    n_hit = n_miss = n_noclass = 0
    hit_examples = []
    for it in fails:
        hit, r = same_class(it)
        if hit is None:
            n_noclass += 1
        elif hit:
            n_hit += 1
            if len(hit_examples) < 8:
                hit_examples.append((it.get("scene_id"), it.get("object_id"),
                                     r.get("selected_gt_id"), r.get("class_name"),
                                     it.get("annotation")))
        else:
            n_miss += 1
    print(f"失败 114 例 · 选中物体类别与查询语言同类: {n_hit}, 不同类: {n_miss}, 无类别: {n_noclass}")

    # 对照：成功 186 例的同类率（应显著更高？还是说明同类场景本身难）
    s_hit = s_miss = s_noclass = 0
    for it in succ:
        hit, _ = same_class(it)
        if hit is None:
            s_noclass += 1
        elif hit:
            s_hit += 1
        else:
            s_miss += 1
    print(f"成功 186 例 · 同类: {s_hit}, 不同类: {s_miss}, 无类别: {s_noclass}")
    print(f"失败同类率 {n_hit/114*100:.1f}% vs 成功同类率 {s_hit/186*100:.1f}%")
    print("\n失败·同类混淆示例:")
    for e in hit_examples:
        print("  ", e)

    # 空间关系短语在失败 annotation 中的出现率（消歧线索存在但用错）
    spat = re.compile(r"\b(on top|bottom|left|right|leaning|under|behind|front|near|next|above|below|closest|touch)\b")
    f_spat = sum(1 for it in fails if spat.search((it.get("annotation") or "").lower()))
    s_spat = sum(1 for it in succ if spat.search((it.get("annotation") or "").lower()))
    print(f"\nannotation 含空间关系短语: 失败 {f_spat}/114 ({f_spat/114*100:.1f}%) vs 成功 {s_spat}/186 ({s_spat/186*100:.1f}%)")

    # 数字/类别词统计：失败里 query 目标类别在场景中的重复度（用 class_name 在同 scene 其他样本中出现次数近似）
    # 跨样本近似：统计失败 scene 的 annotation 中同一类别词
    by_scene = {}
    for it in fails:
        by_scene.setdefault(it.get("scene_id"), []).append(it)
    dup_scene = sum(1 for sc, items in by_scene.items() if len(items) > 1)
    print(f"\n失败 case 涉及 scene 数: {len(by_scene)}, 其中多查询同 scene: {dup_scene}")


if __name__ == "__main__":
    main()
