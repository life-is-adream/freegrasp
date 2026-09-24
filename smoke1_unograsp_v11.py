# -*- coding: utf-8 -*-
"""UNOGrasp×PathB1 融合 · Smoke 1（3 例，RDR-20260922）

目的：验证 ① mimo-v2.6-pro 模型名可用 + 图像输入 OK；
     ② MiMo 对 `unograsp_obstruction_v11_json` prompt 的服从性
        （合法单 JSON、obstruction_paths 前置、schema 键齐、路径终点=free）；
     ③ token 预算（1024）是否足够。
方法：直接构造 Call 2 输入（UNOGrasp v11 system prompt + 真实 candidate table
     + 场景 png），绕开完整 pipeline（pipeline 集成已由静态测试覆盖）。
偏差声明：geometry_features=None（简化），与正式跑的 candidate table 有差，
     只影响提示丰富度，不影响服从性结论。
样本：3 个 blocker_first 失败（S2 类，来自 stage0_error_taxonomy.json）。
成本：3 次 MiMo 调用（boss 2026-09-22 当轮授权）。
"""
import base64
import json
import os
import sys
import time

FROZEN_REPO = "/home/public/zb/FreeGrasp_code-freegrasp_codex"
if FROZEN_REPO not in sys.path:
    sys.path.insert(0, FROZEN_REPO)
PATCH_DIR = os.path.dirname(os.path.abspath(__file__))
if PATCH_DIR not in sys.path:
    sys.path.insert(0, PATCH_DIR)

from unograsp_v11_prompt import UNOGRASP_OBSTRUCTION_V11_PROMPT, REQUIRED_SCHEMA_KEYS

MODEL = "mimo-v2.6-pro"
MAX_TOKENS = 2048
OUT = os.path.join(PATCH_DIR, "smoke3_report.json")

# (scene_id, query_object_id, annotation) —— blocker_first S2@bf 失败例（gt_no_query，真选错障碍）
CASES = [
    (1318, 1, "the small blue box"),
    (1733, 3, "the extra box"),
    (1784, 7, "the small plastic box underneath a knife"),
]


def main():
    from evaluate_mimo import create_mimo_client, patch_freegrasp_client
    from utils.utils import (
        _make_reasoning_messages,
        _call_reasoning_model,
        process_grasping_result,
        parse_visible_points,
        load_image_as_base64,
    )
    try:
        from utils.utils import _format_visible_object_candidates
    except ImportError:
        _format_visible_object_candidates = None

    client = create_mimo_client()
    patch_freegrasp_client(client)
    print(f"[smoke] client base_url = {client.base_url}", flush=True)

    results = []
    for scene, qid, annotation in CASES:
        scene_dir = os.path.join(FROZEN_REPO, "data", "output", "molmo_output", f"scene{scene}")
        id_txt = os.path.join(scene_dir, f"{scene}_id.txt")
        png = os.path.join(scene_dir, f"{scene}.png")
        rec = {"scene": scene, "query": qid, "annotation": annotation}
        try:
            parsed_data, _ = parse_visible_points(id_txt)
            b64 = load_image_as_base64(png)
            user_parts = [f"Requested target: {annotation}."]
            if _format_visible_object_candidates is not None:
                cand = _format_visible_object_candidates(
                    parsed_data, include_spatial=True, include_semantic=None,
                    semantic_labels=None, include_geometry=False,
                )
                user_parts.append(cand)
            else:
                user_parts.append("Visible IDs: " + ", ".join(str(p[0]) for p in parsed_data))
            user_text = "\n\n".join(user_parts)

            messages = _make_reasoning_messages(UNOGRASP_OBSTRUCTION_V11_PROMPT, user_text, b64)
            t0 = time.time()
            output = _call_reasoning_model(messages, MODEL, MAX_TOKENS)
            rec["latency_s"] = round(time.time() - t0, 1)
            rec["raw_len_chars"] = len(output)
            rec["raw_head"] = output[:400]

            result = process_grasping_result(output, annotation, "full")
            rec["selected_object_id"] = result.get("selected_object_id")
            rec["mask_query"] = result.get("mask_query")
            rj = result.get("reasoning_json") or {}
            rec["has_reasoning_json"] = bool(rj)
            paths = rj.get("obstruction_paths") if isinstance(rj, dict) else None
            rec["n_paths"] = len(paths) if isinstance(paths, list) else 0
            if isinstance(paths, list) and paths:
                steps = paths[0].get("steps", [])
                rec["path0_steps"] = [s.get("object_id") for s in steps]
                rec["path0_endpoint_free"] = bool(steps) and str(steps[-1].get("blocked_by_evidence", "")).startswith("none")
            rec["schema_keys_ok"] = all(k in output for k in REQUIRED_SCHEMA_KEYS)
            rec["parse_ok"] = result.get("selected_object_id") is not None
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        results.append(rec)
        print(json.dumps(rec, ensure_ascii=False)[:500], flush=True)

    n_ok = sum(1 for r in results if r.get("parse_ok"))
    summary = {
        "model": MODEL, "max_tokens": MAX_TOKENS, "n_cases": len(results),
        "n_parse_ok": n_ok,
        "n_paths_ok": sum(1 for r in results if r.get("n_paths", 0) >= 1),
        "n_schema_ok": sum(1 for r in results if r.get("schema_keys_ok")),
        "max_raw_chars": max((r.get("raw_len_chars", 0) for r in results), default=0),
    }
    json.dump({"summary": summary, "results": results}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("== summary ==", json.dumps(summary), flush=True)
    print("== written ==", OUT, flush=True)


if __name__ == "__main__":
    main()
