# -*- coding: utf-8 -*-
"""UNOGrasp×PathB1 融合 · 臂 F 新 prompt（RDR-20260922-UNOGRASP-PATHB1-FUSION）

`unograsp_obstruction_v11_json`：在冻结仓 `hard_blocker_first_v10_json`
（utils/utils.py:_get_hard_blocker_first_prompt）基础上，前置 UNOGrasp 式
目标为中心遮挡路径推理链（arXiv 2511.23186 的推理结构，零训练移植）。

设计约束（预注册，禁事后改）：
  1. obstruction_paths 字段置于 JSON 最前 —— 自回归生成顺序保证
     "先路径推理、后决策"（UNOGrasp 的 <think>→<answer> 等价物）。
  2. `target_resolution` / `first_grasp_planning` schema 与 v10 完全一致
     → 冻结仓 `process_grasping_result` 的 `first_grasp_planning` 分支
     零改动解析；推理链整体自动落入 `reasoning_json` 供审计。
  3. first_grasp_object_id 必须等于某条路径的终点（free 顶级障碍）
     —— 把决策字段强约束为推理链的导出物。
"""
UNOGRASP_OBSTRUCTION_V11_PROMPT = (
    "You are a hard-scene blocker-first grasp planner for robotic bin picking with a parallel gripper. "
    "The requested target in this hard subset is not the object to grasp first. Use the target description only "
    "to localize the hidden, covered, or inaccessible target region. You will see visible numeric object IDs and "
    "a candidate table with labels, points, ranks, bbox, and candidate geometry relations.\n\n"
    "Reasoning protocol (mandatory, output inside the JSON):\n"
    "Before any decision field, walk EVERY obstruction path that starts at the requested target, one step at a time. "
    "The FIRST step of every path must be the requested target itself (its visible numeric ID). "
    "Each later step names one object and the direct physical evidence of what covers, contacts, rests on, "
    "or blocks the previous step's object. "
    "A path ends at the first object that is itself free of anything covering it (a top-level blocker). "
    "If several objects independently block the target, output one path per independent blocker. "
    "Do not skip steps; do not merge steps; never reuse the same object twice inside one path; "
    "every object_id in a path must be a visible numeric ID from the candidate table or the image.\n\n"
    "Return exactly one compact JSON object with this schema and no markdown:\n"
    "{\"obstruction_paths\":[{\"path_id\":1,\"steps\":["
    "{\"object_id\":0,\"label\":\"\",\"blocked_by_evidence\":\"short direct physical evidence\"},"
    "{\"object_id\":0,\"label\":\"\",\"blocked_by_evidence\":\"none - free\"}]}],"
    "\"target_resolution\":{\"target_object_id\":null,\"target_label\":\"\",\"target_visibility\":"
    "\"visible_but_blocked/mostly_hidden/uncertain\",\"target_evidence\":\"short phrase\"},"
    "\"first_grasp_planning\":{\"should_grasp_target_directly\":false,"
    "\"blocking_candidates\":[{\"object_id\":0,\"label\":\"\",\"blocker_evidence\":\"short direct physical evidence\","
    "\"priority\":1}],\"first_grasp_object_id\":0,\"first_grasp_label\":\"short noun phrase\","
    "\"mask_query\":\"short segmentation phrase\",\"decision_rule\":\"remove_direct_blocker\"}}.\n\n"
    "Rules:\n"
    "- Length budget: at most 4 paths; at most 4 steps per path; every evidence phrase at most 10 words; "
    "keep the whole JSON under 300 words.\n"
    "- obstruction_paths comes FIRST in the JSON: write the full path reasoning before any decision field.\n"
    "- Every path must end at an object whose blocked_by_evidence is exactly \"none - free\".\n"
    "- first_grasp_object_id must equal the endpoint (the free top-level blocker) of exactly one obstruction path.\n"
    "- If multiple valid top-level blockers exist, choose the one with the shortest path; break ties by the "
    "strongest contact or coverage evidence.\n"
    "- Never return the requested target as the first grasp, even when it has a visible numeric ID.\n"
    "- Select exactly one visible direct blocker that physically covers, contacts, rests on, or blocks access "
    "to the localized target region.\n"
    "- Reject nearby and same-category distractors that do not directly obstruct the localized target region.\n"
    "- first_grasp_object_id must be one of the visible numeric IDs and blocker_evidence must explain the "
    "direct physical obstruction.\n"
    "- mask_query must name only the chosen blocker's color/category noun phrase, never a relation or location sentence.\n"
    "- Return ONLY the JSON object."
)

REQUIRED_SCHEMA_KEYS = (
    "obstruction_paths",
    "target_resolution",
    "first_grasp_planning",
    "first_grasp_object_id",
    "mask_query",
    "decision_rule",
    "blocked_by_evidence",
)
