# -*- coding: utf-8 -*-
"""UNOGrasp×PathB1 融合 · 离线静态自测（unittest，零 API / 零 GPU / ¥0）

覆盖四层：
  T1 prompt 文本完整性：schema 键齐全、单 JSON 对象约束语句存在
  T2 模拟输出 → 冻结仓 process_grasping_result 真解析器：schema 兼容零改动
  T3 patch 行为：v10 被替换、其余变体透传、冻结仓文件 mtime 不变
  T4 模拟输出经 id_repair 判定路径：selected 在 visible ids 内时不触发修复
运行：/home/wyl/miniconda3/envs/freegrasp/bin/python test_unograsp_fusion_static.py
"""
import json
import os
import sys
import unittest

FROZEN_REPO = "/home/public/zb/FreeGrasp_code-freegrasp_codex"
if FROZEN_REPO not in sys.path:
    sys.path.insert(0, FROZEN_REPO)
PATCH_DIR = os.path.dirname(os.path.abspath(__file__))
if PATCH_DIR not in sys.path:
    sys.path.insert(0, PATCH_DIR)

from unograsp_v11_prompt import (  # noqa: E402
    UNOGRASP_OBSTRUCTION_V11_PROMPT,
    REQUIRED_SCHEMA_KEYS,
)

MOCK_OUTPUT = json.dumps({
    "obstruction_paths": [
        {"path_id": 1, "steps": [
            {"object_id": 2, "label": "knife",
             "blocked_by_evidence": "brown box rests on top of the knife handle"},
            {"object_id": 5, "label": "brown box",
             "blocked_by_evidence": "none - free"},
        ]},
    ],
    "target_resolution": {
        "target_object_id": 2, "target_label": "knife",
        "target_visibility": "visible_but_blocked",
        "target_evidence": "knife under brown box",
    },
    "first_grasp_planning": {
        "should_grasp_target_directly": False,
        "blocking_candidates": [
            {"object_id": 5, "label": "brown box",
             "blocker_evidence": "brown box rests on the knife", "priority": 1},
        ],
        "first_grasp_object_id": 5,
        "first_grasp_label": "brown box",
        "mask_query": "brown box",
        "decision_rule": "remove_direct_blocker",
    },
}, ensure_ascii=False)

VISIBLE_IDS = [1, 2, 3, 4, 5]


class T1PromptText(unittest.TestCase):
    def test_schema_keys_present(self):
        for key in REQUIRED_SCHEMA_KEYS:
            self.assertIn(key, UNOGRASP_OBSTRUCTION_V11_PROMPT)

    def test_single_json_object_constraint(self):
        self.assertIn("Return exactly one compact JSON object", UNOGRASP_OBSTRUCTION_V11_PROMPT)
        self.assertIn("Return ONLY the JSON object", UNOGRASP_OBSTRUCTION_V11_PROMPT)

    def test_paths_first_and_endpoint_rules(self):
        self.assertIn("obstruction_paths comes FIRST", UNOGRASP_OBSTRUCTION_V11_PROMPT)
        self.assertIn("must equal the endpoint", UNOGRASP_OBSTRUCTION_V11_PROMPT)
        self.assertIn("none - free", UNOGRASP_OBSTRUCTION_V11_PROMPT)

    def test_never_return_target_rule_kept(self):
        # v10 的硬规则必须保留（blocker_first 语义不回退）
        self.assertIn("Never return the requested target as the first grasp",
                      UNOGRASP_OBSTRUCTION_V11_PROMPT)


class T2ParserCompat(unittest.TestCase):
    """模拟输出过冻结仓真解析器 process_grasping_result（full 协议）。"""

    def test_mock_output_parses(self):
        from utils.utils import process_grasping_result

        result = process_grasping_result(MOCK_OUTPUT, "the knife", "full")
        self.assertEqual(result.get("selected_object_id"), 5)
        self.assertEqual(result.get("mask_query"), "brown box")
        rj = result.get("reasoning_json") or {}
        self.assertIn("obstruction_paths", rj)
        self.assertEqual(rj["obstruction_paths"][0]["steps"][-1]["object_id"], 5)

    def test_mock_output_survives_id_repair_gate(self):
        # selected=5 ∈ visible ids → 不应触发 id_repair（无需下游修复）
        from utils.utils import process_grasping_result, _needs_id_repair

        result = process_grasping_result(MOCK_OUTPUT, "the knife", "full")
        self.assertFalse(_needs_id_repair(result, VISIBLE_IDS))


class T3PatchBehavior(unittest.TestCase):
    def test_patch_replaces_only_v10(self):
        from run_arm_f_unograsp import install_prompt_patch
        import utils.utils as um

        original = um.get_reasoning_system_prompt
        try:
            install_prompt_patch()
            self.assertEqual(um.get_reasoning_system_prompt("hard_blocker_first_v10_json"),
                             UNOGRASP_OBSTRUCTION_V11_PROMPT)
            for variant in ("direct_target", "target_first_evidence_gate_v9_json",
                            "legacy", "json", "hard_occlusion"):
                self.assertEqual(um.get_reasoning_system_prompt(variant), original(variant))
        finally:
            um.get_reasoning_system_prompt = original

    def test_frozen_utils_py_untouched(self):
        # 冻结仓文件 mtime 基线：本测试运行前后不变（fail-closed）
        path = os.path.join(FROZEN_REPO, "utils", "utils.py")
        m0 = os.path.getmtime(path)
        from run_arm_f_unograsp import install_prompt_patch
        import utils.utils as um

        original = um.get_reasoning_system_prompt
        try:
            install_prompt_patch()
        finally:
            um.get_reasoning_system_prompt = original
        self.assertEqual(os.path.getmtime(path), m0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
