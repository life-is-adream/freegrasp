# -*- coding: utf-8 -*-
"""UNOGrasp×PathB1 融合 · 臂 F 运行 wrapper（RDR-20260922-UNOGRASP-PATHB1-FUSION）

职责（单一、可审计）：
  1. 把冻结仓加入 sys.path（冻结仓零改动）。
  2. monkey-patch `utils.utils.get_reasoning_system_prompt`：
     仅当 Call 2 被分派到 `hard_blocker_first_v10_json`（即 route_mode=blocker_first）
     时返回 `unograsp_obstruction_v11_json` prompt；其余变体全部透传原函数。
     direct / target_first 路由不受影响（严格单变量）。
  3. 调用冻结仓 `evaluate_mimo_full_protocol.run_full_protocol(parse_args())`，
     全部 CLI 参数与臂 E 一致（在启动命令里传，见 runbook）。

用法（在冻结仓根目录执行，环境 conda env `freegrasp`）：
  cd /home/public/zb/FreeGrasp_code-freegrasp_codex
  /home/wyl/miniconda3/envs/freegrasp/bin/python \
      /home/public/zb/freegrasp_unograsp_fusion_20260922_v1/run_arm_f_unograsp.py \
      --model-name mimo-v2.6-pro --max-output-tokens 1024 \
      --subset-policy eghc_consensus_v1 --segmentation-variant eghc_mask_repair_v1 \
      --routing-variant evidence_gated_cascade --feature-source deployable_predicted \
      --route-guard-variant eghc_consensus_v1 --splits 0 --reuse-molmo \
      --run-root /home/public/zb/freegrasp_unograsp_fusion_20260922_v1/arm_F_split0 \
      [--fixed-sample-ids ...] [--max-total-rows 3]

审计：本文件只 patch prompt 选择逻辑，不触碰评测/统计/缓存；patch 生效信息
打印到 stdout 且写入 run_root/PATCH_RECORD.json。
"""
import json
import os
import sys
import time

FROZEN_REPO = "/home/public/zb/FreeGrasp_code-freegrasp_codex"
PATCH_DIR = os.path.dirname(os.path.abspath(__file__))
if FROZEN_REPO not in sys.path:
    sys.path.insert(0, FROZEN_REPO)

from unograsp_v11_prompt import (  # noqa: E402
    UNOGRASP_OBSTRUCTION_V11_PROMPT,
    REQUIRED_SCHEMA_KEYS,
)

TARGET_VARIANT = "hard_blocker_first_v10_json"  # v10：EGHC3_v3 blocker_first 的 Call 2 prompt
REPLACEMENT_VARIANT = "unograsp_obstruction_v11_json"

PATCH_RECORD = {
    "wrapper": os.path.abspath(__file__),
    "frozen_repo": FROZEN_REPO,
    "target_variant": TARGET_VARIANT,
    "replacement_variant": REPLACEMENT_VARIANT,
    "patched_at_epoch": time.time(),
    "frozen_get_reasoning_system_prompt_unchanged": True,
}


def install_prompt_patch():
    """Patch utils.utils.get_reasoning_system_prompt（模块属性替换，冻结仓文件零改动）。

    utils.py 内部以模块全局名直接调用该函数（utils.py:2696 附近
    `system_prompt = get_reasoning_system_prompt(active_prompt_variant)`），
    因此替换 utils.utils 命名空间中的绑定即可生效。
    """
    import utils.utils as um

    original = um.get_reasoning_system_prompt

    def patched_get_reasoning_system_prompt(prompt_variant="legacy"):
        if prompt_variant == TARGET_VARIANT:
            return UNOGRASP_OBSTRUCTION_V11_PROMPT
        return original(prompt_variant)

    patched_get_reasoning_system_prompt.__original__ = original
    um.get_reasoning_system_prompt = patched_get_reasoning_system_prompt

    # 自证：patch 后行为
    assert um.get_reasoning_system_prompt(TARGET_VARIANT) == UNOGRASP_OBSTRUCTION_V11_PROMPT
    assert um.get_reasoning_system_prompt("direct_target") == original("direct_target")
    assert um.get_reasoning_system_prompt("target_first_evidence_gate_v9_json") == (
        original("target_first_evidence_gate_v9_json")
    )
    assert um.get_reasoning_system_prompt("legacy") == original("legacy")
    PATCH_RECORD["self_check"] = "PASS"
    return original


def write_patch_record(run_root):
    try:
        os.makedirs(run_root, exist_ok=True)
        with open(os.path.join(run_root, "PATCH_RECORD.json"), "w", encoding="utf-8") as f:
            json.dump(PATCH_RECORD, f, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"[arm-F wrapper] WARN: cannot write PATCH_RECORD: {exc}", flush=True)


def main(argv=None):
    import evaluate_mimo_full_protocol as emp

    args = emp.parse_args()
    # patch 必须在 run_full_protocol 之前安装
    install_prompt_patch()
    print(f"[arm-F wrapper] prompt patch installed: {TARGET_VARIANT} -> {REPLACEMENT_VARIANT}", flush=True)
    print(f"[arm-F wrapper] args: model={args.model_name} tokens={args.max_output_tokens} "
          f"policy={args.subset_policy} run_root={args.run_root}", flush=True)
    write_patch_record(args.run_root)
    for key in REQUIRED_SCHEMA_KEYS:
        assert key in UNOGRASP_OBSTRUCTION_V11_PROMPT, f"schema key missing: {key}"
    return emp.run_full_protocol(args)


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
