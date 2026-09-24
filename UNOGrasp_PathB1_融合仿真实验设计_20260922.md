# UNOGrasp × PathB1 融合仿真实验设计（2026-09-22）

> **RDR-20260922-UNOGRASP-PATHB1-FUSION**（待登记）
> 对象：当前最高可部署方法 **EGHC3_v3**（split-0 RSR 62.00% / SSR 57.33%）之上的下一代 blocker_first 改造。
> 前身：PathB_B1（`pathb_blk_arb_v1`）已于 2026-09-21 判 **null closure**（ΔRSR +0.00pp）。
> 本设计 = 把 **UNOGrasp（arXiv 2511.23186, CVPR 2026, FBK）的目标为中心遮挡图推理**移植进 EGHC 路由层。
> 纪律：RDR / 零 GT 信息注入 / 付费须当轮授权 / fail-closed / 负结果 = 有效 closure。

---

## 0. 结论先行

1. **可行性判定：有戏，且切入点与此前诊断结论不同。** 2026-09-22 失败模式诊断曾判定"UNOGrasp 式感知/掩码改进打不到主导失败模式"——那只针对 UNOGrasp 的**感知脚手架**（mask 质量/遮挡比例线索）。本次 Stage 0 细化分析显示：**86% 的失败（98/114）发生在遮挡语境，且形态与 UNOGrasp 训练的推理结构精确对齐**。UNOGrasp 的**推理结构**（遮挡路径逐步推理 + SoM ID 锚定）恰好打的是"该先挪谁/选错实例"这两个主导失败形态。
2. **建议主实验 = Stage 1（C1+C2，prompt 层融合，MiMo router 重跑）**：单变量对照，代理终点 `selected_gt_id ∈ gt_ids`（在本数据集上与 RSR 完全等价，见 §2.4），n=300 配对，McNemar exact + bootstrap。Power 保障区间：真实增益 ≥4pp 时 power ≥0.83。
3. **预期（诚实估计 [I]）**：中位预期 **+5pp**（RSR 62% → ~67%），源于修复 S1 型失败的 1/3 与 S2 型的 10–20%。若真实效应 <4pp，实验判 null 并给出等效性上界——依然是有效 closure，与 B1 同等待遇归档。
4. **成本**：Stage 1 需重跑 300 次 MiMo router 调用（改 prompt → 缓存失效，必须新调用）= **付费，须 boss 当轮授权**；精确报价走 `request_budget.py`（fail-closed，不报记忆价）。**服务器 GPU 已恢复（2026-09-22 实查）**：RTX 3090 24GB（已用 15.9GB）+ RTX 3080 Ti 12GB（空闲）+ 2×GTX 1080 Ti，驱动 580.178.04 正常 ⇒ **C3（UNOGrasp checkpoint 推理）在服务器 3090 上可行**，不再依赖本地 GPU；此前 2026-09-21 记录的"驱动损坏"阻塞已解除。

---

## 1. UNOGrasp 是什么（调研结论，一手出处）

| 项 | 内容 | 出处 |
|---|---|---|
| 论文 | *Obstruction reasoning for robotic grasping*，Jiao et al., FBK + Univ. Trento | arXiv 2511.23186（2025-11-28），CVPR 2026 |
| 与本项目关系 | **同源升级版**：其 related work 明确引用 FreeGrasp（arXiv 2503.13082 = 本项目原论文）为 preliminary work（"exploiting Molmo for visual grounding and prompting GPT-4o to reason whether to clear obstructing objects first"）；其官方代码仓库名即 `tev-fbk/freegrasp_code` | [E] arXiv HTML related work 节 |
| 方法核心 | 目标为中心遮挡图 `G_t`：有向边 `(o_i, o_j)` = o_i 被 o_j 遮挡；推理沿**遮挡路径**逐步走（目标→…→顶级障碍），输出 **F(o_t) = top-level accessible obstructors**（下一步该移除的集合） | [E] arXiv §3 |
| 视觉锚定 | SoM 唯一数字标记 + 每实例质心 `(x,y)`；推理链每步锚定**物理相邻（接触）**邻居 | [E] arXiv §3 + HF 模型卡 |
| 训练 | Qwen2.5-VL-3B：SFT（2ep，遮挡线索写进推理链）→ RFT（GRPO 1ep，reward = λ_fmt·格式 + IoU(F_pred, F_gt) 集合级奖励） | [E] arXiv 附录 C.1 |
| 开源资产 | **UnoBench 数据集 + 两个 checkpoint 已发布**（`FBK-TeV/UnoGrasp-Ratio-RL-IoU-som-small` / `-nlp-small`，HuggingFace）；推理/评测脚本在 GitHub `tev-fbk/UnoGrasp`（2026-05-31 发布） | [E] GitHub README + HF 模型卡 |
| 关键数字 | 合成 Oracle SR-F1：Easy 83.3 / Medium 69.1 / Hard 54.5；真机 UR5e 30 场景均值 50%（Gemini 40%）；**ratio 线索在 Hard +5.8pp；IoU 奖励在 Hard +4.4pp** | [E] boss 拆解 + arXiv 表 |

**对本项目最重要的结构对齐 [E]**：UNOGrasp 的 `F(o_t)`（top-level accessible obstructors）与本项目 RSR 的 `acceptable_gt_ids`（首抓集合 = 剪枝遮挡图自目标可达的叶节点，Q0 语义门结论）**是同一个数学对象的两侧**——本项目评测的"正确首动作"就是 UNOGrasp 训练模型直接预测的量。融合不是概念嫁接，是**同构对接**。

---

## 2. Stage 0 定性分析结果（本次新增，¥0，本地）

> 脚本：`unograsp_pathb1_stage0{,b,c}.py`；数据：`output/failure_mode_diag/EGHC3_v3/split_0/details_*.json`（300 例）。
> 分类判据预注册于脚本 docstring，跑前写死。

### 2.1 失败错误形态分类（n=114）

| 形态 | 定义 | n | 占比 | UNOGrasp 组件对症？ |
|---|---|---|---|---|
| **S1 早抓目标** | `selected == query 目标` 但目标 ∉ `gt_ids` → 该先移除障碍却直接抓了被挡目标 | **45** | 39.5% | ✅ 遮挡路径推理（核心场景） |
| **S2 选错非目标** | `selected != query` 且 router 报告了 `blocking_relations`（有遮挡证据）却选了 gt 集合外物体 | **53** | 46.5% | ⚠️ 部分（推理链约束 + SoM 锚定） |
| S3 无关物体 | 无任何遮挡证据的纯 grounding 错误 | 11 | 9.6% | ❌ 遮挡推理救不了 |
| S4 无效输出 | `selected_gt_id ≤ 0` 或 parse 失败 | 5 | 4.4% | ❌ 工程问题 |

**S1+S2 = 98/114 = 86%** —— 失败主导形态在遮挡语境中。

### 2.2 按 route_mode 交叉（G0a 架构核实后修订）

| route | 失败数 | S1 | S2 | S4 | 解读 |
|---|---|---|---|---|---|
| blocker_first | 60 | 14 | **43** | 3 | Call 2（`hard_blocker_first_v10_json`）选错 → **靶点 1** |
| direct | 40 | **23** | 5 | 11 | **靶点 2**：router 判 direct 后**不调 Call 2**（selected 直取 router 的 target_object_id），23 例是 router 单调用链的遮挡/可达性误判 |
| target_first | 14 | 8 | 5 | 1 | Call 2（`target_first_evidence_gate_v9_json`），第一版不动 |

**G0a 架构核实 [E]（服务器实查）**：EGHC3_v3 的 `evidence_gated_cascade` 为两层 VLM 调用——Call 1 router（`eghc_router_prompt()`）输出路由 JSON；`route_mode=direct` 且 target 有效时**直接取 router 的 target_object_id**（Call 2 不调用）；否则 Call 2 按 route 分派 prompt（`direct_target` / `target_first_evidence_gate_v9_json` / `hard_blocker_first_v10_json`）。**顶层 `prompt_variant=legacy` 在此路径下不被使用**。Call 2 输入已含 candidate table（labels/points/ranks/bbox/pairwise relations）——UNOGrasp 推理链所需证据素材大半已在。

**G1 核实 [E]（2026-09-22 服务器实查，`g1_s1direct_gt_occlusion.py`）**：`gt_ids` 源头 = 数据集 `groundTruthObjIds`（FreeGraspData 自带 GT 抓取序列 +1，`reasoning_eval.py:326`），与 npz `occlusion` 像素图为独立数据源（后者 0/1 二值异常、语义未定，**弃用作判据**）。**结论：S1-direct 23 例全部是 router 与 GT 抓取序列的矛盾（router 误判"可直抓"），无 GT 标注异常，此前 scene 3576 疑点解除 → 23 例全部属可修类**。

### 2.2b 可修靶点结构（修订后）

| 靶点 | 位置 | 覆盖失败 | 改动 |
|---|---|---|---|
| **靶点 1** | Call 2 blocker_first prompt | S2 43 + S1 14 = **57 例** | 新 prompt 变体，输出 schema 兼容 `process_grasping_result`（`first_grasp_planning` 结构不变，推理链字段前置） |
| **靶点 2** | Call 1 router / direct 复核 | S1-direct **23 例** | router prompt 或 guard 逻辑（第二变体，主实验有效后追加） |
| 不可修 | — | S3 11 + S4 5 | grounding 错误/无效输出，遮挡推理无关 |

理论上限（靶点 1 全修）：(186+57)/300 = **81.0%**；现实预期（见 §4.4）。

### 2.3 失败与语言特征**无关**（重要 null 结果）

| 信号 | 失败组 | 成功组 | 判定 |
|---|---|---|---|
| 选中物体与查询同类率 | 53.5% | 59.7% | 无差异 |
| annotation 含空间关系短语率 | 48.2% | 52.7% | 无差异 |

→ 失败不是"难语言"驱动的，是 router **单步直觉式输出**（一次性 JSON：candidates+blocking+route）在逐场景 grounding 上的脆弱性。这与 UNOGrasp 论文对通用 VLM 的诊断（Qwen 过度推理/幻觉、Gemini 多路径过早终止）**同构**：没有专门的遮挡路径推理训练/结构，通用模型在这类任务上系统性弱。

### 2.4 代理终点的合法性 [E]

诊断（2026-09-22）已证明：成功 186/186 全部 `selected_gt_id ∈ gt_ids`，失败 0/114 满足。⇒ 在本数据集上 **`selected_gt_id ∈ gt_ids` 与 `rsr_correct` 完全等价**。因此**只重跑 router 决策层**（不重跑 mask/LangSAM/SAM/仿真执行）即可无损预测 RSR——主实验成本从"全 pipeline"降到"router 单调用 ×300"。

### 2.5 理论修复空间（G1 后修订：23 例 S1-direct 确认为可修类）

| 情形 | 靶点 1（57 例） | 靶点 2（23 例） | RSR 上界 |
|---|---|---|---|
| 全修（不可能达到） | 57 | 23 | (186+80)/300 = **88.7%** |
| UNOGrasp 推理链现实水平 | ~12–20 | ~5–8 | ~(186+20)/300 ≈ **68.7%** |
| 悲观 | 6 | 2 | 64.7%（+2.7pp） |

主实验只打靶点 1：预期 +2~7pp（中位 +4pp，与 power 保障区间 4–5pp 匹配）；靶点 2 为追加项。注意 UNOGrasp Hard SR-F1 ≈55% 是"零证据推断"任务的水平；EGHC 的 Call 2 已有 candidate table + guard 双保险，任务更易。

---

## 3. 融合方案设计

### 3.1 与 PathB1 的关系（为什么这是"融合"而非新方法）

B1 的位置 = blocker_first 路由的**仲裁层放宽**（identity_score 置信度替换），已 null。其教训（Stage A 机制证据）：**改变行为 ≠ 改变成功率**——放宽仲裁改变了 9 个样本的路由，但没改变决策依据的质量。

本融合**占据 B1 的生态位**（blocker_first 路径的下一代改造），但换掉的是**证据结构本身**：

```
B1：     仲裁依据 = 置信度阈值（同一份单步直觉，放宽门槛）
融合：   仲裁依据 = 遮挡路径推理链（目标 → 谁挡它 → 谁挡那个 → 顶级障碍）
                      ↑ UNOGrasp 的推理结构，每步锚定接触邻居与视觉证据
```

### 3.2 三个组件（按成本递增）

| 组件 | 内容 | 靶子 | 训练 | 成本 |
|---|---|---|---|---|
| **C1 推理链 prompt（靶点 1）** | 新 Call 2 prompt 变体 `unograsp_obstruction_v11_json`：在 `hard_blocker_first_v10_json` 基础上嵌入 UNOGrasp 遮挡路径推理结构——**JSON 内前置 `obstruction_paths` 字段**（自回归生成顺序：路径推理先于决策字段），每步引用可见 ID + 接触/覆盖证据，多路径分别走完再汇合；`first_grasp_planning` schema 保持不变 → `process_grasping_result` **零改动**（推理链自动存入 `reasoning_json` 供审计） | S1 14 + S2 43（blocker_first Call 2） | 零训练 | MiMo 调用（付费，小额） |
| ~~C2 SoM 锚定~~ | **取消**：G0a 实查确认 FreeGraspData 图像自带 ID 数字标注（legacy/router prompt 自述 "labeled all objects id in the image"），SoM 锚定已存在于基线 | — | — | ¥0 |
| **C2'（第二变体，暂缓）** | 靶点 2：router 层 direct 复核（修 S1-direct 23 例）——主实验有效后追加，需改 Call 1 prompt 或 guard | S1-direct 23 | 零训练 | 同上 |
| **C3 独立 obstruction 仲裁器** | 下载开源 `UnoGrasp-Ratio-RL-IoU-som-small`（Qwen2.5-VL-3B），作为独立遮挡复核器 | S1-direct 23 例（漏检遮挡） | 零训练（用现成 checkpoint） | **服务器 RTX 3090（24GB）推理可行**（bf16 3B-VL ≈ 7GB，与现有 15.9GB 占用可共存；3080 Ti 12GB 备选）；实验一律在服务器 GPU 上跑。⚠️ **获取通路硬约束**：checkpoint 在 HuggingFace，本环境与服务器均无法直连 HF（2026-09-22 实查）→ 需 boss 本机手动下载（约 7GB）后上传服务器 |

### 3.3 主实验 = 靶点 1 纯单变量（修订后）

~~C1+C2 打包~~ → **C2 已取消**（SoM 锚定已存在于基线图像）。主实验变为**严格单变量**：唯一差异 = blocker_first 的 Call 2 prompt（`hard_blocker_first_v10_json` → `unograsp_obstruction_v11_json`），图像/输入表/解析器/guard/retry 全冻结。`direct`/`target_first` 路由的 Call 2 prompt 不动（分别 0 次额外调用/同表）。

**调用次数核算 [E]**：Call 1 router 每样本 1 次（两臂相同）；Call 2 仅在非 direct 或 direct 无效时调用——blocker_first 150 样本必调，target_first 18 样本必调，direct 132 样本大部分不调（router 直取）。**臂 F 与臂 E 的差异调用仅发生在 route_mode=blocker_first 的样本上（约 141–150 次/臂）**——但两臂都要跑满 300 样本（router 调用不可省）。总新增 MiMo 调用 ≈ 300（臂 E）+ 300（臂 F）+ 少量 smoke。

---

## 4. 实验设计（预注册）

### 4.1 臂设计

| 配置项 | 臂 E（对照） | 臂 F（融合） |
|---|---|---|
| `model` | **mimo-v2.6-pro**（boss 2026-09-22 指定升级；两臂一致，仍为单变量） | 同 |
| Call 1 router prompt | `eghc_router_prompt()` | 同（不动） |
| Call 2 blocker_first prompt | `hard_blocker_first_v10_json` | **`unograsp_obstruction_v11_json` ← 唯一变量** |
| Call 2 direct/target_first prompt | `direct_target` / `target_first_evidence_gate_v9_json` | 同（不动） |
| 图像输入 | FreeGraspData 自带 ID 标注图 | 同（同一份文件） |
| `route_guard_variant` | eghc_consensus_v1 | 同 |
| `routing_variant` | evidence_gated_cascade | 同 |
| `segmentation_variant` | eghc_mask_repair_v1 | 同 |
| `mimo_retry_mode` | **id_repair**（G0b 实证 EGHC3_v3 实际值，两臂一致贴部署现状） | **id_repair**（同） |
| `molmo_cache_policy` | reuse_existing（`cache_sha256=705d088…` 同一份） | 同 |
| `splits` | `["0"]` | `["0"]` |
| 其余全部 | 冻结 | 冻结 |

⚠️ 对照臂 E 是 EGHC3_v3 同配置复刻（retry 口径与 G0b 实证一致：`id_repair`），兼作 temperature=0 抖动带的 sanity 对照。**⚠️ 模型升级注意 [E]**：历史 62.00% 由 mimo-v2.5 跑出，**与本实验（v2.6-pro）跨模型不可直接比**；臂 E（v2.6-pro + 原配置）承担对照职责——若 v2.6-pro 本身有增益/回退，会体现在臂 E 与历史值的差异上，作为独立观察报告，不与臂 F 混算。主终点恒为**两臂配对差**。

### 4.2 终点

- **主终点（代理）**：`selected_gt_id ∈ gt_ids` 的样本比例差（ΔRSR_proxy），合法性依据 §2.4。
- **次终点**：`route_mode` 分布变化、S1/S2 形态迁移（用本次 Stage 0 分类器复判臂 F 失败形态）。
- **探索性（不升格主终点）**：blocker_first 子集条件分析；S1-direct 23 例条件分析。

### 4.3 统计方法（沿用 B1 框架）

- 配对键 `(scene_id, object_id)`，McNemar exact + 配对 bootstrap（2000 次，seed 20260822）。
- 判定规则（先写死）：ΔRSR_proxy 95% CI **下界 > 0** 判有效；**CI 跨 0 → null closure**，不为追分重跑、不改终点、不事后挑子集。
- 叠加等效性上界报告（discordant 数 → CI 半宽）。

### 4.4 Power / MDE（脚本精确值：`unograsp_pathb1_power.py`）

| 真实增益 | 副作用 p10 | 期望不一致对 | power（McNemar exact, α=0.05） |
|---|---|---|---|
| 3pp | 1% | 15 | 0.65（不足） |
| **4pp** | 1% | 18 | **0.83** |
| **5pp** | 1% | 21 | **0.93** |
| 5pp | 2% | 27 | 0.76（边缘） |
| 8pp | 2% | 36 | 0.99 |

⇒ 本设计可靠检测的最小真实效应 ≈ **4–5pp**。机制预期（§2.5 中位 +5pp）落在保障区间内；若真实效应 <4pp 判 null。

---

## 5. 机械门（跑之前必过）

| 门 | 内容 | 状态 |
|---|---|---|
| **G0a** | 服务器核实 router prompt 构造代码：输入模态（是否传图）、prompt 模板位置、`prompt_variant` 参数机制、C2 叠图注入点；核实 router-only 重放是否绕开 SAM（GPU 已恢复，即便需要 SAM 也不再阻塞，只影响成本） | ⬜ 待执行（只读 SSH，¥0） |
| **G0b** | 核实 EGHC3_v3 manifest 的 `mimo_retry_mode` 实际值（B1 设计遗留疑点） | ⬜ |
| **G0c** | `--help` 核对 `evaluate_mimo_full_protocol.py` 全部参数名（纪律：禁凭记忆写命令） | ⬜ |
| **G1** | ✅ **已完成（2026-09-22）**：S1-direct 23 例全部为 router 与 GT 抓取序列（`groundTruthObjIds`）的矛盾——可修类，无 GT 标注异常；npz occlusion 图语义未定弃用。见 §2.2 | ✅ |
| **G2** | 预算：`request_budget.py` 报价（臂 E 300 + 臂 F 300 + smoke ≈ 3-9 次 MiMo 调用） | ⬜ |
| **G3** | ~~叠图抽检~~ **取消**（C2 已取消，图像自带 ID 标注） | ✅ |

---

## 6. 成本与授权请求

| 项 | 金额 | 授权状态 |
|---|---|---|
| Stage 0（已完成） | ¥0 | ✅ 已执行 |
| G0/G1 机械门（只读 SSH + 本地） | ¥0 | 待 boss 放行执行 |
| Stage 1 臂 F（300 次 MiMo router 调用 + 300 次 C2 叠图本地生成） | **待 request_budget.py 精确报价** | ⬜ **须 boss 当轮授权** |
| Stage 1 臂 E | 若 G0a 确认可复用已有 EGHC3_v3 details（retry 口径差异需评估）→ ¥0；否则同臂 F | ⬜ |
| Stage 2（C3 checkpoint，服务器 3090 推理 300 样本） | checkpoint 下载（HuggingFace，免费）+ 推理电耗，无 API 费 | ⬜ 可行，随 Stage 1 结果排期 |

---

## 7. 风险与限制

1. **打包变量归因限制**（§3.3）：C1/C2 单独贡献不可分——预注册声明，不事后拆。
2. **代理终点风险**：等价性在 split-0 / EGHC3_v3 上证明；若臂 F 的 prompt 改变了下游行为路径（如 guard 逻辑对推理链格式的兼容性），代理可能失真 → G0a 核实 + 臂 F 抽 10 例跑全 pipeline 验证一致性。
3. **MiMo 对长推理链的服从性未知**：UNOGrasp 结构基于 Qwen2.5-VL 后训练；MiMo 零样本服从 think/answer 格式的比率待 smoke（≤3 例，纪律：最多 3 轮 smoke）。若服从性差 → prompt 工程迭代受"1 假设/轮"约束，最多 3 版。
4. **S1-direct 23 例可能含 GT 问题**（G1）：若核实为标注问题，实际上限降 2–4pp，预期区间下调。
5. **服务器 GPU 已恢复**（2026-09-22 实查：3090 24GB + 3080 Ti 12GB，驱动 580.178.04），Stage 1 与 C3 均不受阻塞；⚠️ 3090 已有 15.9GB 占用（他人/他任务），C3 排期前须确认显存共享可行或错峰。
6. 本设计与论文的关系：**遵 2026-09-22 boss 口径——UNOGrasp 不进入论文定稿定位、非本项目方法**。本线定位为**仿真性能改进实验**；若有效，论文写法为"外部推理结构启发的 prompt 改造"（EGHC 消融矩阵新条目 `UNOGRASP_FUSION_v1`），定稿引用前须 boss 重新裁决；若 null，作为 B1 的姊妹负结果（"证据结构升级亦无效"将是有力 limitation 证据）。

---

## 8. 执行顺序

```
G0a/b/c（¥0 只读核实）→ G1（GT 遮挡核实）→ G3（叠图抽检）
  → [boss 授权 + G2 报价] → 3 例 smoke（prompt 服从性）
  → 臂 F 重跑 300 → 配对分析（McNemar + bootstrap + 形态迁移复判）
  → 有效 → C1-only 消融 + 全 pipeline 一致性抽检 + C3（服务器 3090）上靶 S1-direct
  → null → closure 归档（与 B1 并列）
```

---

## 9. 证据与工具索引

| 项 | 位置 |
|---|---|
| UNOGrasp 论文 | https://arxiv.org/abs/2511.23186 ｜ 项目页 https://tev-fbk.github.io/UnoGrasp/ |
| 开源 checkpoint | https://huggingface.co/FBK-TeV/UnoGrasp-Ratio-RL-IoU-som-small |
| 代码仓库 | https://github.com/tev-fbk/UnoGrasp（注意其官方仓库名 `freegrasp_code` 与本项目原论文的渊源） |
| Stage 0 脚本 | `E:/FreeGrasp_code-main/unograsp_pathb1_stage0{,b,c}.py` |
| Power 脚本 | `E:/FreeGrasp_code-main/unograsp_pathb1_power.py` |
| 失败分类明细 | `E:/FreeGrasp_code-main/output/failure_mode_diag/stage0_error_taxonomy.json`（114 例逐条） |
| 前置文档 | `EGHC3_v3_失败模式诊断_20260922.md`、`PathB_B1_消融实验设计_20260921.md`、`PathB_B1_StageA_配对分析结果_20260921.md` |
