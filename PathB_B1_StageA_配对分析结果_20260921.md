# PathB B1 消融 · Stage A 配对再分析结果（2026-09-21）

> 零成本、离线、只读。不调 API、不用 GPU、未在服务器落盘（脚本经 stdin 执行）。
> 数据：split-0，n=300，两臂样本 `(scene_id, object_id)` **100% 配对成功**（common=300，only_in_A=0，only_in_B=0）。
> 基线 A = `EGHC3_v3`（当前最高可部署）｜处理 B = `PATHB_FULL_B1_blk_arb`（B1 = `pathb_blk_arb_v1`）
> 统计：配对 McNemar exact + 配对 bootstrap（2000 次重采样，seed 20260822，簇 = 每样本）

---

## 1. 结论先行

**B1 对 EGHC3_v3 无显著增益 → 判 null，建议 closure，不启动 Stage B 付费重跑。**

- 主终点 **ΔRSR = +0.00 pp**，95% CI **[−3.00, +3.00]**，McNemar exact **p = 1.0000**。
- 次终点 **ΔSSR = −0.33 pp**，95% CI **[−3.33, +2.67]**，**p = 1.0000**。
- 机制上，B1 **唯一应当生效的 `blocker_first` 路径上点估计为负**（RSR −0.67 / SSR −1.33 pp），同样不显著。

---

## 2. 主终点（全样本配对，n=300）

| 指标 | A (EGHC3_v3) | B (B1) | Δ (B−A) | 95% CI | B1赢/输（不一致对） | McNemar exact p | 判定 |
|---|---|---|---|---|---|---|---|
| **RSR** | 186 | 186 | **+0.00 pp** | **[−3.00, +3.00]** | 10 / 10 | 1.0000 | **NULL** |
| **SSR** | 172 | 171 | **−0.33 pp** | **[−3.33, +2.67]** | 10 / 11 | 1.0000 | **NULL** |

> 校验：与两臂 `full_protocol_summary.json` 完全一致（A: RSR 186 / SSR 172；B: RSR 186 / SSR 171），配对无遗漏。

**等效性解读**：n=300 下 95% CI 半宽约 ±3 pp ⇒ **B1 的真实效应被限制在 ±3 pp 以内**，即使存在也不具实用意义。此即"效应量的等效性上界"，而非"证明恰好为 0"。

---

## 3. 机制分层（按**基线臂** route 定义子集，避免选择偏倚）

> ⚠️ 方法要点：`route_mode` 是**处理后变量**（B1 会改变路由分配），因此子集必须用**基线臂 A 的 route** 划分，否则引入选择偏倚。

| 基线 route | n | RSR A→B | ΔRSR | 95% CI | p | SSR A→B | ΔSSR | 95% CI | p |
|---|---|---|---|---|---|---|---|---|---|
| **blocker_first** | 150 | 90→89 | **−0.67 pp** | [−5.33, +4.00] | 1.0000 | 87→85 | **−1.33 pp** | [−6.67, +4.00] | 0.8036 |
| direct | 132 | 92→92 | +0.00 pp | [−3.03, +3.03] | 1.0000 | 82→82 | +0.00 pp | [−3.03, +3.03] | 1.0000 |
| target_first | 18 | 4→5 | +5.56 pp | [0.00, +16.67] | 1.0000 | 3→4 | +5.56 pp | [0.00, +16.67] | 1.0000 |

- **`blocker_first`（n=150，B1 唯一应生效处）**：点估计为负，CI 跨 0 ⇒ **没有机制信号**，不支持启动 Stage B。
- `direct`（n=132）：完全无变化，符合预期（B1 不干预 direct）。
- `target_first`（n=18）：看似 +5.56 pp，但不一致对仅 1 例、CI 跨 0、n 过小 ⇒ **纯噪声，不得解读**。

---

## 4. 关键机制证据：B1 确实改了路由，但没换来成功率

`route_mode` 分布（两臂）：

| route | A (EGHC3_v3) | B (B1) | 变化 |
|---|---|---|---|
| direct | 132 | 134 | +2 |
| **blocker_first** | **150** | **141** | **−9** |
| target_first | 18 | 25 | **+7** |

（`pre_guard_route_mode` / `recommended_route` 同向：blocker_first 188→183，target_first 4→5。）

👉 **这不是"B1 没生效"**：B1 的放宽仲裁确实把 **9 个样本从 blocker_first 改写出去**（主要流向 target_first +7 / direct +2）。但改写后的结果并没有变好——blocker_first 子集 RSR 反降 1 例、SSR 降 2 例。

**判读**：B1 有真实的行为效应（路由迁移），但**行为效应未转化为成功率增益**。这是比"无效应"更强的负证据——机制被激活了却没帮上忙。

---

## 5. 难度分层（探索性）

| subset | n | ΔRSR (CI) | ΔSSR (CI) |
|---|---|---|---|
| easy_no_ambi | 50 | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] |
| easy_yes_ambi | 50 | +0.00 [−6.00, +6.00] | +0.00 [−6.00, +6.00] |
| medium_no_ambi | 50 | −2.00 [−6.00, 0.00] | −2.00 [−6.00, 0.00] |
| medium_yes_ambi | 50 | −2.00 [−12.00, +8.00] | −2.00 [−14.00, +10.00] |
| hard_no_ambi | 50 | +2.00 [−4.00, +8.00] | +2.00 [−4.00, +8.00] |
| hard_yes_ambi | 50 | +2.00 [−8.00, +12.00] | +0.00 [−10.00, +10.00] |

各层 CI 全部跨 0，正负方向不一致（medium 略负、hard 略正）⇒ **无系统性模式**，符合纯噪声。

---

## 6. 重要限制（必须随结论一起报告）

1. **已有两 run 不是纯单变量对照**：`subset_policy`（eghc_consensus_v1 → pathb_blk_arb_v1）、`mimo_retry_mode`（疑 id_repair → none）、`git_commit`（7a34492 → b782344）三处同时变化。
   ⇒ 本结果回答的是**"已有 B1 run 的产物相对 EGHC3_v3 实测差多少"**，严格说**不能**归因为 B1 的纯因果效应。
2. 但本次 null 是**两臂实测对比**，且 B1 侧同时被去掉了 retry（倾向**低估** B1）⇒ 即便如此仍为 null，**更支持"B1 无实用增益"**这一结论。
3. 仅 split-0（n=300）。EGHC3_v3 在 ablation_matrix 内**只有 split_0**，故 full900 配对不可行（缺 A 臂 split1/2 逐样本数据）。
4. 路由迁移分析基于 `route_mode` 字段；`frozen_route_used` 全为 `False`（布尔标志位，非路由名），不可用作路由分层。

---

## 7. 下一步建议

- **不启动 Stage B**（付费重跑）：Stage A 的机制分析（blocker_first 子集无线索）未满足启动条件，按预注册规则应为 **null closure**。
- 论文写法建议：作为**负结果 / limitation** 归档——"在最高可部署方法 EGHC3_v3 上进一步放宽 blocker_first 仲裁（pathB1）未带来显著增益（ΔRSR 0.00 pp，95% CI [−3.00, +3.00]），且该变体虽改变了路由分配（blocker_first 150→141）却未提升成功率"。
- 若仍要因果结论，则需按设计文档 §4.1 跑**纯单变量臂**（两臂仅差 `route_guard_variant`），且**必须先修复 GPU 驱动**（当前 `nvidia-smi` 不可用，sam1/vit_h 跑不动）。

---

## 8. 复现方式

脚本：`E:\FreeGrasp_code-main\pathb_b1_paired_stageA.py`（只读，无副作用）

```
ssh wyl@10.99.16.20 'python3 -' < "E:/FreeGrasp_code-main/pathb_b1_paired_stageA.py"
```

参数：`SEED=20260822`、`B_BOOT=2000`、簇 = 每样本、`ROUTE_KEYS` 优先级 `route_mode > pre_guard_route_mode > recommended_route`（且取值须属 `{direct, target_first, blocker_first}`）。
