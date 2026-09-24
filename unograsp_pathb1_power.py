# -*- coding: utf-8 -*-
"""配对 McNemar power / MDE 计算：UNOGrasp×PathB1 融合实验（split-0, n=300）。
方法：配对二分类，精确二项检验（McNemar exact 的 power 等价于 B(n_disc, 0.5) 检验）。
假设不一致对总数 n_disc 由真实增益 delta 与基线/处理臂的成功率推出：
  p01 - p10 = delta;  n_disc = n * (p01 + p10)
保守取 p10（处理输基线）= 1%（prompt 变体可能有副作用），扫 delta 求 power>=0.8 的最小值。
"""
import math


def binom_pmf(k, n, p):
    return math.comb(n, k) * p**k * (1-p)**(n-k)


def mcnemar_exact_power(n_disc, p01_frac):
    """power of exact binomial two-sided test at alpha=0.05.
    p01_frac = P(B wins | discordant)."""
    if n_disc == 0:
        return 0.0
    # 临界值：双侧 0.05，B(n,0.5) 的拒绝域
    crit_lo, crit_hi = 0, n_disc
    cum = 0.0
    k = 0
    while k <= n_disc and cum + binom_pmf(k, n_disc, 0.5) <= 0.025:
        cum += binom_pmf(k, n_disc, 0.5)
        crit_lo = k + 1
        k += 1
    cum = 0.0
    k = n_disc
    while k >= 0 and cum + binom_pmf(k, n_disc, 0.5) <= 0.025:
        cum += binom_pmf(k, n_disc, 0.5)
        crit_hi = k - 1
        k -= 1
    power = sum(binom_pmf(k, n_disc, p01_frac) for k in range(n_disc + 1)
                if k < crit_lo or k > crit_hi)
    return power


def main():
    n = 300
    base_rate = 186 / 300  # 0.62
    print(f"n={n}, baseline RSR={base_rate:.4f}")
    print(f"{'delta(pp)':>10} {'p01':>8} {'p10':>8} {'n_disc(期望)':>12} {'power':>8}")
    for delta_pp in [3, 4, 5, 6, 7, 8]:
        delta = delta_pp / 100
        for p10 in [0.01, 0.02]:
            p01 = p10 + delta
            if p01 > 1:
                continue
            n_disc = n * (p01 + p10)
            # n_disc 取期望整数
            nd = round(n_disc)
            pw = mcnemar_exact_power(nd, p01 / (p01 + p10))
            print(f"{delta_pp:>10} {p01:>8.3f} {p10:>8.3f} {nd:>12} {pw:>8.3f}")
    # 等效性 CI 半宽参考（B1 实测 ±3.0pp）
    print("\n参考：B1 实测 95% CI 半宽 ±3.0pp（discordant=21 对）")


if __name__ == "__main__":
    main()
