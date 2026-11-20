# -*- coding: utf-8 -*-
"""
multimodal_figures.py —— 国内多形态层插图 fig11—fig14

数据源：outputs/multimodal_results.json（由 multimodal.py 实跑产生）
输出：../05_图表与看板/figures/fig11..fig14.png（200 dpi，A4 单栏适用）
可单独运行：python multimodal_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import FIG_PALETTE as C
from common import PKG_DIR as PKG, FIG_DIR as OUT
from common import apply_fig_style, clean_axes as _clean

apply_fig_style(dpi=200)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(PKG, "04_运行结果", "multimodal_results.json")  # 与 multimodal.py 输出一致
os.makedirs(OUT, exist_ok=True)


def _box(ax, x, y, w, h, text, fc="#EAF1F8", ec=C[0], fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)


def _arrow(ax, p1, p2, color=C[0], lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=lw, color=color, linestyle=ls, zorder=1))


def fig11(R):
    """三类形态 + 传统课堂对照的研究对象谱系与数据—指标—标签映射。"""
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.955, "学习形态连续统：正式度递减 · 组织化程度递减 · 结果可观测性递减",
            ha="center", fontsize=9.6, fontweight="bold", color=C[0])

    cols = [(0.015, "MOOC", "课程级 / 学期", "edX / OULAD / 学堂在线", "#E8F0F9"),
            (0.262, "微课", "章节级 / 约 7 min", "天池高校慕课画像", "#EAF4E6"),
            (0.509, "短视频", "单条 / 7 日窗口", "清华 FIB WWW 2025", "#FDF0E4"),
            (0.756, "课堂对照", "人—学科 / 学年", "UCI #320", "#F0F0F0")]
    for x, name, gran, src, fc in cols:
        _box(ax, x, 0.705, 0.228, 0.105, name, fc=fc, fs=9.5, bold=True)
        ax.text(x + 0.114, 0.667, gran, ha="center", fontsize=7.2, color="#444444")
        ax.text(x + 0.114, 0.627, src, ha="center", fontsize=6.8, color="#777777")
    for i in range(3):
        _arrow(ax, (cols[i][0] + 0.232, 0.757), (cols[i + 1][0] - 0.006, 0.757),
               color="#999999", lw=1.0)

    ax.text(0.5, 0.545, "统一五维指标框架（形态内 z-score 标准化，禁止跨形态直接合并原始计数）",
            ha="center", fontsize=8.6, color=C[0], fontweight="bold")
    dims = ["投入强度", "持续性", "内容完成度", "互动深度", "情境支持"]
    for i, d in enumerate(dims):
        _box(ax, 0.035 + i * 0.193, 0.455, 0.173, 0.062, d, fc="#F7F9FC", ec="#7F9EC4", fs=8)

    ax.text(0.5, 0.385, "分层标签（同构代理，非同一构念）", ha="center",
            fontsize=8.6, color=C[2], fontweight="bold")
    labs = ["认证 / 通过\n结果性", "单元测验通过\n过程—结果性", "后 2 日回访\n行为持续性", "期末成绩及格\n零在线对照"]
    for i, (x, _, _, _, _) in enumerate(cols):
        _box(ax, x, 0.245, 0.228, 0.105, labs[i], fc="#FFFFFF",
             ec=C[2] if i < 3 else "#7F7F7F", fs=8)

    n = R["headline"]["forms"]
    ax.text(0.5, 0.145,
            f"建模样本：MOOC {n['MOOC']:,} · 微课 {n['微课']:,} · 短视频 {n['短视频']:,} · "
            f"课堂对照 {n['课堂对照']:,} ＝ {R['headline']['n_total']:,} 条",
            ha="center", fontsize=8.4, color="#333333")
    ax.text(0.5, 0.065, "流程：形态内预处理 → 分形态建模 → 分布偏移诊断 → 跨形态迁移测评 → 系统评价 → 干预决策",
            ha="center", fontsize=8, color="#555555")
    fig.savefig(f"{OUT}/fig11_forms.png"); plt.close(fig)


def fig12(R):
    """分布偏移：Wasserstein 距离与 PSI（以 MOOC 为基准）。"""
    dist = R["distribution"]
    forms = [f for f in R["meta"]["forms"] if f != "MOOC"]
    dims = list(R["meta"]["dim_cn"].values())
    keys = R["meta"]["dims"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    w = [[dist["matrix"][f][k]["w1"] for f in forms] for k in keys]
    W = np.array(w)
    x = np.arange(len(keys)); bw = 0.25
    for j, f in enumerate(forms):
        ax1.bar(x + (j - 1) * bw, W[:, j], bw, label=f, color=C[j + 1])
    ax1.set_xticks(x); ax1.set_xticklabels(dims, fontsize=7.4, rotation=18)
    ax1.set_ylabel("分布距离 W1")
    ax1.set_title("(a) 五维分布偏移（相对 MOOC）", fontsize=9)
    _clean(ax1); ax1.legend(fontsize=7.3, frameon=False)

    P = np.array([[dist["matrix"][f][k]["psi"] for k in keys]
                  for f in R["meta"]["forms"]])
    im = ax2.imshow(P, cmap="YlOrRd", aspect="auto", vmin=0, vmax=float(np.percentile(P, 92)))
    ax2.set_xticks(range(5)); ax2.set_xticklabels(dims, fontsize=7.4, rotation=18)
    ax2.set_yticks(range(4)); ax2.set_yticklabels(R["meta"]["forms"], fontsize=7.8)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax2.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=6.8,
                     color="#222222" if P[i, j] < 3 else "white")
    ax2.set_title("(b) PSI 群体稳定性指数", fontsize=9)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig12_shift.png"); plt.close(fig)


def fig13(R):
    """跨形态迁移：AUC 矩阵与域内—迁移对比。"""
    tr = R["transfer"]["pairs"]
    names = ["MOOC", "微课", "短视频"]
    M = np.full((3, 3), np.nan)
    for p in tr:
        M[names.index(p["train"]), names.index(p["test"])] = p["transfer_auc"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1, 1.15]})
    im = ax1.imshow(M, cmap="Blues", vmin=0.45, vmax=0.95)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(names, fontsize=8)
    ax1.set_yticks(range(3)); ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("测试形态 →"); ax1.set_ylabel("训练形态 ↓")
    for i in range(3):
        for j in range(3):
            if not np.isnan(M[i, j]):
                ax1.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                         color="white" if M[i, j] > 0.75 else "#222222")
            else:
                ax1.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999999")
    ax1.set_title("(a) 跨形态迁移 AUC", fontsize=9)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    y = np.arange(len(tr)); width = 0.36
    ax2.barh(y - width / 2, [p["indomain_auc"] for p in tr], width,
             label="域内 5 折 CV", color=C[0])
    ax2.barh(y + width / 2, [p["transfer_auc"] for p in tr], width,
             label="跨域直接迁移", color=C[2])
    ax2.set_yticks(y); ax2.set_yticklabels([f"{p['train']}→{p['test']}" for p in tr], fontsize=7.2)
    ax2.set_xlim(0, 1.05); ax2.set_xlabel("AUC")
    ax2.set_title("(b) 域内 vs 跨域性能", fontsize=9)
    _clean(ax2, "x"); ax2.legend(fontsize=7.2, frameon=False, loc="lower right")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig13_transfer.png"); plt.close(fig)


def fig14(R):
    """系统评价与系统决策：TOPSIS 排序、三方案增益、选中干预组合。"""
    ev, dc = R["evaluate"], R["decide"]
    rank = sorted(ev["rank"], key=lambda d: -d["closeness"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    names = [r["unit"].replace("MOOC·", "") for r in rank]
    vals = [r["closeness"] for r in rank]
    colors = [C[0] if "MOOC" in r["unit"] else (C[3] if "微课" in r["unit"] else
              (C[2] if "短视频" in r["unit"] else "#9E9E9E")) for r in rank]
    ax1.barh(names, vals, color=colors, height=0.62)
    for i, v in enumerate(vals):
        ax1.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7.2)
    ax1.set_xlim(0, max(vals) * 1.22); ax1.set_xlabel("TOPSIS 相对贴近度 C")
    ax1.set_title("(a) 形态×平台 6 单元综合评价", fontsize=9)
    ax1.tick_params(axis="y", labelsize=7.2)
    _clean(ax1, "x")

    labels = ["0-1 规划", "经验法", "普惠法"]
    gains = [dc["opt_gain"], dc["heur_gain"], dc["uniform_gain"]]
    bars = ax2.bar(labels, gains, color=[C[0], C[2], C[3]], width=0.55)
    for b, v in zip(bars, gains):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(gains) * 0.02, f"{v:,.0f}",
                 ha="center", fontsize=8)
    ax2.set_ylim(0, max(gains) * 1.16)
    ax2.set_ylabel("预期新增高成效人数 / 人")
    ax2.set_title(f"(b) 预算 {dc['budget']/1e4:.1f} 万元下的方案对比", fontsize=9)
    _clean(ax2)
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig14_eval_decide.png"); plt.close(fig)


def build_all(results_path: str = RES):
    if not os.path.exists(results_path):
        print(f"[skip] 未找到 {results_path}，请先运行 python multimodal.py")
        return
    R = json.load(open(results_path, encoding="utf-8"))
    fig11(R); fig12(R); fig13(R); fig14(R)
    print(f"[OK] fig11—fig14 已生成 -> {OUT}")


if __name__ == "__main__":
    build_all()
