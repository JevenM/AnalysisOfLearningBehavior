# -*- coding: utf-8 -*-
"""
figures.py —— 报告插图生成
输出 outputs/figures/fig01..fig10.png（200 dpi，宋体/黑体，适合 A4 单栏）
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import FIG_PALETTE as C
from common import PKG_DIR as PKG, FIG_DIR as OUT
from common import apply_fig_style, clean_axes as _clean

apply_fig_style(dpi=200)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PKG, "04_运行结果")            # 实跑结果（与 multimodal.py / app.py 一致）
os.makedirs(OUT, exist_ok=True)


def _box(ax, x, y, w, h, text, fc="#EAF1F8", ec="#1F4E79", fs=9, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0.9, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3)


def _arrow(ax, xy1, xy2, style="-|>", color="#444444", lw=0.9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))


# ---------------------------------------------------------------- 图1 系统边界
def fig01():
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0.01,rounding_size=0.02",
                                linewidth=0.9, edgecolor="#999999", facecolor="#FAFAFA", linestyle=(0, (4, 3))))
    ax.text(0.51, 0.055, "系统环境：教育政策 · 网络与终端条件 · 社会学习文化 · 平台竞争",
            fontsize=7.6, color="#555555", ha="center")

    ax.add_patch(FancyBboxPatch((0.10, 0.13), 0.84, 0.70, boxstyle="round,pad=0.01,rounding_size=0.02",
                                linewidth=1.2, edgecolor="#1F4E79", facecolor="none"))
    ax.text(0.12, 0.79, "线上学习系统（研究对象）", fontsize=9, fontweight="bold", color="#1F4E79")

    _box(ax, 0.10, 0.46, 0.20, 0.16, "学习者\n（先备知识·动机·时间）", fs=8)
    _box(ax, 0.40, 0.62, 0.20, 0.14, "课程与资源\n（视频·测验·作业）", fs=8)
    _box(ax, 0.40, 0.30, 0.20, 0.14, "平台机制\n（推荐·提醒·论坛）", fs=8)
    _box(ax, 0.70, 0.46, 0.18, 0.16, "教师与助教\n（答疑·督学）", fs=8)
    _box(ax, 0.32, 0.155, 0.36, 0.085, "行为数据：访问·时长·互动·成绩", fs=8,
         fc="#F2F2F2", ec="#7F7F7F")

    _arrow(ax, (0.30, 0.56), (0.40, 0.66), rad=0.05)
    _arrow(ax, (0.30, 0.52), (0.40, 0.38), rad=-0.05)
    _arrow(ax, (0.60, 0.68), (0.70, 0.56), rad=0.05)
    _arrow(ax, (0.60, 0.36), (0.70, 0.49), rad=-0.05)
    _arrow(ax, (0.50, 0.30), (0.50, 0.242))

    _box(ax, 0.36, 0.855, 0.28, 0.072, "学习成效：达标 / 认证 / 留存", fs=8.5,
         fc="#E2EFDA", ec="#548235", bold=True)
    _arrow(ax, (0.50, 0.845), (0.50, 0.83), ls=":", color="#548235", lw=1.0)

    ax.text(0.045, 0.48, "分析与干预（本研究切入点）", rotation=90, fontsize=7.5,
            ha="center", va="center", color="#C55A11")
    _arrow(ax, (0.075, 0.48), (0.10, 0.48), color="#C55A11")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig01_boundary.png"); plt.close(fig)


# ---------------------------------------------------------------- 图2 ISM
def fig02(ism):
    lv = ism["levels"]
    names = ism["names"]
    fig, ax = plt.subplots(figsize=(6.3, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n_lv = len(lv)
    for i, grp in enumerate(lv):
        y = 0.86 - i * (0.76 / max(n_lv - 1, 1)) - 0.04
        ax.text(0.02, y + 0.045, f"L{i+1}", fontsize=9, fontweight="bold", color="#1F4E79")
        w = 0.76 / max(len(grp), 1)
        for j, sid in enumerate(grp):
            x = 0.11 + j * (w + 0.02)
            fc = "#E2EFDA" if i == 0 else ("#EAF1F8" if i < n_lv - 1 else "#FBE5D6")
            ec = "#548235" if i == 0 else ("#1F4E79" if i < n_lv - 1 else "#C55A11")
            _box(ax, x, y, w * 0.96, 0.10, names[sid].replace("（", "\n（"), fs=8, fc=fc, ec=ec)
        if i > 0:
            ax.annotate("", xy=(0.52, y + 0.115), xytext=(0.52, y + 0.175),
                        arrowprops=dict(arrowstyle="-|>", color="#C55A11", lw=1.1))
    ax.text(0.52, 0.955, "表层结果 ← 逐层驱动 → 深层根因", fontsize=9, ha="center",
            fontweight="bold", color="#333333")
    ax.text(0.995, 0.5, "驱动作用方向", rotation=90, fontsize=7.5, va="center",
            ha="right", color="#C55A11")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig02_ism.png"); plt.close(fig)


# ---------------------------------------------------------------- 图3 技术路线
def fig03():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = ["问题界定\n（CATWOE）", "系统建模\n（ISM递阶）", "系统分析\n（K-means画像）",
             "系统预测\n（无泄漏分类）", "系统评价\n（AHP-熵权-模糊）", "系统决策\n（0-1规划）"]
    xs = np.linspace(0.055, 0.845, len(steps))
    for i, (x, s) in enumerate(zip(xs, steps)):
        _box(ax, x, 0.55, 0.135, 0.22, s, fs=7.6, fc="#EAF1F8" if i % 2 == 0 else "#F2F7FD")
        if i:
            _arrow(ax, (xs[i - 1] + 0.135, 0.66), (x, 0.66))
    # 反馈回路
    ax.plot([0.9125, 0.9125, 0.1225], [0.55, 0.30, 0.30], color="#C55A11", lw=1.0, ls="--")
    _arrow(ax, (0.20, 0.30), (0.1225, 0.53), color="#C55A11")
    ax.text(0.50, 0.265, "数据与结论反馈：修正问题界定与干预策略（闭环）", fontsize=7.6,
            color="#C55A11", ha="center")
    # 三个维度说明
    ax.text(0.03, 0.90, "时间维：数据获取 → 建模分析 → 决策实施 → 运行反馈",
            fontsize=7.6, color="#1F4E79")
    ax.text(0.03, 0.06, "知识维：教育学 · 统计学 · 运筹学 · 数据科学 · 平台运营",
            fontsize=7.6, color="#1F4E79")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig03_route.png"); plt.close(fig)


# ---------------------------------------------------------------- 图4 相关矩阵
def fig04(corr):
    cols, M = corr["cols"], np.array(corr["matrix"])
    cn = ["访问次数", "活跃天数", "学习时长", "单次时长", "论坛发帖", "首次滞后",
          "完成度", "作业率", "测验分", "终评成绩"][: len(cols)]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cn, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cn, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.4,
                    color="white" if abs(M[i, j]) > 0.55 else "#333333")
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("Pearson 相关系数", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig04_corr.png"); plt.close(fig)


# ---------------------------------------------------------------- 图5 聚类定阶
def fig05(scan):
    k = [d["k"] for d in scan]
    inert = [d["inertia"] for d in scan]
    sil = [d["silhouette"] for d in scan]
    fig, ax1 = plt.subplots(figsize=(5.4, 3.2))
    ax1.plot(k, inert, "o-", color=C[0], ms=4, lw=1.3, label="簇内平方和 SSE")
    ax1.set_xlabel("聚类数 k"); ax1.set_ylabel("SSE", color=C[0])
    ax1.tick_params(axis="y", labelcolor=C[0]); _clean(ax1)
    ax2 = ax1.twinx()
    ax2.plot(k, sil, "s--", color=C[2], ms=4, lw=1.3, label="轮廓系数")
    ax2.set_ylabel("平均轮廓系数", color=C[2])
    ax2.tick_params(axis="y", labelcolor=C[2])
    for s in ("top",):
        ax2.spines[s].set_visible(False)
    ax1.axvline(4, color="#999999", ls=":", lw=1)
    ax1.annotate("拐点 k=4", xy=(4, inert[2]), xytext=(4.7, inert[2] + (inert[0] - inert[-1]) * 0.22),
                 fontsize=8, color="#333333",
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig05_kselect.png"); plt.close(fig)


# ---------------------------------------------------------------- 图6 画像雷达
def fig06(profile_records):
    dims = [("hours", "学习时长"), ("days", "活跃天数"), ("sess", "单次时长"),
            ("forum", "论坛发帖"), ("video", "资源完成度"), ("assign", "作业提交率"),
            ("quiz", "测验均分")]
    df = pd.DataFrame(profile_records)
    vals = df[[d[0] for d in dims]].values.astype(float)
    lo, hi = vals.min(0), vals.max(0)
    norm = (vals - lo) / (hi - lo + 1e-12)
    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, polar=True)
    for i, row in df.iterrows():
        v = norm[i].tolist(); v += v[:1]
        ax.plot(ang, v, lw=1.4, color=C[i], label=f"{row['name']}（{row['share']:.1%}）")
        ax.fill(ang, v, color=C[i], alpha=0.07)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([d[1] for d in dims], fontsize=8.5)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(color="#DDDDDD", lw=0.6)
    ax.spines["polar"].set_color("#CCCCCC")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig06_radar.png"); plt.close(fig)


# ---------------------------------------------------------------- 图7 ROC
def fig07(roc, metrics):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", color="#AAAAAA", lw=0.9)
    fpr, tpr = roc["fpr"], roc["tpr"]
    ax.plot(fpr, tpr, color=C[0], lw=1.6,
            label=f"Logistic  AUC={metrics['Logistic回归']['auc']:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color=C[0])
    ax.set_xlabel("假正率 FPR"); ax.set_ylabel("真正率 TPR")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    _clean(ax, "both")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("认证取得预测的 ROC 曲线（测试集）", fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig07_roc.png"); plt.close(fig)


# ---------------------------------------------------------------- 图8 特征重要性
def fig08(importance):
    df = pd.DataFrame(importance)
    # 优先用实跑结果自带的中文名（analysis.py 输出 feature_cn），旧字典仅作兜底，
    # 避免特征集调整（如新增 early_w4_days 早期预警特征）后出现 KeyError。
    fallback = {"visit_count": "访问次数", "active_days": "活跃天数", "study_hours": "学习时长",
                "avg_session_min": "单次时长", "forum_posts": "论坛发帖",
                "first_active_lag": "首次参与滞后", "video_rate": "资源完成度",
                "assign_rate": "作业提交率", "quiz_avg": "测验均分",
                "course_num": "选课门数", "early_w4_days": "前4周活跃天",
                "early_w4_clicks": "前4周点击"}
    if "feature_cn" in df.columns:
        df["name"] = [cn if isinstance(cn, str) and cn else fallback.get(f, f)
                      for f, cn in zip(df["feature"], df["feature_cn"])]
    else:
        df["name"] = [fallback.get(f, f) for f in df["feature"]]
    df = df.sort_values("importance")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    colors = [C[0] if v > df["importance"].median() else "#8FAADC" for v in df["importance"]]
    ax.barh(df["name"], df["importance"], color=colors, height=0.62)
    for y, v in enumerate(df["importance"]):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=7.5, color="#333333")
    ax.set_xlabel("随机森林特征重要性（基尼不纯度下降）")
    _clean(ax, "x")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig08_importance.png"); plt.close(fig)


# ---------------------------------------------------------------- 图9 TOPSIS
def fig09(rank):
    df = pd.DataFrame(rank).sort_values("closeness")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    cols = ["#C55A11" if r <= 3 else ("#8FAADC" if r <= 6 else "#BFBFBF") for r in
            (len(df) - df.reset_index().index)]
    ax.barh(df["course_name"], df["closeness"], color="#2E75B6", height=0.62)
    for y, (v, r) in enumerate(zip(df["closeness"], df["rank"])):
        ax.text(v + 0.008, y, f"{v:.3f}（第{r}名）", va="center", fontsize=7.2, color="#333333")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("TOPSIS 相对贴近度 C")
    _clean(ax, "x")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig09_topsis.png"); plt.close(fig)


# ---------------------------------------------------------------- 图10 预算-收益
def fig10(sweep):
    df = pd.DataFrame(sweep)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(df["budget"] / 1e4, df["opt"], "o-", color=C[0], ms=4, lw=1.5, label="0-1 规划（最优）")
    ax.plot(df["budget"] / 1e4, df["heur"], "s--", color=C[2], ms=4, lw=1.3, label="经验法（帮扶最弱）")
    ax.plot(df["budget"] / 1e4, df["uniform"], "^:", color=C[3], ms=4, lw=1.3, label="普惠法（人均平摊）")
    ax.set_xlabel("干预总预算 / 万元"); ax.set_ylabel("预期新增达标人数 / 人")
    _clean(ax, "both")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig10_budget.png"); plt.close(fig)


def build_all(results_path=None):
    results_path = results_path or os.path.join(RESULTS, "results.json")
    R = json.load(open(results_path, encoding="utf-8"))
    fig01(); fig02(R["ism"]); fig03(); fig04(R["corr"]); fig05(R["cluster"]["scan"])
    fig06(R["cluster"]["profile_records"]); fig07(R["predict"]["roc"], R["predict"]["metrics"])
    fig08(R["predict"]["importance"]); fig09(R["evaluate"]["rank"]); fig10(R["sweep"])
    try:
        import multimodal_figures
        multimodal_figures.build_all()
        print(f"[OK] 14 张图已生成 -> {OUT}")
    except ImportError:
        print(f"[OK] 10 张图已生成 -> {OUT}（未找到 multimodal_figures.py，跳过国内多形态插图）")


if __name__ == "__main__":
    build_all()
