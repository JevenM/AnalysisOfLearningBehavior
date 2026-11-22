# -*- coding: utf-8 -*-
"""生成选题总体方法架构图 fig00_architecture.png（全中文，静态方法学图，不依赖运行结果）。

层次结构（自下而上）：
  数据层（六源真实数据集）→ 特征层（适配器/统一 schema/清洗）→ 建模层（聚类→挖掘→
  相关→预测→评价→仿真→决策 七个环节）→ 应用层（报告/证据图/三个看板/控制台）。
每层标注用到的具体方法与产物文件名，与研究报告第 3 章方法体系一一对应。

用法：python 03_源代码/build_architecture_figure.py
产物：05_图表与看板/figures/fig00_architecture.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(PKG, "05_图表与看板", "figures", "fig00_architecture.png")

from common import setup_cjk_font  # noqa: E402  统一中文字体（模块导入即设置）


def _box(ax, x, y, w, h, text, fc, ec="#1F4E79", fs=9, bold=False, tc="#1B2A3A"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.2, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=4, fontweight="bold" if bold else "normal",
            linespacing=1.45)


def _arrow(ax, x1, y1, x2, y2, color="#4A6E8F"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color=color, lw=1.6, zorder=2))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    setup_cjk_font()
    fig, ax = plt.subplots(figsize=(13, 15.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 15.2)
    ax.axis("off")
    ax.text(6.5, 15.0, "多源学习行为分析统一管线 —— 总体方法架构",
            ha="center", fontsize=17, fontweight="bold", color="#1F4E79")
    ax.text(6.5, 14.55,
            "六源真实数据 → 统一特征表 → 七环节建模链 → 报告与看板（全部固定随机种子，可复现）",
            ha="center", fontsize=10.5, color="#3E5670")

    # ── 数据层 ──────────────────────────────────────────────────────────
    _box(ax, 0.3, 12.5, 2.2, 1.5, "数据层（六源真实数据）\nHarvardX-MITx 慕课\n清华 FIB 短视频\n天池高校慕课",
         "#DCEAF7", fs=8.6)
    _box(ax, 2.75, 12.5, 2.2, 1.5, "（续）\nASSISTments 答题\nOULAD 开放大学\nUCI 课堂对照",
         "#DCEAF7", fs=8.6)
    _box(ax, 5.2, 12.5, 3.2, 1.5, "事件流明细（点击流）\nstudent_id / 时间戳 / 动作 /\n资源类型 / 完成标记",
         "#DCEAF7", fs=8.6)
    _box(ax, 8.65, 12.5, 4.05, 1.5,
         "适配器层 up_adapters.py\n字段别名映射 → 统一 schema\n缺失容忍 + 行为特征聚合\n（50,000 上限随机抽样防单源主导）",
         "#C8DCF0", fs=8.6)
    for x in (1.4, 3.85, 6.8, 10.6):
        _arrow(ax, x, 12.42, x, 11.72)

    # ── 特征层 ──────────────────────────────────────────────────────────
    _box(ax, 0.3, 10.55, 12.4, 1.12,
         "统一特征层  up_build_unified.py —— 24 维统一行为特征 + 标签列 + 人口属性列\n"
         "去重 → IQR 离群截断 → 中位数填充 → 统一表 up_unified.csv（66,644 名学习者）+ 默认事件流 up_clicks.csv",
         "#C8DCF0", fs=9.4, bold=True)
    _arrow(ax, 6.5, 10.48, 6.5, 9.78)

    # ── 建模层（七环节） ────────────────────────────────────────────────
    layers = [
        ("环节① 聚类画像", "肘部法 + 轮廓系数自动选 K\nK-Means（GPU 可加速）\nPCA / t-SNE / UMAP 三视角\n→ 四类画像命名",
         "up_cluster.py\nup_learners.csv\nup_metrics.json"),
        ("环节② 行为挖掘", "LSA 行为序列转移\nSNA 资源共现网络（PageRank）\nApriori 路径关联规则\n时序节律分型 / 场景对比",
         "up_mine.py\nup_mining_summary\n.json"),
        ("环节③ 相关与分层", "Pearson 相关（含 p 值）\n群体异质性分析\n投入三分层增益对比\n随机森林特征重要性",
         "up_associate.py\nup_association\n_summary.json"),
        ("环节④ 预测预警", "RF / GBDT / LogReg\n学业成绩等级预测\n结课认证判定（5 折交叉）\n前 4 周早期预警名单",
         "up_predict.py\nup_early_warning\n_students.csv"),
        ("环节⑤ 统一评价", "AHP 层次分析定权\n（一致性检验 CR）\n模糊综合评判分级\n→ 学习质量评分",
         "up_evaluate.py\nup_learning_quality\n_scores.csv"),
        ("环节⑥ 策略仿真", "系统动力学（SD）\n投入-供给-激励三回路\n四种平台策略 52 周仿真\n→ 最优策略选择",
         "up_evaluate.py\nup_sd_simulation\n*.csv/png"),
        ("环节⑦ 决策建议", "干预队列（证据链+\n教师复核标记）\n画像→内容推荐映射\n全链路优化建议",
         "up_decide.py\nup_queues.csv\nup_optimization\n_suggestions.csv"),
    ]
    y0, box_h, gap = 9.72, 2.06, 0.1
    for i, (title, methods, outputs) in enumerate(layers):
        x = 0.3 + i * 1.79
        _box(ax, x, y0, 1.66, 0.52, title, "#2E6DA4", tc="white", fs=8.8, bold=True)
        _box(ax, x, y0 - 1.34, 1.66, 1.24, methods, "#EAF2FA", fs=7.4)
        _box(ax, x, y0 - 2.02, 1.66, 0.6, outputs, "#F5EFE3", ec="#8A6D3B", fs=7.0)
        if i < len(layers) - 1:
            _arrow(ax, x + 1.68, y0 - 0.72, x + 1.80, y0 - 0.72)
    ax.text(0.35, y0 + 0.66, "建模层：统一管线七环节链（up_cluster → up_report，run_all.py 编排 / 控制台可改参重跑）",
            ha="left", fontsize=10.5, fontweight="bold", color="#1F4E79")
    _arrow(ax, 6.5, y0 - 2.10, 6.5, y0 - 2.62)

    # ── 输出层 ──────────────────────────────────────────────────────────
    _box(ax, 0.3, 5.42, 3.4, 1.5,
         "汇总报告\nup_report.md\n（结论 + 全部关键数字）\n管线审计日志 JSONL",
         "#E8F3E4", ec="#3A7D44", fs=8.8)
    _box(ax, 4.0, 5.42, 4.2, 1.5,
         "18 张环节证据图\n聚类投影 / 行为网络 / 预警特征 /\n质量分布 / SD 仿真 …\n（逐图标注实跑结论）",
         "#E8F3E4", ec="#3A7D44", fs=8.8)
    _box(ax, 8.5, 5.42, 4.2, 1.5,
         "三个交互看板\n主看板（七屏叙事）／形态画像看板\n内容画像看板 —— 跑完自动刷新\n控制台：改参/换数据 → 分钟级重跑",
         "#E8F3E4", ec="#3A7D44", fs=8.8)
    _arrow(ax, 2.0, 5.34, 2.0, 4.78)
    _arrow(ax, 6.1, 5.34, 6.1, 4.78)
    _arrow(ax, 10.6, 5.34, 10.6, 4.78)

    # ── 质量保障条 ──────────────────────────────────────────────────────
    _box(ax, 0.3, 3.6, 12.4, 1.1,
         "全程质量保障：全部真实公开数据（零合成样本）｜固定随机种子 SEED=20260101 全链可复现｜"
         "GPU/CPU 双后端自适应｜沙箱试验（换数据/改参）不污染正式结果｜"
         "数据缺失优雅降级（单源输入自动跳过无数据环节）",
         "#FBF3DC", ec="#B7791F", fs=9.0)
    _arrow(ax, 6.5, 3.52, 6.5, 3.0)

    # ── 证据链定位 ──────────────────────────────────────────────────────
    _box(ax, 0.3, 1.55, 5.9, 1.36,
         "证据线一 · 平台行为主线（本项目核心）\n六源统一表 → 七环节建模链 → 学习者画像 / 预警 / 干预\n"
         "回答：行为如何预测结果、平台应如何干预",
         "#F3E8F7", ec="#7A4E9E", fs=8.8)
    _box(ax, 6.8, 1.55, 5.9, 1.36,
         "证据线二 · 国内内容级证据（多形态测评）\nMOOC / 微课 / 短视频 / 课堂四形态\n"
         "统一指标框架 + 跨形态迁移测评（fig11—14）\n内容画像：时段 / 兴趣 / 标签（fig20—22）",
         "#F3E8F7", ec="#7A4E9E", fs=8.8)
    _arrow(ax, 3.25, 1.47, 3.25, 0.95)
    _arrow(ax, 9.75, 1.47, 9.75, 0.95)
    _box(ax, 2.4, 0.3, 8.2, 0.6,
         "研究报告第 3—8 章：方法体系与实证结果（每章结论均可在看板对应屏与证据图中溯源）",
         "#1F4E79", tc="white", fs=10, bold=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT} 已生成（{os.path.getsize(OUT):,} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
