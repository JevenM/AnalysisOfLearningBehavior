# -*- coding: utf-8 -*-
"""
up_stages.py —— 统一管线全部阶段函数（阶段实现库）

本模块持有原 main.py 的 9 阶段实现函数；同目录的 8 个 up_*.py 为各阶段
独立 CLI 入口（读上游落盘文件 → 写本阶段产物），由 run_all.py 子进程编排。

阶段 → 函数对照：
  [1/2] 数据层+特征层（up_build_unified.py）   build_unified_dataset / build_clickstream / eda_summary
  [3]   分析层（up_cluster.py）                pca_named_components / cluster_personas / label_personas
  [4]   挖掘层（up_mine.py）                   sna_analysis / lsa_analysis / temporal_analysis /
                                               apriori_path_mining / scene_comparison / basic_stats_analysis
  [5]   关联层（up_associate.py）              correlation_analysis / heterogeneity_analysis /
                                               engagement_tier_analysis
  [6]   预测层（up_predict.py）                rf_performance_model / performance_model_extended /
                                               run_predictions / gm11_forecast
  [7]   评价层（up_evaluate.py）               ahp_fuzzy_scores（AHP 委托 evaluators）/ sd_simulation
  [8]   决策层（up_decide.py）                 rule_queues / generate_recommendations /
                                               generate_optimization_suggestions
  [9]   输出层（up_report.py）                 write_report
"""
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   # Agg 后端已在 common 中统一设置
from scipy.stats import pearsonr

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, silhouette_score,
                             r2_score, mean_absolute_error)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from common import (PALETTE, SEED, ClusterEngine, ensure_device, knee_distances,
                    log, df_markdown)
from evaluators import ahp as ahp_core
from up_adapters import ALL_FEATURES, LSA_ACTIONS

# ----------------------------------------------------------------------
# 画像体系（Learner-Persona + ai-personalized-learning 思路）
# ----------------------------------------------------------------------
PERSONA_NAMES = [
    "自主深度型",
    "引导依赖型",
    "考前冲刺型",
    "风险流失型",
]
PERSONA_DESC = {
    "自主深度型": "高投入、深度探索内容、测验得分高、极少求助。",
    "引导依赖型": "中等投入、偏好视频、频繁求助、依赖结构化指导。",
    "考前冲刺型": "日常活跃度低、临近考核爆发、测验尝试率高、策略性取用内容。",
    "风险流失型": "点击与活跃天数极低、得分差、几乎无互动，流失风险高。",
    "高投入·强表现型": "投入高但测验表现一般——学习努力，方法或反馈环节待改进。",
    "高投入·弱表现型": "投入高但测验表现弱——学习时长未转化为成绩，需方法指导。",
    "低投入·强表现型": "投入很低但成绩不错——高效学习型或策略性取用内容。",
    "低投入·弱表现型": "投入与表现双低——与风险流失型同向，需优先干预。",
}
CONTENT_REC = {
    "自主深度型": "研究性阅读材料、开放性项目、同伴讨论论坛、挑战级测评。",
    "引导依赖型": "分步视频教程、例题演示、即时反馈测验、脚手架式学习路径。",
    "考前冲刺型": "精简摘要、记忆卡片、模拟试卷、间隔重复工具、截止日提醒。",
    "风险流失型": "短微内容单元、游戏化激励、主动推送通知、导师早期介入。",
    "高投入·强表现型": "错题诊断反馈、元认知策略训练、同伴互评与答疑。",
    "高投入·弱表现型": "学习方法辅导、分步脚手架测验、错因讲解视频。",
    "低投入·强表现型": "挑战性拓展任务、竞赛与项目制内容，维持学习动机。",
    "低投入·弱表现型": "微任务打卡、游戏化激励、导师早期介入与主动提醒。",
}

# 统一特征列的中文名称（图表坐标轴/图例/热力图刻度使用；未知特征回退原列名）
FEAT_CN = {
    "total_clicks": "总点击量", "active_days": "活跃天数",
    "avg_session_min": "平均会话时长(分)", "forum_posts": "论坛发帖数",
    "resource_clicks": "资源浏览次数", "video_clicks": "视频点击次数",
    "quiz_attempts": "测验尝试次数", "avg_quiz_score": "测验平均分",
    "assignment_submit_lag": "作业提交延迟", "help_requests": "求助次数",
    "login_freq_per_week": "每周登录频次", "accuracy": "答题正确率",
    "avg_response_time": "平均作答用时", "hesitation_rate": "犹豫率",
    "engagement_score": "互动投入度", "hint_usage": "提示使用次数",
    "success_rate": "成功完成率", "skip_rate": "跳过率",
    "unique_days_ratio": "独立活跃天占比", "level_ups": "等级提升次数",
    "code_lines": "代码行数", "code_functions": "代码函数数",
    "comment_ratio": "代码注释率", "video_completion_rate": "视频完播率",
    "study_hours_per_week": "周学习时长", "submission_rate": "作业提交率",
    "study_hours": "学习时长", "attendance_rate": "出勤率",
}


def feat_cn(name: str) -> str:
    """特征列 → 中文名（未收录时原样返回）。"""
    return FEAT_CN.get(name, name)

# AHP 评价体系：三个维度及其下属指标；两两比较矩阵（行为 vs 效果 vs 互动）
AHP_DIMS = ["学习行为", "学习效果", "互动参与"]
AHP_INDICATORS = {
    "学习行为": ["total_clicks", "active_days", "login_freq_per_week", "avg_session_min"],
    "学习效果": ["avg_quiz_score", "accuracy", "success_rate"],
    "互动参与": ["forum_posts", "help_requests", "engagement_score"],
}
AHP_MATRIX = np.array([
    [1.0,   1.0 / 2, 3.0],
    [2.0,   1.0,     4.0],
    [1.0 / 3, 1.0 / 4, 1.0],
])
# 反向指标（取值越大质量越差，归一化时取反）
REVERSE_INDICATORS = {"help_requests", "hint_usage", "assignment_submit_lag",
                      "skip_rate", "avg_response_time"}


# ======================================================================
# 分析层：命名主成分 + 聚类画像
# ======================================================================
def select_cluster_columns(uni: pd.DataFrame, min_fill: float = 0.6) -> list[str]:
    avail = [c for c in ALL_FEATURES if c in uni.columns]
    fill_ratio = uni[avail].notna().mean()
    cols = fill_ratio[fill_ratio >= min_fill].index.tolist()
    log(f"  聚类特征（缺失率 ≤ {1 - min_fill:.0%}）: {cols}")
    return cols


def eda_summary(uni: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    avail = [c for c in ALL_FEATURES if c in uni.columns]
    desc = uni[avail].describe().T.round(2)
    desc["missing"] = uni[avail].isna().sum()
    desc.to_csv(os.path.join(out_dir, "up_eda_summary.csv"), encoding="utf-8-sig")
    return desc


def pca_named_components(uni: pd.DataFrame, cols: list[str], out_dir: str,
                         n_components: int = 4):
    """PCA 降维并按载荷模式为主成分命名（学习投入度/资源偏好度/互动活跃度/学业表现）。"""
    X = StandardScaler().fit_transform(uni[cols].values.astype(float))
    n_comp = min(n_components, len(cols))
    pca = PCA(n_components=n_comp, random_state=SEED)
    scores = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100
    loadings = pd.DataFrame(pca.components_.T, index=cols,
                            columns=[f"PC{i + 1}" for i in range(n_comp)])
    loadings.round(3).to_csv(os.path.join(out_dir, "up_pca_loadings.csv"),
                             encoding="utf-8-sig")

    # 命名启发式：每个 PC 归属于 |载荷| 总和最大的语义域
    DOMAIN_MAP = [
        ("学习投入度", ["total_clicks", "active_days", "avg_session_min",
                        "login_freq_per_week", "engagement_score", "unique_days_ratio"]),
        ("资源偏好度", ["video_clicks", "resource_clicks", "code_lines",
                        "code_functions", "comment_ratio"]),
        ("互动活跃度", ["forum_posts", "help_requests", "hint_usage", "level_ups"]),
        ("学业表现", ["avg_quiz_score", "accuracy", "success_rate",
                      "assignment_submit_lag"]),
    ]
    pc_names = []
    used = set()
    for j in range(n_comp):
        col = loadings.iloc[:, j]
        best_name, best_score = f"PC{j + 1}", -1.0
        for name, feats in DOMAIN_MAP:
            s = sum(abs(col[f]) for f in feats if f in col.index)
            if s > best_score:
                best_name, best_score = name, s
        if best_name in used:
            best_name = f"{best_name}(次要)"
        pc_names.append(best_name)
        used.add(best_name)
    log(f"  PCA 主成分命名: "
        + ", ".join(f"PC{i + 1}={n}({v:.1f}%)" for i, (n, v) in enumerate(zip(pc_names, var))))

    score_df = pd.DataFrame(scores, columns=[f"pc_{n}" for n in pc_names])
    score_df.insert(0, "student_id", uni["student_id"].values)
    score_df.to_csv(os.path.join(out_dir, "up_pca_component_scores.csv"), index=False,
                    encoding="utf-8-sig")
    return score_df, loadings, var, pc_names


def cluster_personas(uni: pd.DataFrame, cols: list[str], k: int, out_dir: str,
                     max_fit: int = 15000, max_plot: int = 5000):
    """肘部法 + 轮廓系数选 K → KMeans（ClusterEngine，支持 GPU）→ PCA/t-SNE/UMAP/雷达/热力图。

    选 K 综合「肘部拐点」与「轮廓系数」两路证据等权融合；显式指定的 k 做合法性
    校验并夹取到 [2, min(8, n-1)]；大数据集（>15000）在固定随机子样本上拟合与
    评估，全量样本用最近质心分配标签。
    """
    X_raw = uni[cols].values.astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    n = len(X)
    if n > max_fit:
        fit_idx = np.sort(np.random.default_rng(SEED).choice(n, max_fit, replace=False))
    else:
        fit_idx = np.arange(n)
    X_fit = X[fit_idx]

    # --- 选 K ---
    k_max = min(8, n - 1)
    k_range = np.arange(2, k_max + 1)
    engine = ClusterEngine(ensure_device())
    inertias, sils = [], []
    for kk in k_range:
        r = engine.fit(X_fit, int(kk))
        inertias.append(r["inertia"])
        sils.append(r["sil"])
    inertias = np.array(inertias, dtype=float)
    sils = np.array(sils, dtype=float)
    sil_n = (sils - np.nanmin(sils)) / (np.nanmax(sils) - np.nanmin(sils) + 1e-9)
    knee_n = knee_distances(k_range.astype(float), inertias)
    knee_n = (knee_n - knee_n.min()) / (knee_n.max() - knee_n.min() + 1e-9)
    knee_k = int(k_range[int(np.argmax(knee_n))])
    combined = 0.5 * sil_n + 0.5 * knee_n
    best_i = int(np.argmax(combined))
    if k <= 0 or k < 2 or k > k_max:
        k = int(k_range[best_i])
        log(f"  自动选择 K = {k}（轮廓系数 {sils[best_i]:.3f}，肘部拐点 K={knee_k}，"
            f"各 K 综合得分: {dict(zip(k_range.tolist(), combined.round(2).tolist()))}）")
    elif k > k_max:
        log(f"  警告: 指定 K={k} 超出可行范围（≤ {k_max}），已夹取为 {k_max}")
        k = k_max

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("聚类数 K 的自动选择（肘部法 + 轮廓系数双准则）",
                 fontsize=14, fontweight="bold")
    axes[0].plot(k_range, inertias, "o-", color=PALETTE[0])
    axes[0].axvline(knee_k, color=PALETTE[3], ls="--", label=f"肘部拐点 K = {knee_k}")
    axes[0].set(xlabel="候选聚类数 K", ylabel="簇内平方和 Inertia（越小越紧凑）",
                title="肘部法：Inertia 随 K 的下降出现拐点")
    axes[0].legend()
    axes[1].plot(k_range, sils, "s-", color=PALETTE[1])
    axes[1].axvline(k, color=PALETTE[2], ls="--", label=f"综合最优 K = {k}")
    axes[1].set(xlabel="候选聚类数 K", ylabel="轮廓系数（越大分离度越好）",
                title="轮廓系数分析：衡量簇内紧密度与簇间分离度")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_elbow_silhouette.png"), bbox_inches="tight")
    plt.close()

    # --- 最终聚类（子样本拟合 + 全量最近质心分配） ---
    r = engine.fit(X_fit, k)
    labels = engine.predict(X, r["centers"])
    sil = r["sil"]
    log(f"  K-Means(K={k}, backend={engine.backend})  "
        f"Silhouette(基于 {len(fit_idx)} 样本) = {sil:.3f}")
    uni["cluster"] = labels

    # --- PCA 散点（大样本时仅绘制子集） ---
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100
    proj = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    proj["cluster"] = labels
    proj.to_csv(os.path.join(out_dir, "up_pca_projection.csv"), index=False)
    log(f"  PCA 方差解释率: PC1={var[0]:.1f}%, PC2={var[1]:.1f}%")
    plot_idx = fit_idx if n > max_plot else np.arange(n)
    lab_p, pca_p = labels[plot_idx], X_pca[plot_idx]

    # 主成分语义解读：取载荷绝对值最大的特征，声明该维度代表什么行为信息
    def _pc_top(j):
        w = pca.components_[j]
        order = np.argsort(-np.abs(w))[:3]
        return "、".join(feat_cn(cols[i]) for i in order)

    fig, ax = plt.subplots(figsize=(9, 6.6))
    for c in range(k):
        m = lab_p == c
        ax.scatter(pca_p[m, 0], pca_p[m, 1], c=PALETTE[c % len(PALETTE)],
                   label=f"画像簇 {c}", alpha=0.75, s=55, edgecolors="white", linewidths=0.4)
    ax.set(xlabel=f"主成分 PC1（解释方差 {var[0]:.1f}%）\n主要由 {_pc_top(0)} 等特征驱动",
           ylabel=f"主成分 PC2（解释方差 {var[1]:.1f}%）\n主要由 {_pc_top(1)} 等特征驱动",
           title="学习者画像聚类 — PCA 二维投影\n（坐标 = 标准化行为特征的线性组合，颜色 = 所属画像簇）")
    ax.legend(title="画像簇", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_pca_clusters.png"), bbox_inches="tight")
    plt.close()

    # --- 非线性嵌入散点：t-SNE + UMAP（展示簇分离度；仅可视化用，不参与建模） ---
    # 坐标尺度无业务含义（簇间距离/簇大小不可解读），不能替代 PCA 建模。
    viz_idx = (np.sort(np.random.default_rng(SEED).choice(
        n, min(n, 10000), replace=False)) if n > 10000 else np.arange(n))
    log(f"  非线性嵌入可视化: {len(viz_idx)} 样本")
    X_pre = PCA(n_components=min(10, X.shape[1]), random_state=SEED) \
        .fit_transform(X[viz_idx])

    try:
        from sklearn.manifold import TSNE
        perp = float(min(30, max(5, len(viz_idx) // 3)))
        log(f"  t-SNE: perplexity={perp:.0f} (大数据集需 1-2 分钟)")
        X_tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                      random_state=SEED).fit_transform(X_pre)
        emb = pd.DataFrame(X_tsne, columns=["tSNE1", "tSNE2"])
        emb["cluster"] = labels[viz_idx]
        emb.to_csv(os.path.join(out_dir, "up_tsne_projection.csv"), index=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        for c in range(k):
            sub = emb[emb["cluster"] == c]
            ax.scatter(sub["tSNE1"], sub["tSNE2"], c=PALETTE[c % len(PALETTE)],
                       label=f"Cluster {c}", alpha=0.55, s=12,
                       edgecolors="none", rasterized=True)
        ax.set(title="学习者画像 — t-SNE 非线性嵌入\n"
                     "（坐标为嵌入空间尺度，无业务含义；本图仅用于展示簇分离结构，不参与建模）")
        ax.set_xlabel("t-SNE 维度 1（无业务含义）")
        ax.set_ylabel("t-SNE 维度 2（无业务含义）")
        ax.legend(title="画像簇", markerscale=3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "up_tsne_clusters.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
    except Exception as e:                      # sklearn 过旧/内存不足时安全跳过
        log(f"  警告: t-SNE 可视化跳过（{type(e).__name__}: {e}）")

    try:
        import umap
        log("  UMAP: n_neighbors=15, min_dist=0.1 (约 10-30 秒)")
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")           # 固定 random_state 会禁用并行，忽略提示
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                random_state=SEED, n_jobs=1)
            X_umap = reducer.fit_transform(X_pre)
        emb = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
        emb["cluster"] = labels[viz_idx]
        emb.to_csv(os.path.join(out_dir, "up_umap_projection.csv"), index=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        for c in range(k):
            sub = emb[emb["cluster"] == c]
            ax.scatter(sub["UMAP1"], sub["UMAP2"], c=PALETTE[c % len(PALETTE)],
                       label=f"Cluster {c}", alpha=0.55, s=12,
                       edgecolors="none", rasterized=True)
        ax.set(title="学习者画像 — UMAP 降维投影\n"
                     "（坐标为嵌入空间尺度，无业务含义；本图仅用于展示簇分离结构，不参与建模）")
        ax.set_xlabel("UMAP 维度 1（无业务含义）")
        ax.set_ylabel("UMAP 维度 2（无业务含义）")
        ax.legend(title="画像簇", markerscale=3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "up_umap_clusters.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
    except ImportError:
        log("  提示: 未安装 umap-learn，UMAP 可视化跳过（pip install umap-learn）")
    except Exception as e:
        log(f"  警告: UMAP 可视化跳过（{type(e).__name__}: {e}）")

    # --- 雷达图 ---
    radar_cols = [c for c in ["total_clicks", "active_days", "avg_session_min",
                              "video_clicks", "quiz_attempts", "avg_quiz_score",
                              "help_requests", "forum_posts", "accuracy",
                              "engagement_score"] if c in cols][:8]
    if radar_cols:
        cm = uni.groupby("cluster")[radar_cols].mean()
        norm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-9)
        n_r = len(radar_cols)
        angles = np.linspace(0, 2 * np.pi, n_r, endpoint=False).tolist()
        angles += angles[:1]
        clusters = list(cm.index)
        ncol = min(len(clusters), 3)
        nrow = int(np.ceil(len(clusters) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow + 0.8),
                                 subplot_kw=dict(polar=True), squeeze=False)
        fig.suptitle("学习者画像行为雷达图\n（每个轴 = 一项行为特征的簇均值经 0–1 归一化，越靠外圈表示该行为越突出）",
                     fontsize=13, fontweight="bold")
        for idx, ax in enumerate(axes.flat):
            if idx >= len(clusters):
                ax.remove()
                continue
            c = clusters[idx]
            vals = norm.loc[c].tolist() + norm.loc[c].tolist()[:1]
            ax.plot(angles, vals, color=PALETTE[idx % len(PALETTE)], lw=2)
            ax.fill(angles, vals, color=PALETTE[idx % len(PALETTE)], alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([feat_cn(c2) for c2 in radar_cols], size=8)
            ax.set_yticklabels([])
            ax.set_title(f"画像簇 {c}（簇内特征均值轮廓）", pad=14, fontweight="bold")
        fig.text(0.5, 0.005,
                 "维度说明：总点击/活跃天数/会话时长 = 学习投入强度；测验均分/正确率 = 学业表现；"
                 "发帖/求助 = 互动与支持需求；视频/资源点击 = 内容消费偏好",
                 ha="center", fontsize=8.5, color="#555555")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "up_radar_personas.png"), bbox_inches="tight")
        plt.close()

    # --- 特征热力图 ---
    hm = uni.groupby("cluster")[cols].mean()
    hm_z = (hm - hm.mean()) / hm.std()
    fig, ax = plt.subplots(figsize=(min(14, 1.0 * len(cols) + 4), 4))
    hm_arr = hm_z.values
    im = ax.imshow(hm_arr, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cols)), [feat_cn(c) for c in cols],
                  rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(k), [f"画像簇 {c}" for c in hm_z.index])
    for i in range(hm_arr.shape[0]):
        for j in range(hm_arr.shape[1]):
            ax.text(j, i, f"{hm_arr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Z 分数（相对全体学习者均值的标准化离差：红=显著高于，绿=显著低于）")
    ax.set_title("各画像簇行为特征对比热力图（Z 分数越红表示该簇在此行为上越突出）",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_heatmap_features.png"), bbox_inches="tight")
    plt.close()

    return uni, engine, scaler, cols, {
        "silhouette": sil, "k": k, "knee_k": knee_k,
        "backend": engine.backend,
        "k_scores": {int(kk): round(float(s), 3) for kk, s in zip(k_range, sils)},
        "n_fit_samples": int(len(fit_idx)),
    }


def label_personas(uni: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """基于簇中心的启发式签名 → 统一画像命名（全中文，簇名严格唯一）。

    命名规则：先按投入/表现/求助签名归入四类标准画像（自主深度型/引导依赖型/
    考前冲刺型/风险流失型），未命中时按投入×表现四象限生成描述型画像名；
    多簇命中同名画像时以"（高投入变体）/（低投入变体）"区分，仍冲突则追加
    "变体 2/3…"——保证每个簇一个唯一名字，推荐表行数恒等于簇数。
    """
    z = uni.groupby("cluster")[cols].mean()
    zscore = (z - uni[cols].mean()) / (uni[cols].std() + 1e-9)

    def get(c, col, default=0.0):
        return zscore.loc[c, col] if col in zscore.columns else default

    mapping, used = {}, set()
    ranked = sorted(z.index, key=lambda c: -get(c, "total_clicks") - get(c, "active_days"))
    for c in ranked:
        eng = get(c, "total_clicks") + get(c, "active_days") + get(c, "engagement_score")
        perf = get(c, "avg_quiz_score") + get(c, "accuracy")
        help_z = get(c, "help_requests") + get(c, "hint_usage")
        video = get(c, "video_clicks")
        quiz_rate = get(c, "quiz_attempts") - get(c, "active_days") * 0.3
        if eng < -0.5 and perf < 0:
            base = "风险流失型"
        elif eng > 0.4 and perf > 0.3 and help_z < 0.3:
            base = "自主深度型"
        elif help_z > 0.3 or video > 0.5:
            base = "引导依赖型"
        elif quiz_rate > 0.2:
            base = "考前冲刺型"
        else:
            base = ("高投入·强表现型" if eng > 0 and perf > 0 else
                    "高投入·弱表现型" if eng > 0 else
                    "低投入·强表现型" if perf > 0 else "低投入·弱表现型")
        if base not in used:
            name = base
        else:                       # 同名画像的第二个及以后簇 → 变体区分
            name = f"{base}（{'高' if eng > 0 else '低'}投入变体）"
            n = 2
            while name in used:     # 兜底去重：绝不允许两簇共用一个名字
                name = f"{base}（变体 {n}）"
                n += 1
        mapping[c] = name
        used.add(name)
    uni["persona"] = uni["cluster"].map(mapping)
    return uni


# ======================================================================
# 挖掘层：SNA / LSA / 时序模式 / Apriori / 场景对比
# ======================================================================
def _pagerank(A: np.ndarray, d: float = 0.85, iters: int = 100) -> np.ndarray:
    """加权 PageRank 幂迭代（无 networkx 依赖）。"""
    n = len(A)
    out = A.sum(1)
    P = np.divide(A, out[:, None], out=np.zeros_like(A, dtype=float),
                  where=out[:, None] > 0)
    pr = np.full(n, 1.0 / n)
    for _ in range(iters):
        pr = d * P.T @ pr + (1 - d) / n
    return pr


def sna_analysis(clicks: pd.DataFrame, out_dir: str, top_k: int = 15):
    """社会网络分析：学习者-资源-课程抽象为节点。
    输出：度中心性、PageRank、资源共现路径关联度（top 转换边）与可视化。"""
    edge = clicks.groupby(["student_id", "resource_id"]).size().reset_index(name="weight")
    learners = edge["student_id"].unique().tolist()
    resources = edge["resource_id"].unique().tolist()
    n = len(learners) + len(resources)
    idx = {s: i for i, s in enumerate(learners)}
    idx.update({s: len(learners) + i for i, s in enumerate(resources)})
    A = np.zeros((n, n))
    # 向量化填装邻接矩阵（20 万级边，逐行 iterrows 会拖慢本阶段）
    si = edge["student_id"].map(idx).to_numpy()
    ri = edge["resource_id"].map(idx).to_numpy()
    wt = edge["weight"].to_numpy(dtype=float)
    A[si, ri] = wt
    A[ri, si] = wt
    deg = A.sum(1)
    pr = _pagerank(A)
    max_deg = deg.max() if deg.max() > 0 else 1.0
    deg_c = deg / max_deg

    lm = pd.DataFrame({
        "student_id": learners,
        "degree": deg[:len(learners)],
        "degree_centrality": deg_c[:len(learners)].round(3),
        "pagerank": pr[:len(learners)].round(5),
    }).sort_values("pagerank", ascending=False)
    lm.to_csv(os.path.join(out_dir, "up_sna_learner_metrics.csv"), index=False,
              encoding="utf-8-sig")

    rm = pd.DataFrame({
        "resource_id": resources,
        "degree": deg[len(learners):],
        "degree_centrality": deg_c[len(learners):].round(3),
        "pagerank": pr[len(learners):].round(5),
        "course_id": [s.split("-R")[0] for s in resources],
    }).sort_values("pagerank", ascending=False)
    rm.to_csv(os.path.join(out_dir, "up_sna_resource_metrics.csv"), index=False,
              encoding="utf-8-sig")

    # 路径关联度：学生内时序上资源→资源的转换次数（top 边）
    ev = clicks.sort_values(["student_id", "timestamp"])
    ev["next_r"] = ev.groupby("student_id")["resource_id"].shift(-1)
    path_edges = (ev.dropna(subset=["next_r"])
                  .groupby(["resource_id", "next_r"]).size()
                  .reset_index(name="count").sort_values("count", ascending=False))
    path_edges = path_edges[path_edges["resource_id"] != path_edges["next_r"]]
    path_edges.head(top_k).to_csv(os.path.join(out_dir, "up_sna_top_paths.csv"),
                                  index=False, encoding="utf-8-sig")

    # 可视化：资源网络（按课程分扇区，节点大小 ∝ PageRank）+ 学习者度分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    courses = rm["course_id"].unique().tolist()
    cmap = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(courses)}
    shown = rm.head(30)
    for _, r in shown.iterrows():
        ang = 2 * np.pi * courses.index(r["course_id"]) / max(len(courses), 1)
        ang += (shown.index.get_loc(r.name) / len(shown) - 0.5) * (2 * np.pi / max(len(courses), 1)) * 0.8
        rad = 0.4 + 0.6 * (1 - r["pagerank"] / (rm["pagerank"].max() + 1e-9))
        x, y = rad * np.cos(ang), rad * np.sin(ang)
        ax1.scatter(x, y, s=100 + 3000 * r["pagerank"], color=cmap[r["course_id"]],
                    alpha=0.75, edgecolors="white", zorder=3)
        ax1.annotate(r["resource_id"], (x, y), fontsize=6, ha="center", va="center")
    # top 路径边
    pos_cache = {}
    for _, r in shown.iterrows():
        ang = 2 * np.pi * courses.index(r["course_id"]) / max(len(courses), 1)
        rad = 0.4 + 0.6 * (1 - r["pagerank"] / (rm["pagerank"].max() + 1e-9))
        pos_cache[r["resource_id"]] = (rad * np.cos(ang), rad * np.sin(ang))
    for _, e in path_edges.head(12).iterrows():
        if e["resource_id"] in pos_cache and e["next_r"] in pos_cache:
            x1, y1 = pos_cache[e["resource_id"]]
            x2, y2 = pos_cache[e["next_r"]]
            ax1.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="->", color="gray", alpha=0.45,
                                         lw=1 + e["count"] / max(path_edges["count"].max(), 1) * 4,
                                         connectionstyle="arc3,rad=0.15"), zorder=1)
    ax1.set_title("资源共现网络\n（节点大小 ∝ PageRank 重要度，箭头 = Top 访问路径）")
    ax1.axis("off")
    ax2.hist(deg_c[:len(learners)], bins=20, color=PALETTE[0], alpha=0.8)
    ax2.set(xlabel="学习者度中心性（连接的独立资源/课程数）", ylabel="学习者人数",
            title="学习者度中心性分布（右偏长尾：多数人只访问少量资源）")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_sna_network.png"), bbox_inches="tight")
    plt.close()

    log(f"  SNA 完成: {len(learners)} 学习者 / {len(resources)} 资源节点; "
        f"PageRank Top 资源: {rm.head(3)['resource_id'].tolist()}")
    return lm, rm, path_edges


def _to_dt(series: pd.Series) -> pd.Series:
    """混合格式时间戳解析：up_clicks.csv 中既有 '2013-02-01' 又有
    '2013-02-01 00:00:00'（不同来源落盘格式不一）。

    pandas>=2.0 用 format='mixed' 逐元素推断；旧版 pandas 退回自动解析
    （errors='coerce' 把无法解析的脏值置 NaT，不中断管线）。
    """
    try:
        return pd.to_datetime(series, format="mixed", errors="coerce")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def lsa_analysis(clicks: pd.DataFrame, out_dir: str):
    """滞后序列分析：点击流行为编码 → lag-1 转换频次 → 调整残差 z 分数
    → 行为转换图谱；并识别 刷课 / 跳步学习 异常模式。"""
    ev = clicks.copy()
    ev["timestamp"] = _to_dt(ev["timestamp"])
    ev["action"] = ev["action"].astype(str).str.lower()
    ev = ev[ev["action"].isin(LSA_ACTIONS)].sort_values(["student_id", "timestamp"])
    ev["prev"] = ev.groupby("student_id")["action"].shift(1)
    tr = (ev.dropna(subset=["prev"]).groupby(["prev", "action"]).size()
            .unstack(fill_value=0)
            .reindex(index=LSA_ACTIONS, columns=LSA_ACTIONS, fill_value=0))
    tr.to_csv(os.path.join(out_dir, "up_lsa_transition_counts.csv"), encoding="utf-8-sig")

    # 调整残差 z 分数（标准 LSA 显著性检验）
    N = tr.values.sum()
    row = tr.values.sum(1, keepdims=True)
    col = tr.values.sum(0, keepdims=True)
    E = row @ col / max(N, 1)
    denom = np.sqrt(E * (1 - row / max(N, 1)) * (1 - col / max(N, 1)) + 1e-12)
    Z = (tr.values - E) / denom
    z_df = pd.DataFrame(Z, index=tr.index, columns=tr.columns).round(2)
    z_df.to_csv(os.path.join(out_dir, "up_lsa_transition_zscores.csv"), encoding="utf-8-sig")

    # 行为转换图谱：节点 = 行为类型，边 = 显著转换（z 越高线越粗）
    fig, ax = plt.subplots(figsize=(8, 7))
    n_a = len(LSA_ACTIONS)
    pos = {a: (np.cos(2 * np.pi * i / n_a), np.sin(2 * np.pi * i / n_a))
           for i, a in enumerate(LSA_ACTIONS)}
    labels_cn = {"video": "视频", "resource": "资源", "quiz": "测验",
                 "forum": "论坛", "help": "求助"}
    flat = [(i, j, Z[i, j]) for i in range(n_a) for j in range(n_a)]
    flat.sort(key=lambda x: -abs(x[2]))
    for i, j, z in flat[:12]:
        if abs(z) < 1.0 or i == j:
            continue
        a1, a2 = LSA_ACTIONS[i], LSA_ACTIONS[j]
        ax.annotate("", xy=pos[a2], xytext=pos[a1],
                    arrowprops=dict(arrowstyle="->", color=PALETTE[0] if z > 0 else PALETTE[2],
                                    alpha=min(abs(z) / 6, 0.9),
                                    lw=1 + min(abs(z), 8) * 0.8,
                                    connectionstyle="arc3,rad=0.18"))
    for a, (x, y) in pos.items():
        ax.scatter(x, y, s=2200, color=PALETTE[1], alpha=0.85, edgecolors="white", zorder=3)
        ax.annotate(labels_cn[a], (x, y), ha="center", va="center", fontsize=11,
                    fontweight="bold", zorder=4)
    ax.set_title("LSA 行为转换图谱（边宽 ∝ |z|，蓝=正显著 红=负显著）",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_lsa_transition_graph.png"), bbox_inches="tight")
    plt.close()

    # 逐学生异常模式检测：刷课（连续视频）/ 跳步学习（未学先测）
    rows = []
    for sid, g in ev.groupby("student_id"):
        acts = g["action"].tolist()
        longest_run, run = 0, 0
        for a in acts:
            run = run + 1 if a == "video" else 0
            longest_run = max(longest_run, run)
        video_ratio = acts.count("video") / len(acts)
        quiz_ratio = acts.count("quiz") / len(acts)
        skips, quizzes = 0, 0
        for p, a in enumerate(acts):
            if a == "quiz":
                quizzes += 1
                window = acts[max(0, p - 3):p]
                if not ("resource" in window or "video" in window):
                    skips += 1
        skip_ratio = skips / quizzes if quizzes else 0.0
        if longest_run >= 5 and video_ratio >= 0.6:
            pattern = "刷课"
        elif skip_ratio > 0.5 and quiz_ratio >= 0.15:
            pattern = "跳步学习"
        else:
            pattern = "正常"
        rows.append({"student_id": sid, "lsa_pattern": pattern,
                     "longest_video_run": longest_run,
                     "video_ratio": round(video_ratio, 3),
                     "skip_quiz_ratio": round(skip_ratio, 3)})
    anoms = pd.DataFrame(rows)
    anoms.to_csv(os.path.join(out_dir, "up_lsa_behavior_patterns.csv"), index=False,
                 encoding="utf-8-sig")
    log(f"  LSA 完成: 行为模式分布 {anoms['lsa_pattern'].value_counts().to_dict()}")
    return anoms, tr, z_df


def temporal_analysis(clicks: pd.DataFrame, out_dir: str):
    """时序模式：登录时间分布（早/中/晚/深夜）、周度周期性、时间行为模式标记。"""
    ev = clicks.copy()
    ev["timestamp"] = _to_dt(ev["timestamp"])   # 点击流自 CSV 读入为字符串，需先解析
    ev["hour"] = ev["timestamp"].dt.hour
    ev["weekday"] = ev["timestamp"].dt.weekday  # 0=周一 ... 6=周日

    def bucket(h):
        if h < 6:
            return "深夜(0-6)"
        if h < 12:
            return "早上(6-12)"
        if h < 18:
            return "下午(12-18)"
        return "晚上(18-24)"

    ev["slot"] = ev["hour"].apply(bucket)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    slot_order = ["早上(6-12)", "下午(12-18)", "晚上(18-24)", "深夜(0-6)"]
    cnt = ev["slot"].value_counts().reindex(slot_order, fill_value=0)
    ax1.bar(slot_order, cnt.values, color=PALETTE[:4])
    ax1.set(title="登录时间分布", ylabel="事件数")
    wd = ev["weekday"].value_counts().sort_index()
    ax2.bar(["周一", "周二", "周三", "周四", "周五", "周六", "周日"],
            [wd.get(i, 0) for i in range(7)], color=PALETTE[0])
    ax2.axhspan(0, max(wd.max(), 1) * 1.05, xmin=5 / 7, color=PALETTE[2], alpha=0.12)
    ax2.set(title="周度行为周期性（周末高亮）", ylabel="事件数")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_temporal_patterns.png"), bbox_inches="tight")
    plt.close()

    # 逐学生时间模式标签
    rows = []
    for sid, g in ev.groupby("student_id"):
        n = len(g)
        night = (g["hour"] < 6).mean()
        weekend = (g["weekday"] >= 5).mean()
        days = g["timestamp"].dt.date.nunique()
        ev_per_day = n / max(days, 1)
        if weekend >= 0.5:
            pat = "集中式周末学习"
        elif night >= 0.35:
            pat = "夜间型学习"
        elif ev_per_day >= 25:
            pat = "碎片化高频学习"
        else:
            pat = "常规型学习"
        rows.append({"student_id": sid, "time_pattern": pat,
                     "night_ratio": round(night, 3), "weekend_ratio": round(weekend, 3),
                     "events_per_day": round(ev_per_day, 1)})
    pat_df = pd.DataFrame(rows)
    pat_df.to_csv(os.path.join(out_dir, "up_temporal_student_patterns.csv"),
                  index=False, encoding="utf-8-sig")
    dist = pat_df["time_pattern"].value_counts().to_dict()
    log(f"  时序模式分布: {dist}")
    return pat_df, dist


def _apriori(transactions: list[frozenset], min_support=0.05, min_conf=0.4):
    """轻量 Apriori：频繁项集挖掘 + 关联规则（support / confidence / lift）。"""
    n = len(transactions)
    cnt1 = Counter()
    for t in transactions:
        for it in t:
            cnt1[frozenset([it])] += 1
    L = {s: c for s, c in cnt1.items() if c / n >= min_support}
    all_L, k = dict(L), 2
    while L and k <= 3:
        keys = list(L.keys())
        C = {keys[i] | keys[j] for i in range(len(keys)) for j in range(i + 1, len(keys))
             if len(keys[i] | keys[j]) == k}
        counts = Counter()
        for t in transactions:
            for c in C:
                if c <= t:
                    counts[c] += 1
        L = {c: v for c, v in counts.items() if v / n >= min_support}
        all_L.update(L)
        k += 1
    rules = []
    for s, c in all_L.items():
        if len(s) < 2:
            continue
        for it in s:
            ant = s - {it}
            if ant not in all_L:
                continue
            conf = c / all_L[ant]
            if conf >= min_conf:
                lift = conf / (all_L[frozenset([it])] / n)
                rules.append({"antecedent": "、".join(sorted(ant)),
                              "consequent": it,
                              "support": round(c / n, 3),
                              "confidence": round(conf, 3),
                              "lift": round(lift, 2)})
    return pd.DataFrame(rules).sort_values("lift", ascending=False) if rules \
        else pd.DataFrame(columns=["antecedent", "consequent", "support",
                                   "confidence", "lift"])


def apriori_path_mining(clicks: pd.DataFrame, out_dir: str,
                        min_support=0.05, min_conf=0.4):
    """学习路径关联规则挖掘（Apriori），并统计负向行为（只看短视频跳过慕课）。"""
    ev = clicks.copy()
    ev["action"] = ev["action"].astype(str).str.lower()
    labels = {"video": "看视频", "resource": "读资源", "quiz": "做测验",
              "forum": "论坛互动", "help": "求助"}
    trans = []
    for sid, g in ev.groupby("student_id"):
        items = {labels.get(a, a) for a in g["action"].unique()}
        if "resource_type" in g.columns:
            rts = g["resource_type"].dropna().unique().tolist()
            rt_map = {"mooc": "慕课长视频", "micro": "微课", "short": "短视频",
                      "oulad": "OULAD 课程资源"}
            items |= {rt_map.get(r, r) for r in rts}
        if items:
            trans.append(frozenset(items))
    rules = _apriori(trans, min_support, min_conf)
    rules.to_csv(os.path.join(out_dir, "up_apriori_rules.csv"), index=False,
                 encoding="utf-8-sig")

    # 负向行为：只看短视频、未接触慕课长视频
    short_only = 0
    for t in trans:
        if "短视频" in t and "慕课长视频" not in t:
            short_only += 1
    if rules.empty:
        log("  Apriori: 未挖掘到满足阈值的规则（可降低 min_support）")
        return rules, {"short_only_pct": 0.0, "top_rules": []}
    # 规则可视化（Top 10 by lift）
    top = rules.head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    txt = [f"{r.antecedent} → {r.consequent}" for r in top.itertuples()]
    ax.barh(txt[::-1], top["lift"][::-1], color=PALETTE[1])
    ax.set(xlabel="提升度 Lift（>1 表示规则前项会提升后项出现概率）",
           title="学习路径关联规则 Top10（Apriori，按提升度排序）")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_apriori_rules.png"), bbox_inches="tight")
    plt.close()
    summary = {"short_only_pct": round(100 * short_only / max(len(trans), 1), 1),
               "top_rules": top.head(3).to_dict("records")}
    log(f"  Apriori 规则 {len(rules)} 条; 只看短视频跳过完整慕课占比: "
        f"{summary['short_only_pct']}%")
    return rules, summary


def scene_comparison(clicks: pd.DataFrame, out_dir: str):
    """慕课长视频 / 微课 / 短视频 / OULAD 课程资源场景的完播率、倍速使用率、拖拽行为对比。"""
    vid = clicks[clicks["action"] == "video"].copy()
    if "resource_type" not in vid.columns or vid["resource_type"].dropna().empty:
        log("  场景对比: 点击流缺少 resource_type 字段，跳过")
        return pd.DataFrame(), pd.DataFrame(), {}
    vid = vid.dropna(subset=["resource_type"])
    rt_names = {"mooc": "慕课长视频", "micro": "微课", "short": "短视频"}
    vid["场景"] = vid["resource_type"].map(rt_names)
    vid = vid[vid["场景"].notna()]
    tbl = vid.groupby("场景").agg(
        播放次数=("completed", "count"),
        完播率=("completed", "mean"),
        倍速使用率=("speed_used", lambda s: (s > 1).mean()),
        平均拖拽次数=("seek_count", "mean"),
    ).round(3)
    tbl["播放占比"] = (tbl["播放次数"] / tbl["播放次数"].sum()).round(3)
    tbl.to_csv(os.path.join(out_dir, "up_scene_comparison.csv"), encoding="utf-8-sig")

    # 逐学生完播率（供相关分析使用）
    per_stu = vid.groupby(["student_id", "场景"])["completed"].mean().unstack(fill_value=np.nan)
    per_stu["video_completion_rate"] = vid.groupby("student_id")["completed"].mean().round(3)
    per_stu["preferred_scene"] = vid.groupby("student_id")["场景"] \
        .agg(lambda s: s.value_counts().idxmax())
    per_stu = per_stu.reset_index()
    per_stu.to_csv(os.path.join(out_dir, "up_scene_per_student.csv"), index=False,
                   encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    plot_tbl = tbl.reset_index()
    axes[0].bar(plot_tbl["场景"], plot_tbl["完播率"], color=PALETTE[:3])
    axes[0].set(title="各场景完播率")
    axes[1].bar(plot_tbl["场景"], plot_tbl["倍速使用率"], color=PALETTE[:3])
    axes[1].set(title="各场景倍速使用率")
    axes[2].bar(plot_tbl["场景"], plot_tbl["平均拖拽次数"], color=PALETTE[:3])
    axes[2].set(title="各场景平均拖拽次数")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_scene_comparison.png"), bbox_inches="tight")
    plt.close()
    log(f"  场景对比: {tbl[['完播率', '倍速使用率', '平均拖拽次数']].to_dict('index')}")
    return tbl, per_stu, tbl[["完播率", "倍速使用率", "平均拖拽次数"]].to_dict("index")


def basic_stats_analysis(uni: pd.DataFrame, out_dir: str) -> dict:
    """人口属性频数统计 + 基础行为指标基线 + 静态属性×行为交叉对比。"""
    demo_cols = [c for c in ["gender", "age_band", "region", "education"]
                 if c in uni.columns]
    # 1) 人口属性频数
    rows = []
    for c in demo_cols:
        vc = uni[c].value_counts()
        for cat, cnt in vc.items():
            rows.append({"dimension": c, "category": cat, "count": cnt,
                         "pct": round(100 * cnt / len(uni), 1)})
    demo_df = pd.DataFrame(rows)
    demo_df.to_csv(os.path.join(out_dir, "up_demographics_distribution.csv"),
                   index=False, encoding="utf-8-sig")
    # 某维度整体缺失（如单来源数据的 region）时优雅降级，不取首行报错
    core = {c: demo_df[demo_df["dimension"] == c].iloc[0]["category"]
            for c in demo_cols
            if (demo_df["dimension"] == c).any()}
    if len(core) < len(demo_cols):
        missing = [c for c in demo_cols if c not in core]
        log(f"  警告: 人口维度 {missing} 在输入数据中无有效值，跳过该维度统计")
    log(f"  人口结构: 核心群体 {core}")

    # 2) 基础行为指标基线
    baseline_cols = [c for c in ["login_freq_per_week", "study_hours_per_week",
                                 "video_completion_rate", "submission_rate",
                                 "forum_posts", "total_clicks", "active_days",
                                 "avg_session_min"] if c in uni.columns]
    desc = uni[baseline_cols].describe().T[["mean", "50%", "min", "max", "std"]]
    desc.columns = ["均值", "中位数", "最小值", "最大值", "标准差"]
    desc.round(2).to_csv(os.path.join(out_dir, "up_behavior_baseline.csv"),
                         encoding="utf-8-sig")

    # 3) 静态属性 × 行为交叉对比
    xrows = []
    for c in demo_cols:
        agg_map = {
            "平均学习时长": ("study_hours_per_week", "mean"),
            "视频偏好度": ("video_clicks", "mean"),
            "平均完成率": ("completion_rate", "mean"),
            "平均成绩": ("final_grade", "mean"),
            "人数": ("student_id", "count"),
        }
        if "video_completion_rate" in uni.columns:
            agg_map["平均完播率"] = ("video_completion_rate", "mean")
        g = uni.groupby(c).agg(**agg_map)
        for cat, r in g.iterrows():
            xrows.append({"dimension": c, "category": cat,
                          **{k: round(v, 2) for k, v in r.items()}})
    cross = pd.DataFrame(xrows)
    cross.to_csv(os.path.join(out_dir, "up_group_behavior_comparison.csv"),
                 index=False, encoding="utf-8-sig")

    # 可视化：教育背景 × 行为差异（代表性交叉图）
    if "education" in uni.columns and "study_hours_per_week" in uni.columns:
        g = uni.groupby("education")[["study_hours_per_week", "completion_rate",
                                      "final_grade"]].mean()
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(g))
        ax2 = ax.twinx()
        ax.bar(x - 0.2, g["study_hours_per_week"], 0.4, color=PALETTE[0], label="周均学习时长")
        ax2.bar(x + 0.2, g["final_grade"], 0.4, color=PALETTE[1], label="最终成绩")
        ax.set_xticks(x, g.index)
        ax.set_ylabel("周均学习时长(小时)")
        ax2.set_ylabel("最终成绩")
        ax.set_title("不同教育背景学习者的行为分化", fontweight="bold")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "up_demographics_cross.png"), bbox_inches="tight")
        plt.close()
    return {"core_groups": core, "demo_df": demo_df, "cross": cross}


# ======================================================================
# 关联层：相关性 / 异质性 / 投入分层
# ======================================================================
def correlation_analysis(uni: pd.DataFrame, out_dir: str):
    """行为指标与最终成绩/结课认证的皮尔逊相关分析（含显著性）。"""
    uni = uni.copy()
    uni["certified"] = (uni["completion_rate"] >= 0.75).astype(int)
    indicators = [c for c in ALL_FEATURES + ["video_completion_rate",
                                             "study_hours_per_week", "submission_rate"]
                  if c in uni.columns and pd.api.types.is_numeric_dtype(uni[c])]
    rows = []
    for ind in indicators:
        for tgt in ["final_grade", "certified"]:
            pair = uni[[ind, tgt]].dropna()
            if len(pair) < 10 or pair[ind].std() == 0:
                continue
            r, p = pearsonr(pair[ind], pair[tgt])
            rows.append({"indicator": ind, "target": tgt, "pearson_r": round(r, 3),
                         "p_value": float(f"{p:.2e}"),
                         "significant": "是" if p < 0.05 else "否"})
    corr = pd.DataFrame(rows)
    corr.to_csv(os.path.join(out_dir, "up_correlation_results.csv"), index=False,
                encoding="utf-8-sig")
    g = corr[corr["target"] == "final_grade"].sort_values("pearson_r", key=abs,
                                                          ascending=False)
    top = g.head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE[1] if r > 0 else PALETTE[2] for r in top["pearson_r"]]
    ax.barh([feat_cn(i) for i in top["indicator"]][::-1], top["pearson_r"][::-1],
            color=colors[::-1])
    ax.set(xlabel="Pearson 相关系数 r（与最终成绩；向右=正相关，向左=负相关）",
           title="行为指标与学业效果相关性 Top10")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_correlation_top_factors.png"), bbox_inches="tight")
    plt.close()
    log(f"  相关性 Top3（成绩）: {top.head(3)[['indicator', 'pearson_r']].to_dict('records')}")
    return corr, top.head(5)


def heterogeneity_analysis(uni: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """性别/地区/教育背景是否调节"学习时长-成绩"相关性（异质性分析）。"""
    rows = []
    for dim in [c for c in ["gender", "region", "education"] if c in uni.columns]:
        for cat, g in uni.groupby(dim):
            pair = g[["study_hours_per_week", "final_grade"]].dropna()
            if len(pair) < 15 or pair["study_hours_per_week"].std() == 0:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(pair["study_hours_per_week"], pair["final_grade"])
            rows.append({"dimension": dim, "category": cat, "n": len(pair),
                         "r_时长-成绩": round(r, 3) if pd.notna(r) else np.nan,
                         "p_value": float(f"{p:.2e}") if pd.notna(p) else np.nan})
    het = pd.DataFrame(rows)
    het.to_csv(os.path.join(out_dir, "up_heterogeneity_analysis.csv"), index=False,
               encoding="utf-8-sig")
    if len(het) and het["r_时长-成绩"].notna().any():
        log(f"  异质性分析: 各群体学习时长-成绩相关系数范围 "
            f"[{het['r_时长-成绩'].min():.3f}, {het['r_时长-成绩'].max():.3f}]")
    return het


def engagement_tier_analysis(uni: pd.DataFrame, out_dir: str):
    """学习投入度低/中/高三分组，量化投入对学习效果的提升幅度。"""
    parts = [c for c in ["total_clicks", "active_days", "avg_session_min",
                         "engagement_score"] if c in uni.columns]
    z = ((uni[parts] - uni[parts].mean()) / (uni[parts].std() + 1e-9)).mean(axis=1)
    uni = uni.assign(engagement_composite=z)
    uni["投入分层"] = pd.qcut(z, q=[0, 1 / 3, 2 / 3, 1], labels=["低投入", "中投入", "高投入"])
    tbl = uni.groupby("投入分层", observed=True).agg(
        人数=("student_id", "count"),
        平均课程完成率=("completion_rate", "mean"),
        平均最终成绩=("final_grade", "mean"),
        平均测验分=("avg_quiz_score", "mean"),
    ).round(2)
    tbl["完成率提升幅度"] = (100 * (tbl["平均课程完成率"] - tbl["平均课程完成率"].iloc[0])).round(1)
    tbl["成绩提升幅度"] = (tbl["平均最终成绩"] - tbl["平均最终成绩"].iloc[0]).round(1)
    tbl.to_csv(os.path.join(out_dir, "up_engagement_tier_analysis.csv"), encoding="utf-8-sig")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    ax1.bar(x - 0.2, tbl["平均课程完成率"], 0.4, color=PALETTE[0], label="平均课程完成率")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, tbl["平均最终成绩"], 0.4, color=PALETTE[1], label="平均最终成绩")
    ax1.set_xticks(x, tbl.index)
    ax1.set_ylabel("课程完成率")
    ax2.set_ylabel("最终成绩")
    ax1.set_title("行为投入度分层与学习效果对比", fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_engagement_tiers.png"), bbox_inches="tight")
    plt.close()
    log(f"  投入分层: 高投入组较低投入组 完成率 +{tbl['完成率提升幅度'].iloc[2]}%, "
        f"成绩 +{tbl['成绩提升幅度'].iloc[2]} 分")
    return uni, tbl


# ======================================================================
# 预测层：RF/GBDT/LogReg/早期预警/GM(1,1)
# ======================================================================
def run_predictions(uni: pd.DataFrame, cols: list[str], out_dir: str) -> dict:
    res = {}
    y = uni["success"].astype(int)
    X = uni[cols].fillna(uni[cols].median())
    if y.nunique() < 2:
        log("  警告: 标签 success 仅 1 类（输入数据覆盖面不足），跳过成功/风险预测")
        return res

    # 1) LogisticRegression 成功预测（data-pipeline 思路）
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                              stratify=y)
    lr = LogisticRegression(max_iter=1000).fit(StandardScaler().fit_transform(X_tr), y_tr)
    proba = lr.predict_proba(StandardScaler().fit(X_tr).transform(X_te))[:, 1]
    res["logreg_acc"] = accuracy_score(y_te, proba > 0.5)
    res["logreg_auc"] = roc_auc_score(y_te, proba)
    coef = pd.Series(lr.coef_[0], index=cols).sort_values(key=abs, ascending=False)
    coef.head(10).to_csv(os.path.join(out_dir, "up_logreg_coefficients.csv"),
                         encoding="utf-8-sig")
    log(f"  LogisticRegression 成功预测: acc={res['logreg_acc']:.3f} "
        f"AUC={res['logreg_auc']:.3f}")

    # 2) RandomForest 风险预测 + 早期特征 vs 全特征（早期预警思想）
    activity_cols = [c for c in ["total_clicks", "active_days", "login_freq_per_week",
                                 "engagement_score", "avg_session_min"] if c in cols]
    early_cols = activity_cols or cols[:3]
    rf_full = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf_early = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    cv_full = cross_val_score(rf_full, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    cv_early = cross_val_score(rf_early, X[early_cols], y, cv=5, scoring="roc_auc",
                               n_jobs=-1)
    res["rf_auc_full"] = cv_full.mean()
    res["rf_auc_early"] = cv_early.mean()
    log(f"  RandomForest 风险预测 5-fold AUC: 全特征={res['rf_auc_full']:.3f}  "
        f"早期特征({early_cols})={res['rf_auc_early']:.3f}")
    rf_full.fit(X, y)
    imp = pd.Series(rf_full.feature_importances_, index=cols).sort_values(ascending=False)
    imp.to_csv(os.path.join(out_dir, "up_rf_feature_importance.csv"), encoding="utf-8-sig")

    # 3) 由行为特征预测画像标签（MSc Thesis predictive_models 思路）
    le = LabelEncoder()
    y_p = le.fit_transform(uni["persona"])
    cv_p = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=SEED,
                                                  n_jobs=-1),
                           X, y_p, cv=5, scoring="accuracy", n_jobs=-1)
    res["persona_pred_acc"] = cv_p.mean()
    log(f"  RandomForest 由行为预测画像标签: acc={res['persona_pred_acc']:.3f} "
        f"(classes={list(le.classes_)})")
    return res


def rf_performance_model(uni: pd.DataFrame, cols: list[str], out_dir: str) -> dict:
    """RF 学业表现预测：历史行为特征 → 课程完成率（回归）+ 最终成绩（回归）
    + 是否完成（分类），并输出特征重要性排序。"""
    from sklearn.ensemble import RandomForestRegressor

    res = {}
    X = uni[cols].fillna(uni[cols].median())
    X_tr, X_te, y_tr, y_te = train_test_split(X, uni, test_size=0.3, random_state=SEED)

    # 最终成绩（回归）
    rf_g = RandomForestRegressor(n_estimators=200, random_state=SEED,
                                 n_jobs=-1).fit(X_tr, y_tr["final_grade"])
    pred_g = rf_g.predict(X_te)
    res["rf_grade_r2"] = r2_score(y_te["final_grade"], pred_g)
    res["rf_grade_mae"] = mean_absolute_error(y_te["final_grade"], pred_g)
    # 课程完成率（回归）
    rf_c = RandomForestRegressor(n_estimators=200, random_state=SEED,
                                 n_jobs=-1).fit(X_tr, y_tr["completion_rate"])
    pred_c = rf_c.predict(X_te)
    res["rf_completion_r2"] = r2_score(y_te["completion_rate"], pred_c)
    # 是否完成（分类：完成率 ≥ 0.75）
    thr = 0.75
    rf_f = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf_f.fit(X_tr, (y_tr["completion_rate"] >= thr))
    res["rf_completion_acc"] = accuracy_score((y_te["completion_rate"] >= thr),
                                              rf_f.predict(X_te))
    log(f"  RF 学业表现预测: 最终成绩 R²={res['rf_grade_r2']:.3f} "
        f"(MAE={res['rf_grade_mae']:.2f}), 完成率 R²={res['rf_completion_r2']:.3f}, "
        f"完成判定 acc={res['rf_completion_acc']:.3f}")

    # 特征重要性排序（以成绩模型为准）
    imp = pd.Series(rf_g.feature_importances_, index=cols).sort_values(ascending=False)
    imp.to_csv(os.path.join(out_dir, "up_rf_grade_feature_importance.csv"),
               encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.head(12)[::-1].plot.barh(ax=ax, color=PALETTE[0])
    ax.set_yticklabels([feat_cn(i) for i in imp.head(12)[::-1].index], fontsize=9)
    ax.set(title="随机森林特征重要性 — 最终成绩预测 Top12",
           xlabel="重要性（该特征在模型中的基尼不纯度下降贡献）")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_rf_feature_importance.png"), bbox_inches="tight")
    plt.close()
    return res


def performance_model_extended(uni: pd.DataFrame, cols: list[str], out_dir: str):
    """RF/GBDT 学业表现预测（成绩等级 + 是否结课），准确率/召回率 + Top5 关键行为，
    并输出高辍学风险学生早期预警名单。"""
    from sklearn.ensemble import GradientBoostingClassifier

    uni = uni.copy()
    X = uni[cols].fillna(uni[cols].median())
    uni["grade_band"] = pd.cut(uni["final_grade"].clip(lower=0), [-0.1, 60, 70, 85, 100.01],
                               labels=["D", "C", "B", "A"])
    uni["certified"] = (uni["completion_rate"] >= 0.75).astype(int)
    has_band = uni["grade_band"].notna()
    X_tr, X_te, u_tr, u_te = train_test_split(X[has_band], uni[has_band],
                                              test_size=0.3, random_state=SEED)

    res, models = {}, {}
    band_classes = pd.unique(u_tr["grade_band"])
    if len(band_classes) < 2:
        # 自定义输入数据等级单一（如仅含一门课/一个群体）时优雅降级：跳过等级分类，
        # 只保留二分类与回归口径，避免训练集类数为 1 时模型崩溃。
        log("  警告: 训练集成绩等级仅 1 类（输入数据覆盖面不足），跳过等级分类，"
            "仅输出认证二分类与回归口径")
    else:
        for name, clf in [("RandomForest", RandomForestClassifier(n_estimators=200,
                                                                  random_state=SEED,
                                                                  n_jobs=-1)),
                          ("GradientBoosting", GradientBoostingClassifier(random_state=SEED))]:
            clf.fit(X_tr, u_tr["grade_band"])
            pred = clf.predict(X_te)
            res[f"{name}_grade_acc"] = accuracy_score(u_te["grade_band"], pred)
            res[f"{name}_grade_recall"] = recall_score(u_te["grade_band"], pred,
                                                       average="macro", zero_division=0)
            res[f"{name}_grade_precision"] = precision_score(u_te["grade_band"], pred,
                                                             average="macro", zero_division=0)
            models[name] = clf
    # 是否结课认证（二分类）
    rf_cert = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                     n_jobs=-1).fit(X_tr, u_tr["certified"])
    cert_pred = rf_cert.predict(X_te)
    res["cert_acc"] = accuracy_score(u_te["certified"], cert_pred)
    res["cert_recall_risk"] = recall_score(u_te["certified"], cert_pred,
                                           pos_label=0, zero_division=0)
    if models:
        best = max(models, key=lambda m: res[f"{m}_grade_acc"])
    else:
        best = "RandomForest"
    res["best_model"] = best

    # Top5 关键行为特征（等级模型可用时取等级模型，降级时取认证模型）
    imp = pd.Series(models.get(best, rf_cert).feature_importances_,
                    index=cols).sort_values(ascending=False)
    res["top5_features"] = imp.head(5).round(3).to_dict()
    fig, ax = plt.subplots(figsize=(9, 5))
    imp.head(10)[::-1].plot.barh(ax=ax, color=PALETTE[0])
    ax.set_yticklabels([feat_cn(i) for i in imp.head(10)[::-1].index], fontsize=9)
    ax.set_xlabel("重要性（该特征在模型中的基尼不纯度下降贡献）")
    ax.set_title(f"学业结果预测特征重要性 Top10（{best} 模型；前 4 周早期行为同样适用）")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_model_feature_importance.png"), bbox_inches="tight")
    plt.close()

    # 早期预警：结课概率最低的学生（高辍学风险）
    if 1 in rf_cert.classes_:
        proba = rf_cert.predict_proba(X)[:, list(rf_cert.classes_).index(1)]
    else:
        log("  警告: certified 仅 1 类，早期预警概率降级为常数 0")
        proba = pd.Series(0.0, index=X.index)
    ew = uni[["student_id", "persona", "region", "completion_rate", "final_grade"]].copy()
    ew["cert_prob"] = proba.round(3)
    ew["risk_level"] = pd.cut(proba, [0, 0.2, 0.4, 1], labels=["高", "中", "低"])
    ew = ew.sort_values("cert_prob").head(50)
    ew.to_csv(os.path.join(out_dir, "up_early_warning_students.csv"), index=False,
              encoding="utf-8-sig")
    grade_txt = (f"成绩等级 acc={res[best + '_grade_acc']:.3f} "
                 f"recall={res[best + '_grade_recall']:.3f}; "
                 if best in models else "成绩等级: 已跳过（等级单一）; ")
    log(f"  扩展建模: 最佳模型={best} {grade_txt}认证 acc={res['cert_acc']:.3f}; "
        f"Top5 关键行为: {list(res['top5_features'])}")
    return res, ew


def _gm11(x0: np.ndarray):
    """灰色 GM(1,1)：单序列少数据预测。返回 (发展系数 a, 灰作用量 b, 后验差比 C, 预测函数)。"""
    x0 = np.asarray(x0, dtype=float)
    x0 = x0 + 1e-6  # 防零
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[1:] + x1[:-1])
    B = np.column_stack([-z1, np.ones(len(z1))])
    Y = x0[1:]
    coef = np.linalg.lstsq(B, Y, rcond=None)[0]
    a, b = coef[0], coef[1]

    def f(k: int) -> float:
        # f(k) 为累加序列 x̂1(k)，k 从 0 起
        if abs(a) < 1e-9:
            return x0[0]
        return (x0[0] - b / a) * np.exp(-a * k) + b / a

    sim1 = np.array([f(k) for k in range(len(x0))])
    sim = np.diff(sim1, prepend=sim1[0] - x0[0])  # 累减还原
    C = np.std(x0 - sim) / (np.std(x0) + 1e-12)   # 后验差比（越小越好）
    return a, b, C, f


def gm11_forecast(clicks: pd.DataFrame, out_dir: str, horizon: int = 2,
                  min_points: int = 4) -> pd.DataFrame:
    """针对每位学习者的日活跃度序列做 GM(1,1) 趋势预测，
    弥补机器学习模型对小样本群体预测差的短板。

    注：采用「天」粒度——清华短视频公开子样本的事件时间仅覆盖约一周
    （每用户最多 7 个日序列点），周粒度序列过短会导致全员数据不足。
    """
    ev = clicks.copy()
    ev["day"] = _to_dt(ev["timestamp"]).dt.normalize()
    daily = ev.groupby(["student_id", "day"]).size().unstack(fill_value=0)
    rows = []
    for sid, series in daily.iterrows():
        x0 = series.values.astype(float)
        if len(x0) < min_points:
            rows.append({"student_id": sid, "n_days": len(x0), "a": np.nan,
                         "posterior_C": np.nan, "trend": "数据不足",
                         "pred_next1": np.nan, "pred_next2": np.nan, "model_grade": "-"})
            continue
        a, b, C, f = _gm11(x0)
        preds = [max(f(k), 0) for k in range(len(x0), len(x0) + horizon)]
        if a < -0.02:
            trend = "上升"
        elif a > 0.02:
            trend = "下降"
        else:
            trend = "平稳"
        grade = "优" if C < 0.35 else ("良" if C < 0.5 else "合格")
        rows.append({"student_id": sid, "n_days": len(x0), "a": round(a, 4),
                     "posterior_C": round(C, 3), "trend": trend,
                     "pred_next1": round(preds[0], 1),
                     "pred_next2": round(preds[1], 1) if len(preds) > 1 else np.nan,
                     "model_grade": grade})
    gm = pd.DataFrame(rows)
    gm.to_csv(os.path.join(out_dir, "up_gm11_forecast.csv"), index=False,
              encoding="utf-8-sig")
    valid = gm[gm["trend"] != "数据不足"]
    log(f"  GM(1,1) 完成: 有效预测 {len(valid)}/{len(gm)} 人, "
        f"趋势分布 {valid['trend'].value_counts().to_dict()}, "
        f"模型精度分布 {valid['model_grade'].value_counts().to_dict()}")
    return gm


# ======================================================================
# 评价层：AHP 模糊综合评判（AHP 委托 evaluators 唯一实现）+ SD 仿真
# ======================================================================
def _trap(v: float, a: float, b: float, c: float, d: float) -> float:
    """梯形隶属度函数 μ(v; a,b,c,d)。"""
    if v <= a or v >= d:
        return 0.0
    if b <= v <= c:
        return 1.0
    if v < b:
        return (v - a) / (b - a)
    return (d - v) / (d - c)


# 评语集：优/良/中/差 的梯形隶属参数与等级分值
FUZZY_GRADES = [
    ("优", (0.80, 0.90, 1.01, 1.01), 95),
    ("良", (0.70, 0.80, 0.90, 1.00), 80),
    ("中", (0.45, 0.55, 0.70, 0.80), 65),
    ("差", (-0.01, 0.0, 0.40, 0.60), 40),
]


def ahp_fuzzy_scores(uni: pd.DataFrame, out_dir: str):
    """AHP 定权（学习行为/学习效果/互动参与三维度两两比较，evaluators 唯一实现）
    + 模糊综合评判 → 每位学习者学习质量量化综合评分（0-100）。"""
    a = ahp_core(AHP_MATRIX)
    w = np.array(a["weights"], dtype=float)
    cr = a["cr"]
    log(f"  AHP 权重: {dict(zip(AHP_DIMS, np.round(w, 3)))}  "
        f"一致性 CR={cr:.3f} ({'通过' if cr < 0.1 else '未通过，矩阵需调整'})")

    # 指标归一化（min-max，反向指标取反；缺失置 0.5）
    norm_cache = {}
    for dim, inds in AHP_INDICATORS.items():
        for ind in inds:
            if ind not in uni.columns:
                continue
            s = uni[ind].astype(float)
            lo, hi = s.min(), s.max()
            v = (s - lo) / (hi - lo + 1e-9)
            if ind in REVERSE_INDICATORS:
                v = 1 - v
            norm_cache[ind] = v.fillna(0.5)

    rows = []
    for i, sid in enumerate(uni["student_id"]):
        dim_scores, B = [], np.zeros(len(FUZZY_GRADES))
        for j, (dim, inds) in enumerate(AHP_INDICATORS.items()):
            vals = [norm_cache[ind].iloc[i] for ind in inds if ind in norm_cache]
            v = float(np.mean(vals)) if vals else 0.5
            dim_scores.append(v)
            r = np.array([_trap(v, *edges) for _, edges, _ in FUZZY_GRADES])
            r = r / (r.sum() + 1e-9)  # 隶属度行归一化
            B += w[j] * r
        score = float(B @ np.array([g for _, _, g in FUZZY_GRADES]))
        level = FUZZY_GRADES[int(np.argmax(B))][0]
        rows.append({"student_id": sid,
                     "dim_behavior": round(dim_scores[0], 3),
                     "dim_effect": round(dim_scores[1], 3),
                     "dim_interaction": round(dim_scores[2], 3),
                     "fuzzy_best_level": level,
                     "quality_score": round(score, 1)})
    qs = pd.DataFrame(rows)
    qs.to_csv(os.path.join(out_dir, "up_learning_quality_scores.csv"), index=False,
              encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(qs["quality_score"], bins=25, color=PALETTE[1], alpha=0.85,
            edgecolor="white")
    ax.axvline(qs["quality_score"].mean(), color=PALETTE[2], ls="--",
               label=f"均值 {qs['quality_score'].mean():.1f}")
    ax.set(xlabel="学习质量综合评分", ylabel="人数",
           title="AHP + 模糊综合评判 — 学习质量评分分布")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_quality_score_distribution.png"),
                bbox_inches="tight")
    plt.close()

    log(f"  学习质量评分: 均值={qs['quality_score'].mean():.1f}, "
        f"等级分布 {qs['fuzzy_best_level'].value_counts().to_dict()}")
    return qs, {"weights": dict(zip(AHP_DIMS, [float(x) for x in np.round(w, 3)])),
                "cr": round(cr, 3)}


def sd_simulation(out_dir: str, weeks: int = 52) -> dict:
    """系统动力学（SD）行为演化仿真。
    三条反馈回路：学习投入(强化回路)、资源供给(调节回路)、平台激励(时滞强化回路)。
    E=学习投入, R=资源供给质量, I=平台激励；欧拉法离散积分模拟不同平台优化策略。"""
    def simulate(invest, incentive_boost, decay_E, weeks):
        E, R, I = 0.30, 0.40, 0.20
        traj = [E]
        for _ in range(weeks):
            dE = 0.35 * (R - 0.25) + 0.45 * I - decay_E * E      # 投入回路 + 疲劳衰减
            dR = 0.5 * invest * (0.5 + 0.5 * E) - 0.15 * R       # 资源供给随投入加大
            dI = 0.4 * max(E - 0.5, 0) - 0.25 * I + incentive_boost  # 激励回路
            E = float(np.clip(E + dE, 0, 1))
            R = float(np.clip(R + dR, 0, 1))
            I = float(np.clip(I + dI, 0, 1))
            traj.append(E)
        return traj

    strategies = {
        "基线策略": dict(invest=0.15, incentive_boost=0.0, decay_E=0.25),
        "加强资源供给": dict(invest=0.45, incentive_boost=0.0, decay_E=0.25),
        "加强平台激励": dict(invest=0.15, incentive_boost=0.02, decay_E=0.25),
        "综合优化策略": dict(invest=0.45, incentive_boost=0.02, decay_E=0.15),
    }
    res = {}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (name, kw) in enumerate(strategies.items()):
        traj = simulate(weeks=weeks, **kw)
        res[name] = {"final_engagement": round(traj[-1], 3),
                     "growth_pct": round(100 * (traj[-1] / traj[0] - 1), 1)}
        ax.plot(range(weeks + 1), traj, color=PALETTE[i], lw=2, label=name)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set(xlabel="周", ylabel="学习者整体投入水平 E", title="系统动力学仿真 — 不同平台策略下行为长期演化")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "up_sd_simulation.png"), bbox_inches="tight")
    plt.close()
    pd.DataFrame({k: v["final_engagement"] for k, v in res.items()},
                 index=["final_engagement"]).T.to_csv(
        os.path.join(out_dir, "up_sd_simulation_results.csv"), encoding="utf-8-sig")
    best = max(res, key=lambda k: res[k]["final_engagement"])
    log(f"  SD 仿真: 52 周后最优策略 = {best}（投入水平 {res[best]['final_engagement']}）")
    return {"results": res, "best_strategy": best}


# ======================================================================
# 决策层：规则引擎队列 + 教师复核 + 内容推荐
# ======================================================================
def rule_queues(uni: pd.DataFrame) -> pd.DataFrame:
    """阈值规则 → support / enrichment / review 队列（dss 规则引擎思想）。"""
    rows = []
    med_score = uni["avg_quiz_score"].median()
    med_eng = uni[["total_clicks", "active_days"]].mean(axis=1).median()
    for _, r in uni.iterrows():
        score = r["avg_quiz_score"] if pd.notna(r["avg_quiz_score"]) else med_score
        eng = np.nanmean([r.get("total_clicks", np.nan), r.get("active_days", np.nan),
                          r.get("engagement_score", np.nan)])
        if pd.isna(eng):
            eng = med_eng
        evidence, action, queue = [], "monitor", "none"
        if score < med_score * 0.6:
            evidence.append(f"score_far_below_median({score:.1f}<{med_score * 0.6:.1f})")
            action, queue = "assign_mentor", "support"
        elif r.get("help_requests", 0) > uni["help_requests"].quantile(0.9):
            evidence.append("frequent_help_seeking")
            action, queue = "offer_guided_resources", "support"
        elif score > med_score * 1.3 and eng > med_eng:
            evidence.append(f"high_score_high_engagement({score:.1f})")
            action, queue = "offer_enrichment", "enrichment"
        elif r.get("lsa_pattern", "正常") in ("刷课", "跳步学习"):
            evidence.append(f"abnormal_behavior_pattern({r['lsa_pattern']})")
            action, queue = "teacher_review", "review"
        elif med_score * 0.8 <= score < med_score * 0.6 or \
                (score < med_score and eng < med_eng * 0.5):
            evidence.append("borderline_or_low_engagement")
            action, queue = "teacher_review", "review"
        rows.append({"student_id": r["student_id"], "persona": r["persona"],
                     "avg_quiz_score": round(score, 1), "queue": queue,
                     "suggested_action": action, "evidence": "; ".join(evidence) or "none"})
    return pd.DataFrame(rows)


def generate_recommendations(uni: pd.DataFrame) -> pd.DataFrame:
    """分画像内容推荐（Learner-Persona + ai-personalized-learning 思路）。"""
    recs = []
    for persona, group in uni.groupby("persona"):
        base = persona.split("（")[0]  # 变体簇复用基础画像的描述与推荐
        recs.append({
            "persona": persona, "n_students": len(group),
            "share_pct": round(100 * len(group) / len(uni), 1),
            "description": PERSONA_DESC.get(base, ""),
            "content_recommendation": CONTENT_REC.get(
                base, "按簇中心特征定制内容策略"),
        })
    return pd.DataFrame(recs)


def generate_optimization_suggestions(uni, recs, scene_tbl, corr_top, sd_info,
                                      early_warning, out_dir) -> pd.DataFrame:
    """基于全部分析结果输出可落地的平台优化策略。"""
    rows = []
    # 1) 画像差异化推送与干预
    for _, r in recs.iterrows():
        base = r["persona"].split("（")[0]
        extra_map = {
            "风险流失型": "推送轻量化短视频学习内容 + 每日微任务签到，提升留存率",
            "考前冲刺型": "考前冲刺包 + 间隔重复记忆卡片推送",
            "引导依赖型": "微课 + 即时反馈测验组合推送，逐步过渡到自主学习",
            "自主深度型": "开放项目与进阶资源优先推送，邀请担任社区助教",
        }
        rows.append({"策略类型": "画像差异化运营", "对象": r["persona"],
                     "建议": extra_map.get(base, str(r["content_recommendation"]))})
    # 2) 区域运营策略（完成率最低的地区补齐资源）
    if "region" in uni.columns:
        rg = uni.groupby("region").agg(完成率=("completion_rate", "mean"),
                                       成绩=("final_grade", "mean"),
                                       人数=("student_id", "count")).round(3)
        worst = rg.sort_values("完成率").head(2)
        for region, r in worst.iterrows():
            rows.append({"策略类型": "分区域运营优化", "对象": f"{region}（完成率 {r['完成率']:.1%}）",
                         "建议": "联合本地高校共建本地化课程内容，增设线下学习中心与带宽/终端支持，"
                                 "补齐薄弱地区资源供给短板"})
    # 3) 早期学业预警机制
    n_high = (early_warning["risk_level"] == "高").sum()
    rows.append({"策略类型": "早期学业预警",
                 "对象": f"{n_high} 名高风险学生（结课概率<0.2）",
                 "建议": "当学生连续 7 天未登录、视频完播率骤降、作业逾期时自动触发预警，"
                         "辅导员 48 小时内一对一介入；预警名单见 up_early_warning_students.csv"})
    # 4) 场景化资源优化设计
    if not scene_tbl.empty:
        for scene, r in scene_tbl.iterrows():
            if scene == "短视频":
                rows.append({"策略类型": "场景化资源优化", "对象": f"{scene}（完播率 {r['完播率']:.0%}）",
                             "建议": "在短视频中增加知识点锚点与章节跳转卡片，配套 1 分钟随堂小测，"
                                     "提升碎片化学习的知识留存率"})
            elif scene == "慕课长视频":
                rows.append({"策略类型": "场景化资源优化", "对象": f"{scene}（拖拽 {r['平均拖拽次数']:.1f} 次）",
                             "建议": "长视频拆分为 8-12 分钟微模块，前置学习目标提示，"
                                     "高拖拽片段重新剪辑压缩"})
            else:
                rows.append({"策略类型": "场景化资源优化", "对象": f"{scene}（完播率 {r['完播率']:.0%}）",
                             "建议": "保持微课节奏优势，增加弹题互动与字幕检索功能"})
    # 5) 平台策略（SD 仿真结论）
    rows.append({"策略类型": "平台长期策略（SD 仿真）", "对象": sd_info["best_strategy"],
                 "建议": f"仿真显示 52 周后「{sd_info['best_strategy']}」下学习者整体投入水平最高"
                         f"（{sd_info['results'][sd_info['best_strategy']]['final_engagement']}），"
                         "建议以资源供给 + 激励机制双轮驱动并降低学习疲劳衰减"})
    sug = pd.DataFrame(rows)
    sug.to_csv(os.path.join(out_dir, "up_optimization_suggestions.csv"), index=False,
               encoding="utf-8-sig")
    log(f"  优化建议: 输出 {len(sug)} 条落地策略")
    return sug


# ======================================================================
# 输出层：报告
# ======================================================================
def write_report(uni, queues, recs, metrics, out_dir, extras=None) -> str:
    extras = extras or {}
    lines = [
        "# 学习者画像分析报告（统一管线 up_）",
        f"生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 数据概览",
        f"- 学生总数: {len(uni)}，数据来源: {uni['source'].value_counts().to_dict()}",
        f"- 聚类数 K = {metrics['k']}，Silhouette = {metrics['silhouette']:.3f}，"
        f"计算后端 = {metrics.get('backend', '-')}",
        "",
        "## 主成分降维（PCA）",
        f"- 主成分命名: {extras.get('pc_names', [])}",
        f"- 方差解释率: {extras.get('pc_var', [])}",
        "",
        "## 行为模式挖掘（SNA + LSA）",
        f"- SNA: PageRank Top 资源 = {extras.get('sna_top_resources', [])}",
        f"- LSA 行为模式分布: {extras.get('lsa_patterns', {})}",
        f"- LSA Top 显著转换: {extras.get('lsa_top_transitions', [])}",
        "",
        "## 预测性能",
        f"- LogisticRegression 成功预测: acc={metrics['logreg_acc']:.3f}, "
        f"AUC={metrics['logreg_auc']:.3f}",
        f"- RF 学业表现预测: 最终成绩 R²={metrics.get('rf_grade_r2', float('nan')):.3f}, "
        f"完成率 R²={metrics.get('rf_completion_r2', float('nan')):.3f}, "
        f"完成判定 acc={metrics.get('rf_completion_acc', float('nan')):.3f}",
        f"- RandomForest 风险预测 5-fold AUC: 全特征={metrics['rf_auc_full']:.3f}, "
        f"早期特征={metrics['rf_auc_early']:.3f}",
        f"- 行为特征预测画像标签 acc = {metrics['persona_pred_acc']:.3f}",
        f"- GM(1,1) 日活跃趋势: {extras.get('gm_trends', {})}",
        "",
        "## 学习质量评价（AHP + 模糊综合评判）",
        f"- AHP 权重: {extras.get('ahp_weights', {})}, 一致性 CR = {extras.get('ahp_cr', '-')}",
        f"- 评分均值: {extras.get('quality_mean', '-')}, "
        f"等级分布: {extras.get('quality_levels', {})}",
        "",
        "## 学习者画像",
    ]
    # 扩展分析摘要（五大模块结论）
    ex = extras or {}
    if ex:
        lines += ["", "## 扩展分析摘要", ""]
        if ex.get("core_groups"):
            lines.append(f"- 核心学习者群体: {ex['core_groups']}")
        if ex.get("time_patterns"):
            lines.append(f"- 时间行为模式分布: {ex['time_patterns']}")
        if ex.get("apriori_top"):
            for r in ex["apriori_top"]:
                lines.append(f"- 关联规则: {r['antecedent']} → {r['consequent']} "
                             f"(support={r['support']}, conf={r['confidence']}, lift={r['lift']})")
        if ex.get("short_only_pct") is not None:
            lines.append(f"- 只看短视频跳过完整慕课的学生占比: {ex['short_only_pct']}%")
        if ex.get("scene"):
            lines.append(f"- 场景行为差异: {ex['scene']}")
        if ex.get("corr_top") is not None and len(ex["corr_top"]):
            cs = ", ".join(f"{r['indicator']}(r={r['pearson_r']})"
                           for r in ex["corr_top"])
            lines.append(f"- 与成绩最相关的行为因素: {cs}")
        if ex.get("tier_lift") is not None:
            lines.append(f"- 投入分层提升幅度: 高投入组较低投入组 {ex['tier_lift']}")
        if ex.get("model_ext"):
            m = ex["model_ext"]
            lines.append(f"- 学业表现预测: 最佳模型 {m.get('best_model')} "
                         f"acc={m.get('RandomForest_grade_acc', 0):.3f}/{m.get('GradientBoosting_grade_acc', 0):.3f}"
                         f"（RF/GBDT），Top5 关键行为: {list(m.get('top5_features', {}))}")
            lines.append(f"- 早期预警: {ex.get('early_warning_count', '-')} 名高风险学生")
        if ex.get("sd_best"):
            lines.append(f"- SD 仿真最优策略: {ex['sd_best']}（52 周投入水平 "
                         f"{ex.get('sd_final', '-')})")
        if ex.get("n_suggestions"):
            lines.append(f"- 优化建议: {ex['n_suggestions']} 条（详见 up_optimization_suggestions.csv）")
    lines += df_markdown(recs)
    lines += ["", "## 干预队列统计", ""]
    qcount = queues["queue"].value_counts()
    lines += [f"- {q}: {n} 人" for q, n in qcount.items()]
    lines += ["", "## 需要教师复核的学生（review/support 队列示例，前 20）", ""]
    sub = queues[queues["queue"].isin(["support", "review"])].head(20)
    lines += df_markdown(sub)
    report = "\n".join(lines)
    with open(os.path.join(out_dir, "up_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    return report
