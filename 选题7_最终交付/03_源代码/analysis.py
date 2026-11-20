# -*- coding: utf-8 -*-
"""
analysis.py —— 学习者行为分析核心流水线
《系统工程原理》实践项目 选题7

对应报告章节（本交付包为 9 章结构，按系统工程方法论单独成章）：
  第3章 支撑数据与分析对象      -> load_clean() / quality_profile() / describe() / corr_matrix()
  第4章 系统建模（目标树 + ISM 递阶结构）-> ism()
  第5章 系统分析（K-means 画像 + 六维画像）-> cluster()
       ※ 画像生成属"系统分析"（自下而上从数据归纳群体）；不是"系统建模"。
         建模章只放目标树与 ISM 这类因素结构模型。画像在决策章被复用为候选单元。
  第6章 系统预测（学业风险）    -> predict()
  第7章 系统评价（组合赋权）    -> ahp() / entropy_weight() / topsis() / evaluate()
  第8章 系统决策（资源配置）    -> decide() / budget_sweep()
  第9章 可验证成果             -> app.py（交互验证 / 改参重跑）

霍尔逻辑维七步 ↔ 本模块：明确问题(1.1-1.2)→选择目标(4.1)→系统综合(4.2/8.1)
→系统分析(第3、5章)→方案优化(第6-8章)→作出决策(8.4)→付诸实施(第9章)，详见报告表 2。

所有随机过程均固定种子，任一函数可单独调用复现。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

from common import ClusterEngine, ensure_device          # GPU 聚类引擎（合并新增）
from evaluators import ahp as _ahp_core                  # 评价唯一实现（合并去重）
from evaluators import entropy_weight_courses as _entropy_core
from evaluators import topsis_closeness as _topsis_core

SEED = 20260101
DATA_SOURCE = "OULAD 真实学习者主表 real_learners.csv（20105 条 / 18634 人 / 13 门课）"

# 画像 / 描述用的 9 维行为特征（全观察窗口）
BEHAV_FEATURES = [
    "visit_count", "active_days", "study_hours", "avg_session_min",
    "forum_posts", "first_active_lag", "video_rate", "assign_rate", "quiz_avg",
    "course_num",
]
# 预测专用特征：仅前期行为 + 人口属性，严格避免同期测评结果造成的标签泄漏。
# 标签 certificate 是学期末真实结果；quiz_avg/final_grade/assign_rate 与它同期同生，
# 若入模等于"用结果预测结果"。前期窗口取到第 4 周，先于结果发生。
PREDICT_FEATURES = [
    "early_w2_clicks", "early_w2_days", "early_w4_clicks", "early_w4_days",
    "visit_count", "active_days", "forum_posts", "first_active_lag",
    "video_rate", "age", "prev_attempts",
]
FEATURE_CN = {
    "visit_count": "访问次数",
    "active_days": "活跃天数",
    "study_hours": "学习时长(等效)",
    "avg_session_min": "单次时长(等效)",
    "forum_posts": "论坛互动",
    "first_active_lag": "首次参与滞后",
    "video_rate": "资源覆盖度",
    "assign_rate": "作业提交率",
    "quiz_avg": "测验均分",
    "course_num": "选课门数",
    "early_w2_clicks": "前2周点击",
    "early_w2_days": "前2周活跃天",
    "early_w4_clicks": "前4周点击",
    "early_w4_days": "前4周活跃天",
    "age": "年龄",
    "prev_attempts": "既往重修次数",
}
CLUSTER_FEATURES = ["visit_count", "active_days", "study_hours", "avg_session_min",
                    "forum_posts", "first_active_lag", "video_rate", "assign_rate", "quiz_avg"]

# 4 类典型画像原型（各维度取值以标准化后的整体分布为参照，0 表示处于平均水平）
PROTOTYPES = {
    "自主达标型": dict(visit_count=0.5, active_days=0.9, study_hours=1.1, avg_session_min=0.7,
                  forum_posts=-0.2, first_active_lag=-0.5, video_rate=0.8, assign_rate=0.7, quiz_avg=0.6),
    "社交深耕型": dict(visit_count=0.8, active_days=0.8, study_hours=0.9, avg_session_min=0.4,
                  forum_posts=1.7, first_active_lag=-0.3, video_rate=0.4, assign_rate=0.4, quiz_avg=0.3),
    "应试突击型": dict(visit_count=-0.2, active_days=-0.4, study_hours=-0.2, avg_session_min=-0.1,
                  forum_posts=-0.2, first_active_lag=1.1, video_rate=-0.7, assign_rate=0.8, quiz_avg=0.0),
    "低投入游离型": dict(visit_count=-0.9, active_days=-0.9, study_hours=-0.9, avg_session_min=-0.8,
                   forum_posts=-0.5, first_active_lag=0.5, video_rate=-0.8, assign_rate=-0.9, quiz_avg=-0.9),
}

# 扩展原型：覆盖 k=2~8 的全部命名需求。行为谱系按"投入度"从低到高排列，
# 新增 4 个端点/细分锚点，使任意 k 都能挑出恰好 k 个语义 distinct 的原型做满秩匹配。
# k=4 仍严格使用上面的 PROTOTYPES 四原型，保证报告主线的命名与数值完全不变。
EXTRA_PROTOTYPES = {
    "零活跃沉默型": dict(visit_count=-1.3, active_days=-1.3, study_hours=-1.3, avg_session_min=-1.3,
                   forum_posts=-0.9, first_active_lag=0.9, video_rate=-1.1, assign_rate=-1.2, quiz_avg=-1.1),
    "稳步进阶型": dict(visit_count=0.2, active_days=0.5, study_hours=0.5, avg_session_min=0.3,
                  forum_posts=0.3, first_active_lag=-0.2, video_rate=0.5, assign_rate=0.6, quiz_avg=0.4),
    "深度钻研型": dict(visit_count=1.2, active_days=1.2, study_hours=1.5, avg_session_min=1.0,
                  forum_posts=2.3, first_active_lag=-0.5, video_rate=0.6, assign_rate=0.7, quiz_avg=0.7),
    "高频沉浸型": dict(visit_count=1.5, active_days=1.5, study_hours=2.0, avg_session_min=1.3,
                  forum_posts=3.0, first_active_lag=-0.7, video_rate=0.7, assign_rate=0.8, quiz_avg=0.9),
}
ALL_PROTOTYPES = {**PROTOTYPES, **EXTRA_PROTOTYPES}

# 每个 k 使用的原型名集合（顺序无关，匈牙利匹配负责一一对应）。
# 集合元素数恒等于 k → linear_sum_assignment 满秩，每个簇获得唯一命名，无需序号后缀。
# k=4 固定为报告现有四原型，其余 k 沿投入度谱系均匀分布锚点。
K_PROTOTYPE_SETS = {
    2: ["低投入游离型", "社交深耕型"],
    3: ["低投入游离型", "应试突击型", "社交深耕型"],
    4: list(PROTOTYPES),
    5: ["零活跃沉默型", "低投入游离型", "应试突击型", "自主达标型", "社交深耕型"],
    6: ["零活跃沉默型", "低投入游离型", "应试突击型", "自主达标型", "社交深耕型", "深度钻研型"],
    7: ["零活跃沉默型", "低投入游离型", "应试突击型", "自主达标型", "稳步进阶型", "社交深耕型", "深度钻研型"],
    8: ["零活跃沉默型", "低投入游离型", "应试突击型", "自主达标型", "稳步进阶型", "社交深耕型", "深度钻研型", "高频沉浸型"],
}

# ----------------------------------------------------------------------------
# 3 数据清洗
# ----------------------------------------------------------------------------

def load_clean(path="data/real_learners.csv"):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    log = {"raw_rows": int(len(raw))}
    log["raw_missing"] = int(raw.isna().sum().sum())
    log["source"] = "OULAD 真实公开数据（非仿真）"

    df = raw.copy()
    # 3.1 课程号统一大写
    df["course_id"] = df["course_id"].str.upper()
    # 3.2 完全重复记录去重
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "record_id"])
    log["after_dedup"] = int(len(df))
    # 3.3 等效时长异常值：真实数据为点击折算值，极个别超大点击者做上限截断
    p99 = df["study_hours"].quantile(0.99)
    bad = df["study_hours"] > p99 * 5
    log["sec_bug_fixed"] = int(bad.sum())
    df.loc[bad, "study_hours"] = p99
    # 3.4 缺失值：计数型用 0 填补（未产生即未发生），比率型用课程中位数填补
    log["missing_forum"] = int(df["forum_posts"].isna().sum())
    log["missing_session"] = int(df["avg_session_min"].isna().sum())
    df["forum_posts"] = df["forum_posts"].fillna(0)
    df["avg_session_min"] = df["avg_session_min"].fillna(
        df.groupby("course_id")["avg_session_min"].transform("median"))
    # 3.5 逻辑一致性校验
    log["days_gt_visits"] = int((df["active_days"] > df["visit_count"]).sum())
    df.loc[df["active_days"] > df["visit_count"], "active_days"] = df["visit_count"]
    # 3.6 极值截断（1%/99% 分位）
    for col in ["study_hours", "avg_session_min", "visit_count"]:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)
    # 3.7 学习者层面选课门数（真实：同一学习者注册的不同课程数）
    df["course_num"] = df.groupby("learner_id")["course_id"].transform("nunique")
    # 3.8 预测特征兜底：早期预警列若源表未提供则补 0，既往重修次数补 0
    for col in ["early_w2_clicks", "early_w2_days", "early_w4_clicks", "early_w4_days"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)
    if "prev_attempts" not in df.columns:
        df["prev_attempts"] = 0
    df["prev_attempts"] = df["prev_attempts"].fillna(0)
    df["region"] = df["region"].astype("category")
    df["education"] = df["education"].astype("category")

    log["clean_rows"] = int(len(df))
    log["clean_missing"] = int(df.isna().sum().sum())
    log["learners"] = int(df["learner_id"].nunique())
    log["courses"] = int(df["course_id"].nunique())
    log["cert_rate"] = round(float(df["certificate"].mean()), 4)
    log["female_rate"] = round(float((df["gender"] == "女").mean()), 4)
    log["median_age"] = float(df["age"].median())
    log["mean_hours"] = round(float(df["study_hours"].mean()), 2)
    log["mean_days"] = round(float(df["active_days"].mean()), 2)
    log["edu_dist"] = {str(k): round(float(v), 4) for k, v in df["education"].value_counts(normalize=True).items()}
    log["region_dist"] = {str(k): round(float(v), 4) for k, v in df["region"].value_counts(normalize=True).items()}
    log["disc_dist"] = {str(k): round(float(v), 4) for k, v in df["discipline"].value_counts(normalize=True).items()}
    return df, log


def quality_profile(df):
    """数据质量六维打分（完整性/一致性/唯一性/有效性/时效性/代表性），0~100。"""
    n = len(df)
    completeness = 100 * (1 - df.isna().sum().sum() / (n * df.shape[1]))
    uniqueness = 100 * (1 - df.duplicated(subset=[c for c in df.columns if c != "record_id"]).mean())
    validity = 100 * float(((df["quiz_avg"].between(0, 100)) &
                            (df["video_rate"].between(0, 1)) &
                            (df["assign_rate"].between(0, 1)) &
                            (df["age"].between(16, 70))).mean())
    consistency = 100 * float((df["active_days"] <= df["visit_count"]).mean())
    timeliness = 100.0        # 观察窗口统一为 120 天，口径内外一致
    representativeness = 100 * float(min(1.0, df["course_id"].nunique() / 10))
    return {
        "完整性": round(completeness, 2), "一致性": round(consistency, 2),
        "唯一性": round(uniqueness, 2), "有效性": round(validity, 2),
        "时效性": round(timeliness, 2), "代表性": round(representativeness, 2),
    }


def describe(df):
    cols = ["visit_count", "active_days", "study_hours", "avg_session_min",
            "forum_posts", "first_active_lag", "video_rate", "assign_rate",
            "quiz_avg", "final_grade"]
    d = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    d["cv"] = d["std"] / d["mean"]
    return d.round(3).reset_index().rename(columns={"index": "变量"}).to_dict("records")


def corr_matrix(df):
    return df[CLUSTER_FEATURES + ["final_grade"]].corr().round(3)


# ----------------------------------------------------------------------------
# 5 行为模式识别
# ----------------------------------------------------------------------------

def cluster(df, k=4, k_range=range(2, 9)):
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES].values)
    # 聚类统一走 ClusterEngine（合并新增）：CPU 后端 = 原 sklearn KMeans 逐位一致；
    # --device gpu 时切换 PyTorch CUDA / cuML 后端加速 KMeans 与轮廓系数。
    engine = ClusterEngine(ensure_device())
    scan = []
    for kk in k_range:
        r = engine.fit(X, int(kk), seed=SEED, n_init=10)
        scan.append({"k": kk,
                     "inertia": round(float(r["inertia"]), 1),
                     "silhouette": round(float(r["sil"]), 4)})

    r = engine.fit(X, k, seed=SEED, n_init=20)
    labels = r["labels"]
    cent = r["centers"]
    df = df.copy()
    df["cluster"] = labels
    sil = float(r["sil"])

    # 轮廓式命名：按 学习时长 与 论坛发帖 中位数划分四象限
    prof = df.groupby("cluster").agg(
        n=("record_id", "size"),
        hours=("study_hours", "mean"),
        days=("active_days", "mean"),
        visits=("visit_count", "mean"),
        sess=("avg_session_min", "mean"),
        forum=("forum_posts", "mean"),
        video=("video_rate", "mean"),
        assign=("assign_rate", "mean"),
        quiz=("quiz_avg", "mean"),
        lag=("first_active_lag", "mean"),
        cert=("certificate", "mean"),
        grade=("final_grade", "mean"),
    ).round(3)
    prof["share"] = (prof["n"] / prof["n"].sum()).round(3)

    # 原型匹配命名：按当前 k 选取恰好 k 个语义不同的行为原型，把聚类质心
    # （标准化空间）与它们做最小距离一对一匹配（匈牙利算法）。因原型数=簇数，
    # 匹配满秩、每个簇获得唯一命名，无需“加序号后缀”兜底。
    # k=4 使用报告现有四原型，命名与数值口径不变；k>4 启用扩展谱系原型。
    proto_names = list(K_PROTOTYPE_SETS.get(k, list(PROTOTYPES)))
    proto = np.array([[ALL_PROTOTYPES[nm][f] for f in CLUSTER_FEATURES] for nm in proto_names])
    dist = ((cent[:, None, :] - proto[None, :, :]) ** 2).sum(-1) ** 0.5
    row, col = linear_sum_assignment(dist)
    names = {int(r): proto_names[c] for r, c in zip(row, col)}
    prof["name"] = [names[int(c)] for c in prof.index]
    prof["proto_dist"] = [round(float(dist[int(c), proto_names.index(names[int(c)])]), 3)
                          for c in prof.index]
    prof = prof.sort_values("hours", ascending=False)

    ari = None
    if "_latent_class" in df.columns:
        ari = float(adjusted_rand_score(df["_latent_class"], labels))

    # 各群体的结构占比：教育/地区（用于画像描述）
    comp = {}
    for c in prof.index:
        sub = df[df["cluster"] == c]
        comp[int(c)] = {
            "edu_top": sub["education"].value_counts(normalize=True).round(3).head(3).to_dict(),
            "region_top": sub["region"].value_counts(normalize=True).round(3).head(3).to_dict(),
            "female": round(float((sub["gender"] == "女").mean()), 3),
        }
    comp = {str(k2): {kk: {str(a): float(b) for a, b in vv.items()} if isinstance(vv, dict) else vv
                      for kk, vv in v.items()} for k2, v in comp.items()}

    return {"k": k, "silhouette": round(sil, 4), "scan": scan,
            "labels": labels, "df": df,
            "profile": prof, "comp": comp, "ari": ari,
            "profile_records": prof.reset_index().to_dict("records")}


# ----------------------------------------------------------------------------
# 6 学习成效预测
# ----------------------------------------------------------------------------

def predict(df, test_size=0.25):
    # 无泄漏口径：只用前期行为 + 人口属性预测学期末真实结果。
    # quiz_avg / final_grade / assign_rate 与标签同期同生，禁止入模。
    feats = PREDICT_FEATURES
    X = df[feats].values
    y = df["certificate"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    models = {
        "Logistic回归": (LogisticRegression(max_iter=2000, class_weight="balanced",
                                            random_state=SEED), True),
        "随机森林": (RandomForestClassifier(n_estimators=300, max_depth=8,
                                             min_samples_leaf=5, random_state=SEED,
                                             n_jobs=-1), False),
        "梯度提升树": (GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                   learning_rate=0.06, random_state=SEED), False),
    }
    res, fitted = {}, {}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, (m, need_scale) in models.items():
        A, B = (Xtr_s, Xte_s) if need_scale else (Xtr, Xte)
        m.fit(A, ytr)
        p = m.predict_proba(B)[:, 1]
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        res[name] = {
            "auc": round(float(roc_auc_score(yte, p)), 4),
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
            "cv_auc": round(float(cross_val_score(m, A, ytr, cv=cv,
                                                  scoring="roc_auc").mean()), 4),
            "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }
        fitted[name] = (m, p, need_scale)

    # 主模型：Logistic 结果用于解释（优势比）
    lr, p_lr, _ = fitted["Logistic回归"]
    odds = pd.DataFrame({"feature": feats,
                         "coef": lr.coef_[0].round(4),
                         "odds_ratio": np.exp(lr.coef_[0]).round(3)})
    odds["feature_cn"] = odds["feature"].map(FEATURE_CN)
    odds = odds.sort_values("coef", ascending=False).to_dict("records")

    rf = fitted["随机森林"][0]
    imp = pd.DataFrame({"feature": feats, "importance": rf.feature_importances_.round(4)})
    imp["feature_cn"] = imp["feature"].map(FEATURE_CN)
    imp = imp.sort_values("importance", ascending=False).to_dict("records")

    fpr, tpr, _ = roc_curve(yte, p_lr)
    return {"metrics": res, "odds": odds, "importance": imp,
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "test_n": int(len(yte)), "pos_rate": round(float(y.mean()), 4),
            "fitted": fitted}


# ----------------------------------------------------------------------------
# 7 课程运营评价：AHP + 熵权 + TOPSIS
# ----------------------------------------------------------------------------

CRITERIA = [
    ("C1", "资源完成度", "video_rate"),
    ("C2", "学习投入", "study_hours"),
    ("C3", "互动参与", "forum_posts"),
    ("C4", "成绩表现", "quiz_avg"),
    ("C5", "持续参与", "active_days"),
    ("C6", "达标转化", "certificate"),
]

# 专家判断矩阵（1~9 标度，行/列顺序同 CRITERIA）
AHP_MATRIX = np.array([
    [1, 1, 2, 2, 3, 3],
    [1, 1, 2, 2, 3, 3],
    [1 / 2, 1 / 2, 1, 1, 2, 2],
    [1 / 2, 1 / 2, 1, 1, 2, 2],
    [1 / 3, 1 / 3, 1 / 2, 1 / 2, 1, 1],
    [1 / 3, 1 / 3, 1 / 2, 1 / 2, 1, 1],
], dtype=float)

RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


# 评价方法唯一实现已抽取至 evaluators.py（合并去重：与 multimodal.py 共用一份）
def ahp(matrix=AHP_MATRIX):
    """AHP 求权：主特征向量法 + CR 一致性检验（唯一实现见 evaluators.py）。"""
    r = _ahp_core(matrix)
    return {"weights": [round(float(x), 4) for x in r["weights"]],
            "lambda_max": round(r["lambda_max"], 4), "ci": round(r["ci"], 4),
            "cr": round(r["cr"], 4), "n": r["n"]}


def entropy_weight(df):
    """熵权法（课程表口径）：按 course_id 聚组取均值 → 极差标准化 → 熵权。"""
    return _entropy_core(df, [col for _, _, col in CRITERIA], nd=4)


def topsis(df, w):
    """TOPSIS 课程质量贴近度排序（矩阵计算委托 evaluators.topsis_closeness）。"""
    courses = sorted(df["course_id"].unique())
    X = np.array([[df[df["course_id"] == c][col].mean() for _, _, col in CRITERIA]
                  for c in courses], dtype=float)
    close = _topsis_core(X, w)
    names = {c: df[df["course_id"] == c]["course_name"].iloc[0] for c in courses}
    rows = [{"course_id": c, "course_name": names[c], "closeness": round(float(cl), 4),
             "rank": 0, **{cri[0]: round(float(x), 3) for cri, x in zip(CRITERIA, xi)}}
            for c, cl, xi in zip(courses, close, X)]
    rows.sort(key=lambda r: -r["closeness"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def evaluate(df, ahp_w=None):
    a = ahp() if ahp_w is None else {"weights": list(ahp_w), "cr": None}
    ent = entropy_weight(df)
    w = 0.5 * np.array(a["weights"]) + 0.5 * np.array(ent["weights"])
    w = w / w.sum()
    rows = topsis(df, w)
    return {"ahp": a, "entropy": ent, "combined": w.round(4).tolist(), "rank": rows,
            "criteria": [{"code": c, "name": n, "field": f} for c, n, f in CRITERIA]}


# ----------------------------------------------------------------------------
# 8 干预资源配置：0-1 整数规划
# ----------------------------------------------------------------------------

# 每个细分群体可选的干预包：成本(元/人)、达标率绝对提升(百分点)
PACKAGES = {
    # 依据平台运营经验与文献设定：成本为人均投入（元），提升为达标率绝对提升（百分点）。
    # 低投入游离型唤醒成本高、转化率低（“高成本—低弹性”），
    # 应试突击型已具备参与基础，单位投入的边际转化最高（“低成本—高弹性”）。
    "自主达标型": [(6, 1.0), (22, 3.0)],
    "社交深耕型": [(10, 2.5), (30, 6.5)],
    "应试突击型": [(12, 4.5), (34, 11.0)],
    "低投入游离型": [(18, 1.5), (55, 4.0)],
    # 扩展画像（k>4 时启用）：按投入度谱系继承相邻原型的高/低弹性假设。
    "零活跃沉默型": [(20, 1.0), (58, 3.0)],   # 唤醒最难，转化最低
    "稳步进阶型": [(7, 1.2), (24, 3.6)],      # 与自主达标型同属低成本高弹性
    "深度钻研型": [(9, 2.8), (28, 7.0)],      # 接近社交深耕型
    "高频沉浸型": [(8, 3.0), (26, 7.5)],      # 投入最高，边际提升空间有限
}


def decide(cluster_res, budget_ratio=0.45):
    """在给定预算下，为各细分群体选择干预强度，使预期新增达标人数最大。

    决策变量 x_j ∈ {0,1}：是否对第 j 个（群体, 强度）组合投放干预。
    目标  max Σ n_s·Δ_s·x_j
    约束  Σ c_j·x_j ≤ B                （预算）
          Σ_{j∈s} x_j ≤ 1             （同一群体只选一种强度）
    """
    prof = cluster_res["profile"].reset_index()
    segs = [{"name": r["name"], "n": int(r["n"]), "cert": float(r["cert"]),
             "cluster": int(r["cluster"])} for _, r in prof.iterrows()]
    extra = []
    for sgn in segs:
        key = sgn["name"].split("型")[0] + "型"
        pk = PACKAGES.get(key, PACKAGES["低投入游离型"])
        for lvl, (cost, uplift) in enumerate(pk):
            extra.append({"seg": sgn["name"], "level": "强化包" if lvl else "基础包",
                          "cost_per": cost, "uplift": uplift,
                          "total_cost": cost * sgn["n"],
                          "gain": sgn["n"] * uplift / 100.0})
    ex = pd.DataFrame(extra)
    total_budget = float(ex["total_cost"].sum()) * budget_ratio
    n_pkg = len(ex)
    cons = [LinearConstraint(ex["total_cost"].values.reshape(1, -1), -np.inf, total_budget)]
    seg_names = list(dict.fromkeys(ex["seg"]))
    M = np.zeros((len(seg_names), n_pkg))
    for i, sg in enumerate(seg_names):
        M[i] = (ex["seg"] == sg).astype(float).values
    cons.append(LinearConstraint(M, -np.inf, 1))
    sol = milp(c=-ex["gain"].values, constraints=cons, integrality=np.ones(n_pkg),
               bounds=Bounds(0, 1), options={"time_limit": 30})
    x = np.round(sol.x).astype(int) if sol.x is not None else np.zeros(n_pkg, dtype=int)
    ex["chosen"] = x
    opt_gain = float(ex.loc[ex["chosen"] == 1, "gain"].sum())
    opt_cost = float(ex.loc[ex["chosen"] == 1, "total_cost"].sum())

    # 对照 1：经验法——按群体达标率由低到高“优先帮扶最弱”，先强化包后基础包
    order = [sg["name"] for sg in sorted(segs, key=lambda d: d["cert"])]
    used, heur_gain, picked = 0.0, 0.0, []
    for name in order:
        cand = ex[ex["seg"] == name].sort_values("total_cost", ascending=False)
        for _, r in cand.iterrows():
            if used + r["total_cost"] <= total_budget:
                used += r["total_cost"]
                heur_gain += r["gain"]
                picked.append({"seg": name, "level": r["level"],
                               "total_cost": float(r["total_cost"]), "gain": float(r["gain"])})
                break

    # 对照 2：普惠法——预算按人数平摊，全体统一投放最接近的基础包强度
    base_sum = float(ex[ex["level"] == "基础包"]["total_cost"].sum())
    base_gain = float(ex[ex["level"] == "基础包"]["gain"].sum())
    if total_budget >= base_sum:
        uni_gain, uni_note = base_gain, "预算可覆盖全部学习者基础包"
    else:
        ratio = total_budget / base_sum
        uni_gain, uni_note = base_gain * ratio, f"人均预算为基准的{ratio:.0%}，按比例折算效果"

    return {
        "budget_ratio": budget_ratio,
        "budget": round(total_budget, 0),
        "total_cost_all": round(float(ex["total_cost"].sum()), 0),
        "options": ex.round(2).to_dict("records"),
        "chosen": ex[ex["chosen"] == 1][["seg", "level", "cost_per", "total_cost", "uplift", "gain"]]
                 .round(2).to_dict("records"),
        "opt_gain": round(opt_gain, 1),
        "opt_cost": round(opt_cost, 0),
        "heur_gain": round(heur_gain, 1),
        "heur_picked": picked,
        "uniform_gain": round(uni_gain, 1),
        "uniform_note": uni_note,
        "status": "optimal" if sol.status == 0 else str(sol.status),
        "segments": segs,
    }


def budget_sweep(cluster_res, ratios):
    out = []
    for r in ratios:
        d = decide(cluster_res, r)
        out.append({"ratio": round(r, 2), "budget": d["budget"],
                    "opt": d["opt_gain"], "heur": d["heur_gain"],
                    "uniform": d["uniform_gain"]})
    return out


# ----------------------------------------------------------------------------
# 附录 A 影响因素 ISM
# ----------------------------------------------------------------------------

ISM_FACTORS = [
    ("S1", "平台资源与功能质量"),
    ("S2", "课程难度与内容设计"),
    ("S3", "学习支持服务"),
    ("S4", "先备知识与自我效能"),
    ("S5", "学习动机与目标承诺"),
    ("S6", "时间投入与学习节律"),
    ("S7", "交互参与（论坛/作业）"),
    ("S8", "学习成效（达标/认证）"),
]
# 直接影响关系：Si -> Sj
ISM_REL = [("S1", "S6"), ("S1", "S7"), ("S1", "S8"),
           ("S2", "S5"), ("S2", "S6"), ("S2", "S8"),
           ("S3", "S6"), ("S3", "S7"),
           ("S4", "S5"), ("S4", "S6"),
           ("S5", "S6"), ("S5", "S7"),
           ("S6", "S7"), ("S6", "S8"),
           ("S7", "S8")]


def ism():
    ids = [f[0] for f in ISM_FACTORS]
    n = len(ids)
    A = np.eye(n, dtype=int)
    for a, b in ISM_REL:
        A[ids.index(a), ids.index(b)] = 1
    R = A.copy()
    while True:
        R2 = ((R @ R) > 0).astype(int)
        if np.array_equal(R2, R):
            break
        R = R2
    remaining = list(range(n))
    levels, level_map = [], {}
    while remaining:
        cur = []
        for i in remaining:
            reach = {j for j in remaining if R[i, j]}
            ante = {j for j in remaining if R[j, i]}
            if reach == reach & ante:
                cur.append(i)
        if not cur:
            cur = remaining[:]
        levels.append(cur)
        for i in cur:
            level_map[ids[i]] = len(levels)
        remaining = [i for i in remaining if i not in cur]
    micmac = [{"id": ids[i], "name": ISM_FACTORS[i][1], "level": level_map[ids[i]],
               "driving": int(R[i].sum() - 1), "dependence": int(R[:, i].sum() - 1)}
              for i in range(n)]
    return {"levels": [[ids[i] for i in lv] for lv in levels],
            "level_of": level_map, "micmac": micmac,
            "reachability": R.tolist(), "ids": ids,
            "names": {i: nm for i, nm in ISM_FACTORS}}


# ----------------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------------

def run_all(data_path="data/real_learners.csv", k=4, budget_ratio=0.45, verbose=True):
    df, log = load_clean(data_path)
    q = quality_profile(df)
    desc = describe(df)
    cr = corr_matrix(df)
    cl = cluster(df, k=k)
    pr = predict(cl["df"])
    ev = evaluate(df)
    dc = decide(cl, budget_ratio)
    sw = budget_sweep(cl, [0.2, 0.3, 0.4, 0.45, 0.55, 0.7, 0.85, 1.0])
    im = ism()

    results = {
        "clean_log": log, "quality": q, "describe": desc,
        "corr": {"cols": list(cr.columns), "matrix": cr.values.round(3).tolist()},
        "cluster": {kk: vv for kk, vv in cl.items() if kk not in ("df", "labels")},
        "predict": {kk: vv for kk, vv in pr.items() if kk != "fitted"},
        "evaluate": ev, "decide": dc, "sweep": sw, "ism": im,
    }
    # 挂载国内内容级证据链的实跑结果（由 multimodal.py 生成），支撑第 3、5、6、7、8 章。
    # 二者共用同一份建模样本与固定种子，报告 / 界面 / 看板均从此处取数。
    mm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "04_运行结果", "multimodal_results.json")
    if os.path.exists(mm_path):
        with open(mm_path, encoding="utf-8") as f:
            results["multimodal"] = json.load(f)
    if verbose:
        print(f"[数据] OULAD 真实数据 {log['raw_rows']} 条 -> 清洗后 {log['clean_rows']} 条，"
              f"学习者 {log['learners']} 人，课程 {log['courses']} 门，真实通过率 {log['cert_rate']:.2%}")
        print(f"[质量] " + "  ".join(f"{k2}={v2}" for k2, v2 in q.items()))
        print(f"[聚类] k={cl['k']} 轮廓系数={cl['silhouette']:.4f} ARI={cl['ari']}")
        for r in cl["profile_records"]:
            print(f"        {r['name']:<8} n={r['n']:<5} 占比={r['share']:.1%} "
                  f"时长={r['hours']:.1f}h 达标率={r['cert']:.1%}")
        for nm, m in pr["metrics"].items():
            print(f"[预测] {nm:<8} AUC={m['auc']:.4f} F1={m['f1']:.4f} CV-AUC={m['cv_auc']:.4f}")
        print(f"[评价] AHP CR={ev['ahp']['cr']:.4f} 组合权重={ev['combined']}")
        print(f"       第1名 {ev['rank'][0]['course_name']} ({ev['rank'][0]['closeness']:.3f})")
        print(f"[决策] 预算={dc['budget']:.0f}元 最优={dc['opt_gain']:.1f}人 "
              f"经验={dc['heur_gain']:.1f}人 普惠={dc['uniform_gain']:.1f}人")
    return results


def save(results, path="outputs/results.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    print(f"[OK] 结果已保存 -> {path}")
