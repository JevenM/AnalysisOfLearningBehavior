#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — 统一学习行为分析与学习者画像管线（一体化入口）

整合了 7 个开源学习分析项目的方法论，形成一条完整管线：

  数据层  多源真实公开数据集适配器（统一接口）：
          UCI Student Performance / 清华 FIB 短视频行为数据集（WWW 2025）/
          阿里云天池高校慕课学习行为脱敏样本(2024–2025) / HarvardX-MITx Person Course /
          中科大 EduData 集成包（ASSISTments 2009）/ data/ 目录通用扫描
          —— 只使用真实公开数据集，不生成任何虚拟或合成数据
  特征层  统一特征 schema、去重、IQR 离群处理、中值填充、衍生特征
  分析层  命名主成分（PCA 降维：学习投入度/资源偏好度/互动活跃度/学业表现）
          → 肘部法 + 轮廓系数选 K → K-Means 聚类画像
          → PCA 散点 / 雷达图 / 特征热力图
  挖掘层  社会网络分析 SNA（学习者-资源-课程网络：度中心性/PageRank/路径关联度）
          + 滞后序列分析 LSA（点击流编码 → 转换图谱 → 刷课/跳步异常检测）
  预测层  RandomForest 学业表现预测（课程完成率/最终成绩 + 特征重要性排序）
          + LogisticRegression 成功预测 + RF 风险预警（早期 vs 全特征）
          + 灰色 GM(1,1) 小样本行为趋势预测
  评价层  AHP 层次分析法（两两比较定权 + 一致性检验）
          + 模糊综合评判 → 每位学习者学习质量量化评分
  决策层  规则引擎产出 support / enrichment / review 干预队列（含行为异常证据）
          + 教师复核记录 + 分画像内容推荐
  输出层  out/ 目录：图表 PNG、metrics/profiles/queues CSV、report.md、审计日志

用法：
    python main.py                    # 融合 data/real/ 等目录下的全部真实公开数据集
    python main.py --k 4              # 指定聚类数（默认自动：肘部拐点 + 轮廓系数综合选择）
    python main.py --device gpu       # GPU 加速（自动探测 PyTorch CUDA / cuML，无则回退 CPU）
    python main.py --data-dir data --out-dir out
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
import warnings
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, silhouette_score,
                             f1_score)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------
SEED = 42
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#CCB974", "#8172B2", "#64B5CD"]

# 统一特征 schema（各数据源映射后对齐到这些列；缺失为 NaN）
CORE_FEATURES = [
    "total_clicks", "active_days", "avg_session_min", "forum_posts",
    "resource_clicks", "video_clicks", "quiz_attempts", "avg_quiz_score",
    "assignment_submit_lag", "help_requests", "login_freq_per_week",
]
EXTENDED_FEATURES = [
    "accuracy", "avg_response_time", "hesitation_rate", "engagement_score",
    "hint_usage", "success_rate", "skip_rate", "unique_days_ratio",
    "level_ups", "code_lines", "code_functions", "comment_ratio",
]
ALL_FEATURES = CORE_FEATURES + EXTENDED_FEATURES

# 各数据源字段 → 统一 schema 的别名映射（键统一小写）
COLUMN_ALIASES = {
    "student_id": "student_id", "userid": "student_id", "user_id": "student_id",
    "lms_clicks": "total_clicks", "clicks": "total_clicks",
    "study_hours": "study_hours", "attendance_rate": "attendance_rate",
    "quiz_score": "avg_quiz_score", "overall_score": "overall_score",
    "success_status": "success_status", "score": "avg_quiz_score",
    "accuracy": "accuracy", "avg_response_time": "avg_response_time",
    "response_time": "avg_response_time", "hesitation_rate": "hesitation_rate",
    "engagement_score": "engagement_score", "hint_usage": "hint_usage",
    "learner_type": "learner_type",
}

PERSONA_NAMES = [
    "Independent Deep Learner",
    "Guided Learner",
    "Exam-Driven Learner",
    "At-Risk / Disengaged",
]
PERSONA_DESC = {
    "Independent Deep Learner": "高投入、深度探索内容、测验得分高、极少求助。",
    "Guided Learner": "中等投入、偏好视频、频繁求助、依赖结构化指导。",
    "Exam-Driven Learner": "日常活跃度低、临近考核爆发、测验尝试率高、策略性取用内容。",
    "At-Risk / Disengaged": "点击与活跃天数极低、得分差、几乎无互动，流失风险高。",
}
CONTENT_REC = {
    "Independent Deep Learner": "研究性阅读材料、开放性项目、同伴讨论论坛、挑战级测评。",
    "Guided Learner": "分步视频教程、例题演示、即时反馈测验、脚手架式学习路径。",
    "Exam-Driven Learner": "精简摘要、记忆卡片、模拟试卷、间隔重复工具、截止日提醒。",
    "At-Risk / Disengaged": "短微内容单元、游戏化激励、主动推送通知、导师早期介入。",
}

# LSA 行为编码（点击流事件类型）
LSA_ACTIONS = ["video", "resource", "quiz", "forum", "help"]

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

log_handle = None


def log(msg: str) -> None:
    """同时输出到控制台与 JSONL 审计日志。"""
    global log_handle
    print(msg)
    if log_handle:
        log_handle.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                                     "msg": msg}, ensure_ascii=False) + "\n")
        log_handle.flush()


def setup_cjk_font() -> str | None:
    """探测并配置中文字体，避免图表中文显示为方块（tofu）。"""
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
                  "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
                  "SimSun", "KaiTi"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            log(f"  中文字体已配置: {name}")
            return name
    log("  警告: 未找到可用中文字体（可安装 微软雅黑/SimHei/Noto Sans CJK SC），"
        "图表中文可能显示为方块")
    return None


# ======================================================================
# 一、数据层：多源真实公开数据集适配器（统一接口）
# ======================================================================
# 接口约定：每个适配器返回 (features_frames, clicks_df|None)
#   features_frames: list[DataFrame]，含 student_id / source / 统一特征列
#                    （ALL_FEATURES 的子集）+ 标签列与真实人口属性列
#   clicks_df: 事件流（student_id/timestamp/action/resource_id/course_id/
#              resource_type/completed/speed_used/seek_count），无事件流的源为 None
# 本仓库只使用真实公开数据集，不生成任何虚拟/合成数据。
ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_CLICKS = 200000        # 事件流总量上限（SNA / LSA / GM(1,1) 的输入规模）
CLICKS_PER_USER = 30       # 每个用户最多抽取的事件数
_ADAPTER_CACHE: dict = {}


def get_source(fn, data_dir):
    """适配器结果缓存（特征表与事件流共用一次加载）。"""
    key = fn.__name__
    if key not in _ADAPTER_CACHE:
        _ADAPTER_CACHE[key] = fn(data_dir)
    return _ADAPTER_CACHE[key]


def load_uci_student(data_dir: str):
    """UCI Student Performance（葡萄牙两所中学数学/葡萄牙语成绩单，395+649 名学生）。
    https://archive.ics.uci.edu/dataset/320/student+performance"""
    frames = []
    for name, tag in [("student-mat.csv", "mat"), ("student-por.csv", "por")]:
        path = os.path.join(data_dir, "real", "uci_student", name)
        if not os.path.exists(path):
            log(f"  [UCI Student Performance] 缺少 {name}，跳过")
            continue
        df = pd.read_csv(path, sep=";")
        study_map = {1: 1.5, 2: 3.5, 3: 7.5, 4: 12.0}   # 周学习时长档位 → 小时中值
        feat = pd.DataFrame({
            "student_id": [f"UCI-{tag}-{i}" for i in range(len(df))],
            "study_hours": df["studytime"].map(study_map).astype(float),
            "avg_quiz_score": df[["G1", "G2", "G3"]].mean(axis=1).astype(float),
            "final_grade": df["G3"].astype(float),
            "quiz_attempts": 3.0,                        # 学期三次正式测验 G1/G2/G3
            "attendance_rate": (100 - df["absences"] / 93 * 100).clip(0, 100),
            "help_requests": (df["schoolsup"] == "yes").astype(float),
            "success_status": np.where(df["G3"] >= 10, "pass", "fail"),
            "gender": df["sex"].map({"F": "女", "M": "男"}),
            "age_band": pd.cut(df["age"], [0, 17, 19, 22, 100],
                               labels=["≤17", "18-19", "20-22", "23+"]).astype(str),
            "region": df["address"].map({"U": "城市", "R": "农村"}),
            "education": "高中",
        })
        feat["source"] = "real:uci_student_performance"
        frames.append(feat)
        log(f"  [UCI Student Performance] 已加载 {name}  n={len(feat)}")
    return frames, None


def load_thu_shortvideo(data_dir: str):
    """清华大学 FIB 实验室大规模短视频行为数据集（WWW 2025 发布）本地子样本。
    点击/停留/完播/点赞等行为记录，并利用类目表标记教育类内容占比。
    http://fi.ee.tsinghua.edu.cn/datasets/short-video-dataset"""
    base = os.path.join(data_dir, "real", "thu_shortvideo")
    path = os.path.join(base, "interaction_sampled.csv")
    if not os.path.exists(path):
        log("  [清华短视频] 未找到 interaction_sampled.csv，跳过")
        return [], None
    usecols = ["user_id", "pid", "category_id", "exposed_time", "watch_time",
               "duration", "cvm_like", "comment", "follow", "collect", "forward",
               "hate", "click", "p_date", "p_hour", "gender", "age"]
    df = pd.read_csv(path, usecols=usecols)
    edu_cats = set()
    cat_path = os.path.join(base, "categories_cn_en.csv")
    if os.path.exists(cat_path):
        cats = pd.read_csv(cat_path)
        edu = cats[cats["category_name_cn"].astype(str).str.contains(
            "教育|学习|知识|科普|技能", na=False)]
        edu_cats = set(edu["category_id"].tolist())
    df["is_edu"] = df["category_id"].isin(edu_cats)
    social = df[["cvm_like", "comment", "follow", "collect", "forward"]].any(axis=1)
    # 向量化完播率（每行 观看时长/视频时长，逐用户平均），避免 groupby.apply
    watch_ratio = (df["watch_time"] / df["duration"].clip(lower=1)).clip(0, 1)

    g = df.groupby("user_id")
    feat = pd.DataFrame({
        "student_id": [f"THUSV-{u}" for u in g.size().index],
        "total_clicks": g.size().values.astype(float),
        "active_days": g["p_date"].nunique().values.astype(float),
        "avg_session_min": (g["watch_time"].mean() / 60).values,
        "video_clicks": g["click"].sum().values.astype(float),
        "forum_posts": g["comment"].sum().values.astype(float),
        "engagement_score": social.groupby(df["user_id"]).mean().values.round(3),
        "completion_rate": watch_ratio.groupby(df["user_id"]).mean().values.round(3),
        "edu_video_ratio": g["is_edu"].mean().values.round(3),
        "gender": g["gender"].first().values,
        "age_band": pd.cut(pd.to_numeric(g["age"].first(), errors="coerce"),
                           [0, 17, 25, 40, 120],
                           labels=["≤17", "18-25", "26-40", "40+"]).astype(str).values,
    })
    feat["source"] = "real:thu_shortvideo"
    log(f"  [清华短视频] 已加载 interaction_sampled.csv  用户 n={len(feat)}, "
        f"行为记录 {len(df)} 条")

    # 事件流（每用户抽样，供 SNA/LSA/GM(1,1) 使用）
    ev = df.sort_values("exposed_time").groupby("user_id", group_keys=False) \
        .head(CLICKS_PER_USER)
    clicks = pd.DataFrame({
        "student_id": "THUSV-" + ev["user_id"].astype(str),
        "timestamp": pd.to_datetime(ev["p_date"].astype(str), format="%Y%m%d") \
            + pd.to_timedelta(ev["p_hour"], unit="h"),
        "action": "video",
        "resource_id": "V" + ev["pid"].astype(str),
        "course_id": "shortvideo",
        "resource_type": "short",
        "completed": (ev["watch_time"] >= 0.9 * ev["duration"].clip(lower=1)) \
            .astype(float).values,
        "speed_used": np.nan,
        "seek_count": np.nan,
    })
    return [feat], clicks


def code_to_features(subs: pd.DataFrame) -> pd.DataFrame:
    """代码文本 → 学生级静态特征（AST/token 统计，无需重型依赖）。"""
    rows = []
    for _, r in subs.iterrows():
        code = r["code"]
        lines = [ln for ln in code.splitlines() if ln.strip()]
        comments = [ln for ln in lines if ln.strip().startswith("#")]
        try:
            tree = ast.parse(code)
            n_funcs = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                          for node in ast.walk(tree))
            n_calls = sum(isinstance(node, ast.Call) for node in ast.walk(tree))
        except SyntaxError:
            n_funcs, n_calls = 0, 0
        rows.append({
            "student_id": r["student_id"], "code_lines": len(lines),
            "code_functions": n_funcs, "code_calls": n_calls,
            "comment_ratio": len(comments) / max(len(lines), 1),
            "avg_quiz_score": r["grade"],
            "source": "code_submissions",
        })
    return pd.DataFrame(rows)


def load_mooc_tianchi(data_dir: str):
    """阿里云天池《中国高校慕课课程多维画像与学习行为脱敏样本数据集(2024–2025)》。
    user_learning_behavior.xls：视频播放/测验/作业/论坛/资源下载等学习行为事件；
    courses.xls：课程维表（中国大学 MOOC 平台真实脱敏样本）。"""
    base = os.path.join(ROOT, "_repos",
                        "阿里云天池《中国高校慕课课程多维画像与学习行为脱敏样本数据集(2024–2025)》")
    beh_path = os.path.join(base, "user_learning_behavior.xls")
    if not os.path.exists(beh_path):
        log("  [天池慕课] 未找到 user_learning_behavior.xls，跳过")
        return [], None
    df = pd.read_excel(beh_path)
    # 真实数据存在脏时间戳（如 3000 年），越界值置 NaT 后丢弃
    df["occur_time"] = pd.to_datetime(df["occur_time"], errors="coerce")
    df = df[df["occur_time"].notna()]
    df["date"] = df["occur_time"].dt.date
    action_map = {"video_play": "video", "quiz_submit": "quiz", "homework_submit": "quiz",
                  "forum_browse": "forum", "resource_download": "resource",
                  "course_eval": "help"}
    df["action"] = df["behavior_type"].map(action_map).fillna("resource")

    # 行为类型布尔列（向量化统计，避免 groupby.apply 逐组遍历）
    for t in ("video_play", "forum_browse", "resource_download", "course_eval"):
        df[f"is_{t}"] = df["behavior_type"].eq(t)
    df["is_quiz"] = df["behavior_type"].isin(["quiz_submit", "homework_submit"])
    progress = (df["progress_after_pct"] / 100).clip(0, 1)

    g = df.groupby("learner_id")
    score_mean = g["score"].mean()
    feat = pd.DataFrame({
        "student_id": [f"MOOC-{u}" for u in g.size().index],
        "total_clicks": g.size().values.astype(float),
        "active_days": g["date"].nunique().values.astype(float),
        "avg_session_min": (g["duration_sec"].mean() / 60).values,
        "video_clicks": g["is_video_play"].sum().values.astype(float),
        "quiz_attempts": g["is_quiz"].sum().values.astype(float),
        "avg_quiz_score": score_mean.values.round(1),
        "final_grade": score_mean.values.round(1),
        "forum_posts": g["is_forum_browse"].sum().values.astype(float),
        "resource_clicks": g["is_resource_download"].sum().values.astype(float),
        "help_requests": g["is_course_eval"].sum().values.astype(float),
        "completion_rate": progress.groupby(df["learner_id"]).mean().values.round(3),
        "success_rate": g["is_passed"].mean().values.round(3),
        "avg_response_time": g["duration_sec"].mean().values,
    })
    feat["source"] = "real:tianchi_mooc_2024_2025"
    log(f"  [天池慕课] 已加载 user_learning_behavior.xls  学习者 n={len(feat)}, "
        f"行为事件 {len(df)} 条")

    clicks = pd.DataFrame({
        "student_id": "MOOC-" + df["learner_id"].astype(str),
        "timestamp": df["occur_time"],
        "action": df["action"],
        "resource_id": df["resource_code"].astype(str),
        "course_id": df["course_id"].astype(str),
        "resource_type": np.where(df["behavior_type"] == "video_play", "mooc", None),
        "completed": (df["progress_after_pct"] >= 95).astype(float),
        "speed_used": pd.to_numeric(df["playback_speed"], errors="coerce"),
        "seek_count": np.nan,
    })
    return [feat], clicks


def load_harvardx_mitx(data_dir: str, max_users: int = 50000):
    """HarvardX-MITx Person Course Dataset（edX 2012-2013 年两校课程真实脱敏数据）。
    按用户聚合跨课程行为：事件数/活跃天数/视频/论坛/章节 + 结业标签与人口属性。"""
    base = os.path.join(ROOT, "_repos",
                        "HarvardX_MITx_Person_Course_Dataset_Exploration-master", "data")
    path = os.path.join(base, "hmx_cleaned.csv")
    uid_col = "user_id"
    if not os.path.exists(path):
        path = os.path.join(base, "HMXPC13_DI_v2_5-14-14.csv")
        uid_col = "userid_DI"
    if not os.path.exists(path):
        log("  [HarvardX/MITx] 未找到数据文件，跳过")
        return [], None
    usecols = [uid_col, "course_id", "nevents", "ndays_act", "nplay_video",
               "nforum_posts", "nchapters", "grade", "certified", "viewed",
               "explored", "gender", "YoB", "education"]
    df = pd.read_csv(path, usecols=[c for c in usecols])
    df = df.rename(columns={uid_col: "user_id"})
    users = df["user_id"].unique()
    if len(users) > max_users:                      # 真实数据随机抽样，控制规模
        users = np.random.default_rng(SEED).choice(users, max_users, replace=False)
        df = df[df["user_id"].isin(users)]
        log(f"  [HarvardX/MITx] 用户量 {len(users)} 超上限，随机抽样 {max_users} 人")
    g = df.groupby("user_id")
    grade_max = g["grade"].max() * 100
    edu_map = {"Less than Secondary": "高中及以下", "Secondary": "高中及以下",
               "High School": "高中及以下", "Bachelor's": "本科", "Master's": "硕士及以上",
               "Doctorate": "硕士及以上"}
    feat = pd.DataFrame({
        "student_id": "EDX-" + g.size().index.astype(str),
        "total_clicks": g["nevents"].sum().values.astype(float),
        "active_days": g["ndays_act"].sum().values.astype(float),
        "video_clicks": g["nplay_video"].sum().values.astype(float),
        "forum_posts": g["nforum_posts"].sum().values.astype(float),
        "resource_clicks": g["nchapters"].sum().values.astype(float),
        "avg_quiz_score": grade_max.values.round(1),
        "final_grade": grade_max.values.round(1),
        "completion_rate": g[["viewed", "explored"]].max().mean(axis=1) \
            .values.round(3),
        "success_status": np.where(g["certified"].max().values == 1,
                                   "certified", "none"),
        "gender": g["gender"].first().map({"m": "男", "f": "女", "M": "男", "F": "女"}) \
            .values,
        "age_band": pd.cut(pd.to_numeric(g["YoB"].first(), errors="coerce"),
                           [0, 1975, 1985, 1995, 2004],
                           labels=["40+", "31-40", "26-30", "18-25"]).astype(str).values,
        "education": g["education"].first().map(edu_map).values,
    })
    feat["source"] = "real:harvardx_mitx_person_course"
    log(f"  [HarvardX/MITx] 已加载 {os.path.basename(path)}  用户 n={len(feat)}")
    return [feat], None


def load_edudata_assistments(data_dir: str):
    """中科大 EduData 集成包托管的 ASSISTments 2009-2010 答题行为数据
    （skill_builder_data_corrected，约 34 万条中学数学答题记录）。
    https://github.com/bigdata-ustc/EduData"""
    path = os.path.join(data_dir, "real", "edudata", "assistment_2009",
                        "skill_builder_data_corrected.csv")
    if not os.path.exists(path):
        log("  [EduData/ASSISTments] 未找到 skill_builder_data_corrected.csv，跳过")
        return [], None
    try:
        df = pd.read_csv(path, usecols=["user_id", "correct", "ms_first_response",
                                        "hint_count", "order_id", "skill_id"])
    except UnicodeDecodeError:
        # 个别行含非 UTF-8 字节；所用列均为数值列，latin-1 可无损解析
        df = pd.read_csv(path, usecols=["user_id", "correct", "ms_first_response",
                                        "hint_count", "order_id", "skill_id"],
                         encoding="latin-1")
    df = df.dropna(subset=["user_id"])
    g = df.groupby("user_id")
    acc = g["correct"].mean()
    feat = pd.DataFrame({
        "student_id": "AST-" + g.size().index.astype(int).astype(str),
        "total_clicks": g.size().values.astype(float),
        "quiz_attempts": g.size().values.astype(float),
        "accuracy": acc.values.round(3),
        "success_rate": acc.values.round(3),
        "avg_quiz_score": (acc * 100).values.round(1),
        "avg_response_time": (g["ms_first_response"].mean() / 1000).values.round(1),
        "hint_usage": g["hint_count"].sum().values.astype(float),
        "help_requests": g["hint_count"].sum().values.astype(float),
        "success_status": np.where(acc.values >= 0.6, "pass", "fail"),
    })
    feat["source"] = "real:edudata_assistments_2009"
    log(f"  [EduData/ASSISTments] 已加载 skill_builder_data_corrected.csv  "
        f"学生 n={len(feat)}, 答题记录 {len(df)} 条")
    return [feat], None


def _cap_clickstream(frames: list) -> pd.DataFrame:
    """合并各真实源事件流，超限时随机抽样（保持真实行为分布，不造数据）。"""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["student_id", "timestamp", "action"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["action"].isin(LSA_ACTIONS)].dropna(subset=["timestamp"])
    if len(df) > MAX_CLICKS:
        df = df.sample(MAX_CLICKS, random_state=SEED).sort_values("timestamp")
        log(f"  事件流超过 {MAX_CLICKS} 条，随机采样至 {len(df)} 条")
    return df.reset_index(drop=True)


def build_clickstream(data_dir: str) -> pd.DataFrame:
    """汇聚各真实数据源的带时间戳事件流（SNA / LSA / GM(1,1) 的输入）。
    可用来源：清华短视频、天池慕课事件、data/clickstream.csv。"""
    frames = []
    for fn in (load_thu_shortvideo, load_mooc_tianchi):
        try:
            _, ev = get_source(fn, data_dir)
            if ev is not None and not ev.empty:
                frames.append(ev)
                log(f"  事件流来源 {fn.__name__}: {len(ev)} 条")
        except Exception as e:
            log(f"  事件流适配器 {fn.__name__} 失败: {e}")
    path = os.path.join(data_dir, "clickstream.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df = normalize_columns(df)
            if {"student_id", "timestamp", "action"}.issubset(df.columns):
                for col, default in [("resource_id", "R0"), ("course_id", "C0"),
                                     ("resource_type", None), ("completed", np.nan),
                                     ("speed_used", np.nan), ("seek_count", np.nan)]:
                    if col not in df.columns:
                        df[col] = default
                frames.append(df)
                log(f"  事件流来源 clickstream.csv: {len(df)} 条")
            else:
                log("  clickstream.csv 缺少必需列（student_id/timestamp/action），跳过")
        except Exception as e:
            log(f"  读取 clickstream.csv 失败: {e}")
    clicks = _cap_clickstream(frames)
    if clicks.empty:
        raise SystemExit("未发现带时间戳的真实事件流：请确认 "
                         "data/real/thu_shortvideo/interaction_sampled.csv 或天池慕课数据存在")
    return clicks


def harmonize_demographics(uni: pd.DataFrame) -> pd.DataFrame:
    """人口属性仅来自真实数据列；缺失保持缺失，不再做任何合成或行为加权。"""
    if "gender" in uni.columns:
        g = uni["gender"].astype(str).str.strip().str.upper()
        uni["gender"] = g.map({"M": "男", "F": "女", "男": "男", "女": "女"})
    if "age_band" not in uni.columns and "age" in uni.columns:
        uni["age_band"] = pd.cut(pd.to_numeric(uni["age"], errors="coerce"),
                                 [0, 17, 25, 40, 120],
                                 labels=["≤17", "18-25", "26-40", "40+"]).astype(str)
    return uni


# ======================================================================
# 二、数据层：真实数据加载（扫描 data/ 目录）
# ======================================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})
    return df


def load_real_data(data_dir: str) -> list[pd.DataFrame]:
    """扫描 data/：*.csv/xlsx → 汇总表；*.jsonl → 事件流；submissions/**/*.py → 代码提交。"""
    frames = []
    for path in glob.glob(os.path.join(data_dir, "*.csv")) + \
                glob.glob(os.path.join(data_dir, "*.xlsx")):
        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
        except Exception as e:
            log(f"  跳过无法读取的文件 {path}: {e}")
            continue
        df = normalize_columns(df)
        if "student_id" not in df.columns:
            log(f"  跳过 {path}（缺少 student_id 列）")
            continue
        df["source"] = f"real:{os.path.basename(path)}"
        frames.append(df)
        log(f"  已加载真实汇总表 {os.path.basename(path)}  shape={df.shape}")
    for path in glob.glob(os.path.join(data_dir, "*.jsonl")):
        try:
            recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        except Exception as e:
            log(f"  跳过无法读取的 JSONL {path}: {e}")
            continue
        events = []
        for rec in recs:
            d = rec.get("details", rec.get("metrics", {}))
            m = d.get("metrics", d) if isinstance(d, dict) else {}
            events.append({"student_id": rec.get("student_id"),
                           **{k: m.get(k) for k in ALL_FEATURES if k in m}})
        df = pd.DataFrame(events).dropna(how="all", subset=[c for c in ALL_FEATURES
                                                            if c != "student_id"])
        if not df.empty:
            df = normalize_columns(df)
            df["source"] = f"real:{os.path.basename(path)}"
            frames.append(df)
            log(f"  已加载真实交互日志 {os.path.basename(path)}  shape={df.shape}")
    code_files = glob.glob(os.path.join(data_dir, "submissions", "**", "*.py"),
                           recursive=True)
    if code_files:
        rows = []
        for fp in code_files:
            sid = os.path.basename(os.path.dirname(fp)) or os.path.basename(fp)
            try:
                rows.append({"student_id": sid, "code": open(fp, encoding="utf-8").read()})
            except Exception:
                pass
        if rows:
            df = code_to_features(pd.DataFrame(rows))
            df["source"] = "real:code_submissions"
            frames.append(df)
            log(f"  已加载真实代码提交 {len(rows)} 个文件")
    return frames


# ======================================================================
# 三、特征层：统一 schema + 清洗
# ======================================================================
def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """去重、IQR 离群截断、中值填充（借鉴 data-pipeline 的清洗模板）。"""
    df = df.drop_duplicates(subset="student_id").reset_index(drop=True)
    num_cols = [c for c in ALL_FEATURES if c in df.columns and
                pd.api.types.is_numeric_dtype(df[c])]
    for c in num_cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            df[c] = df[c].clip(lo, hi)
        df[c] = df[c].fillna(df[c].median())
    return df


def build_unified_dataset(args) -> pd.DataFrame:
    """融合所有真实公开数据集 → 统一特征表（union 列，缺失由 clean_features 填充）。

    数据源（均为真实公开数据，不含任何合成数据）：
      1. UCI Student Performance（395+649 名中学生成绩单）
      2. 清华 FIB 短视频行为数据集（WWW 2025，本地子样本）
      3. 阿里云天池《中国高校慕课课程多维画像与学习行为脱敏样本数据集(2024–2025)》
      4. HarvardX/MITx Person Course Dataset（edX 真实脱敏数据）
      5. 中科大 EduData 集成包（ASSISTments 2009 答题行为）
      6. data/ 根目录通用扫描（CSV/XLSX 汇总表、JSONL 事件流、代码提交）
    """
    adapters = [load_uci_student, load_thu_shortvideo, load_mooc_tianchi,
                load_harvardx_mitx, load_edudata_assistments]
    frames = []
    for fn in adapters:
        try:
            src_frames, _ = get_source(fn, args.data_dir)
            frames += src_frames
        except Exception as e:
            log(f"  适配器 {fn.__name__} 加载失败: {e}")
    frames += load_real_data(args.data_dir)
    if not frames:
        raise SystemExit("未发现任何真实数据集，请先按 README 将公开数据集放入 data/real/")
    parts = []
    for df in frames:
        keep = ["student_id", "source"] + [c for c in ALL_FEATURES if c in df.columns]
        extra = [c for c in ["success_status", "overall_score", "grade",
                             "learner_type", "study_hours", "attendance_rate",
                             "code_calls", "completion_rate", "final_grade",
                             "gender", "age_band", "region", "education"]
                 if c in df.columns]
        parts.append(df[keep + extra])
    uni = pd.concat(parts, ignore_index=True, sort=False)
    # 标签列：优先使用数据集自带的成功/结业标签，否则由真实成绩中位数派生
    if "success_status" in uni.columns and uni["success_status"].notna().any():
        uni["success"] = uni["success_status"].astype(str).str.lower().isin(
            ["successful", "success", "certified", "pass", "1", "true"])
    else:
        score = uni["avg_quiz_score"]
        if "overall_score" in uni.columns:
            score = score.fillna(uni["overall_score"])
        uni["success"] = score >= np.nanmedian(score)
    # 以下两列仅对数据集未提供的来源按其真实行为派生代理值（非合成样本）
    if "completion_rate" not in uni.columns:
        eng = uni[["total_clicks", "active_days", "engagement_score"]].mean(axis=1)
        eng_n = (eng - eng.min()) / (eng.max() - eng.min() + 1e-9)
        score_n = (uni["avg_quiz_score"] / 100).clip(0, 1).fillna(0.5)
        uni["completion_rate"] = (0.6 * eng_n + 0.4 * score_n).clip(0, 1).round(3)
    if "final_grade" not in uni.columns:
        eng = uni[["total_clicks", "active_days", "engagement_score"]].mean(axis=1)
        eng_n = (eng - eng.min()) / (eng.max() - eng.min() + 1e-9)
        score_n = (uni["avg_quiz_score"] / 100).clip(0, 1)
        uni["final_grade"] = (100 * (0.7 * score_n.fillna(0.5) + 0.3 * eng_n)).round(1)
    uni["completion_rate"] = uni["completion_rate"].fillna(uni["completion_rate"].median())
    uni["final_grade"] = uni["final_grade"].fillna(uni["final_grade"].median())
    uni = clean_features(uni)
    # 人口属性只保留真实列，缺失保持缺失（不再合成）
    uni = harmonize_demographics(uni)
    # 基础行为代理指标（仅由真实字段派生；字段缺失时保持缺失，不再合成）
    if "study_hours" in uni.columns:            # UCI 自报周学习时长（真实字段）
        uni["study_hours_per_week"] = pd.to_numeric(uni["study_hours"], errors="coerce")
    elif "login_freq_per_week" in uni.columns:
        uni["study_hours_per_week"] = (uni["avg_session_min"] *
                                       uni["login_freq_per_week"] / 60).round(1)
    else:
        uni["study_hours_per_week"] = np.nan
    if "assignment_submit_lag" in uni.columns:
        uni["submission_rate"] = ((uni["assignment_submit_lag"] + 5) / 10).clip(0, 1).round(3)
    else:
        uni["submission_rate"] = np.nan
    log(f"\n统一数据集: {uni.shape[0]} 名学生 × {uni.shape[1]} 列, "
        f"来源分布: {uni['source'].value_counts().to_dict()}")
    return uni


# ======================================================================
# 四、分析层：EDA + 聚类画像
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
    desc.to_csv(os.path.join(out_dir, "eda_summary.csv"), encoding="utf-8-sig")
    return desc


# ======================================================================
# 计算加速：可选 GPU 后端（--device auto|cpu|gpu）
# ======================================================================
# 探测顺序：RAPIDS cuML（Linux/WSL，KMeans 全 GPU）→ PyTorch CUDA（Windows 可用，
# GPU 加速 KMeans / 轮廓系数 / 最近质心分配）→ CPU（sklearn + RF 多核并行）。
# 显式指定 --device gpu 但无可用后端时自动回退 CPU 并告警，不中断管线。
_DEVICE_PREF = "auto"
_ENGINE_INFO: dict = {}


def detect_device(pref: str = "auto") -> dict:
    """探测可用 GPU 计算后端；pref=cpu 强制 CPU。"""
    global _ENGINE_INFO
    info = {"pref": pref, "backend": "cpu", "device_name": None}
    if pref != "cpu":
        try:
            import cuml  # noqa: F401
            info["backend"] = "cuml"
        except Exception:
            pass
        if info["backend"] == "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    info["backend"] = "torch"
                    info["device_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    _ENGINE_INFO = info
    return info


def _torch_device():
    import torch
    return torch.device("cuda")


def torch_kmeans(X_np: np.ndarray, k: int, seed: int = SEED, n_init: int = 3,
                 max_iter: int = 100, tol: float = 1e-5):
    """GPU KMeans（PyTorch Lloyd 迭代，多次初始化取最优）。"""
    import torch
    dev = _torch_device()
    torch.manual_seed(seed)
    X = torch.as_tensor(np.ascontiguousarray(X_np, dtype=np.float32), device=dev)
    n = X.shape[0]
    best = None
    for _ in range(max(1, n_init)):
        centers = X[torch.randperm(n, device=dev)[:k]].clone()
        for _ in range(max_iter):
            labels = torch.cdist(X, centers).argmin(1)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, labels, X)
            counts = torch.zeros(k, device=dev)
            counts.index_add_(0, labels, torch.ones(n, device=dev))
            counts = counts.clamp(min=1)
            new_centers = sums / counts.unsqueeze(1)
            shift = float((new_centers - centers).norm(dim=1).max())
            centers = new_centers
            if shift < tol:
                break
        inertia = float(torch.cdist(X, centers).min(dim=1).values.pow(2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, centers, labels)
    inertia, centers, labels = best
    return centers.cpu().numpy(), labels.cpu().numpy(), inertia


def torch_silhouette(X_np: np.ndarray, labels: np.ndarray, chunk: int = 4096) -> float:
    """GPU 轮廓系数（分块计算成对距离，控制显存占用）。"""
    import torch
    dev = _torch_device()
    X = torch.as_tensor(np.ascontiguousarray(X_np, dtype=np.float32), device=dev)
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=dev)
    n = X.shape[0]
    uniq, y_inv = torch.unique(y, return_inverse=True)
    k = len(uniq)
    sizes = torch.bincount(y_inv, minlength=k).float()
    s = torch.zeros(n, device=dev)
    for st in range(0, n, chunk):
        ed = min(st + chunk, n)
        d = torch.cdist(X[st:ed], X)
        same = y_inv[st:ed].unsqueeze(1) == y_inv.unsqueeze(0)
        a = (d * same).sum(1) / (sizes[y_inv[st:ed]] - 1).clamp(min=1)
        sums = torch.zeros((ed - st, k), device=dev)
        sums.scatter_add_(1, y_inv.unsqueeze(0).expand(ed - st, -1), d)
        mean = sums / sizes.unsqueeze(0)
        mean.scatter_(1, y_inv[st:ed].unsqueeze(1), float("inf"))   # 屏蔽自身簇
        b = mean.min(dim=1).values
        s[st:ed] = (b - a) / torch.maximum(a, b).clamp(min=1e-12)
    s[sizes[y_inv] <= 1] = 0            # 单点簇轮廓系数记 0（与 sklearn 一致）
    return float(s.mean())


def torch_nearest_centers(X_np: np.ndarray, centers_np: np.ndarray,
                          chunk: int = 16384) -> np.ndarray:
    """GPU 最近质心分配（全量样本标签，分块控显存）。"""
    import torch
    dev = _torch_device()
    C = torch.as_tensor(np.ascontiguousarray(centers_np, dtype=np.float32), device=dev)
    X = torch.as_tensor(np.ascontiguousarray(X_np, dtype=np.float32), device=dev)
    labels = torch.empty(X.shape[0], dtype=torch.long, device=dev)
    for st in range(0, X.shape[0], chunk):
        ed = min(st + chunk, X.shape[0])
        labels[st:ed] = torch.cdist(X[st:ed], C).argmin(1)
    return labels.cpu().numpy()


class ClusterEngine:
    """KMeans + 轮廓系数统一引擎，按 detect_device 探测的后端自动加速。"""

    def __init__(self, info: dict | None = None):
        self.backend = (info or _ENGINE_INFO or {"backend": "cpu"})["backend"]

    def fit(self, X_fit: np.ndarray, k: int, seed: int = SEED, n_init: int = 10) -> dict:
        if self.backend == "torch":
            centers, labels, inertia = torch_kmeans(X_fit, k, seed=seed, n_init=3)
            sil = torch_silhouette(X_fit, labels)
            return {"labels": labels, "centers": centers, "inertia": inertia, "sil": sil}
        if self.backend == "cuml":
            from cuml.cluster import KMeans as cuKMeans
            km = cuKMeans(n_clusters=k, random_state=seed,
                          n_init=3).fit(X_fit.astype(np.float32))
            labels = np.asarray(km.labels_).astype(int)
            try:
                sil = float(silhouette_score(X_fit, labels))
            except ValueError:
                sil = np.nan
            return {"labels": labels, "centers": np.asarray(km.cluster_centers_),
                    "inertia": float(km.inertia_), "sil": sil}
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(X_fit)
        try:
            sil = silhouette_score(X_fit, km.labels_)
        except ValueError:
            sil = np.nan
        return {"labels": km.labels_, "centers": km.cluster_centers_,
                "inertia": km.inertia_, "sil": sil}

    def predict(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        if self.backend == "torch":
            return torch_nearest_centers(X, centers)
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        return d.argmin(1).astype(int)


def _knee_distances(ks: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """肘部得分：inertia 下降曲线上每个 k 到首尾连线的垂直距离（kneedle 思路）。"""
    x0, y0, x1, y1 = ks[0], ys[0], ks[-1], ys[-1]
    denom = np.hypot(x1 - x0, y1 - y0) + 1e-12
    return np.abs((y1 - y0) * ks - (x1 - x0) * ys + x1 * y0 - y1 * x0) / denom


def cluster_personas(uni: pd.DataFrame, cols: list[str], k: int, out_dir: str,
                     max_fit: int = 15000, max_plot: int = 5000):
    """肘部法 + 轮廓系数选 K → KMeans → PCA / 雷达 / 热力图。

    选 K 修复点（相对旧实现）：
      1. 自动模式综合「肘部拐点」与「轮廓系数」两路证据等权融合（旧实现肘部法
         只画图不参与选择，仅取轮廓 argmax，与文档描述不符）；
      2. 显式指定的 k 做合法性校验并夹取到 [2, min(8, n-1)]（旧实现 k>=n 时
         KMeans 直接报错、k=1 时 silhouette 报错）；
      3. 大数据集下 silhouette 为 O(n²) 不可行，改为固定随机子样本上拟合与评估，
         全量样本用最近质心分配标签；
      4. K 选择与最终聚类使用一致的 n_init 与同一份标准化矩阵。
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
    engine = ClusterEngine(_ENGINE_INFO or {"backend": "cpu"})
    inertias, sils = [], []
    for kk in k_range:
        r = engine.fit(X_fit, int(kk))
        inertias.append(r["inertia"])
        sils.append(r["sil"])
    inertias = np.array(inertias, dtype=float)
    sils = np.array(sils, dtype=float)
    sil_n = (sils - np.nanmin(sils)) / (np.nanmax(sils) - np.nanmin(sils) + 1e-9)
    knee_n = _knee_distances(k_range.astype(float), inertias)
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
    fig.suptitle("Selecting Optimal Number of Clusters", fontsize=14, fontweight="bold")
    axes[0].plot(k_range, inertias, "o-", color=PALETTE[0])
    axes[0].axvline(knee_k, color=PALETTE[3], ls="--", label=f"Elbow K = {knee_k}")
    axes[0].set(xlabel="K", ylabel="Inertia", title="Elbow Method")
    axes[0].legend()
    axes[1].plot(k_range, sils, "s-", color=PALETTE[1])
    axes[1].axvline(k, color=PALETTE[2], ls="--", label=f"K = {k}")
    axes[1].set(xlabel="K", ylabel="Silhouette", title="Silhouette Analysis")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elbow_silhouette.png"), bbox_inches="tight")
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
    proj.to_csv(os.path.join(out_dir, "pca_projection.csv"), index=False)
    log(f"  PCA 方差解释率: PC1={var[0]:.1f}%, PC2={var[1]:.1f}%")
    plot_idx = fit_idx if n > max_plot else np.arange(n)
    lab_p, pca_p = labels[plot_idx], X_pca[plot_idx]
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(k):
        m = lab_p == c
        ax.scatter(pca_p[m, 0], pca_p[m, 1], c=PALETTE[c % len(PALETTE)],
                   label=f"Cluster {c}", alpha=0.75, s=55, edgecolors="white", linewidths=0.4)
    ax.set(xlabel=f"PC1 ({var[0]:.1f}%)", ylabel=f"PC2 ({var[1]:.1f}%)",
           title="Learner Personas — PCA Projection")
    ax.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_clusters.png"), bbox_inches="tight")
    plt.close()

    # --- 非线性嵌入散点：t-SNE + UMAP（展示簇分离度；仅可视化用，不参与建模） ---
    # PCA 是线性投影，重尾行为特征使簇沿 PC1 呈条状排布；非线性嵌入能展开局部
    # 邻域结构，簇团分离度更清晰。做法：万级子样本 → PCA 预降到 10 维去噪提速
    # → t-SNE(init='pca') 与 UMAP(n_neighbors=15, min_dist=0.1)，均固定随机种子。
    # 两者坐标尺度均无业务含义（簇间距离/簇大小不可解读），且无 transform
    # 方法（UMAP 有但此处不做建模用途），不能替代 PCA 建模。
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
        emb.to_csv(os.path.join(out_dir, "tsne_projection.csv"), index=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        for c in range(k):
            sub = emb[emb["cluster"] == c]
            ax.scatter(sub["tSNE1"], sub["tSNE2"], c=PALETTE[c % len(PALETTE)],
                       label=f"Cluster {c}", alpha=0.55, s=12,
                       edgecolors="none", rasterized=True)
        ax.set(title="Learner Personas — t-SNE Projection "
                     "(cluster separation view; axis scale has no business meaning)")
        ax.legend(title="Cluster", markerscale=3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "tsne_clusters.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
    except Exception as e:                      # sklearn 过旧/内存不足时安全跳过
        log(f"  警告: t-SNE 可视化跳过（{type(e).__name__}: {e}）")

    try:
        import umap
        log("  UMAP: n_neighbors=15, min_dist=0.1 (约 10-30 秒)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # 固定 random_state 会禁用并行，忽略提示
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                random_state=SEED, n_jobs=1)
            X_umap = reducer.fit_transform(X_pre)
        emb = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
        emb["cluster"] = labels[viz_idx]
        emb.to_csv(os.path.join(out_dir, "umap_projection.csv"), index=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        for c in range(k):
            sub = emb[emb["cluster"] == c]
            ax.scatter(sub["UMAP1"], sub["UMAP2"], c=PALETTE[c % len(PALETTE)],
                       label=f"Cluster {c}", alpha=0.55, s=12,
                       edgecolors="none", rasterized=True)
        ax.set(title="Learner Personas — UMAP Projection "
                     "(cluster separation view; axis scale has no business meaning)")
        ax.legend(title="Cluster", markerscale=3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "umap_clusters.png"),
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
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow),
                                 subplot_kw=dict(polar=True), squeeze=False)
        fig.suptitle("Behavioral Radar — Learner Personas", fontsize=14, fontweight="bold")
        for idx, ax in enumerate(axes.flat):
            if idx >= len(clusters):
                ax.remove()
                continue
            c = clusters[idx]
            vals = norm.loc[c].tolist() + norm.loc[c].tolist()[:1]
            ax.plot(angles, vals, color=PALETTE[idx % len(PALETTE)], lw=2)
            ax.fill(angles, vals, color=PALETTE[idx % len(PALETTE)], alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_cols, size=7)
            ax.set_yticklabels([])
            ax.set_title(f"Cluster {idx}", pad=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "radar_personas.png"), bbox_inches="tight")
        plt.close()

    # --- 特征热力图 ---
    hm = uni.groupby("cluster")[cols].mean()
    hm_z = (hm - hm.mean()) / hm.std()
    fig, ax = plt.subplots(figsize=(min(14, 1.0 * len(cols) + 4), 4))
    hm_arr = hm_z.values
    im = ax.imshow(hm_arr, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(k), [f"Cluster {c}" for c in hm_z.index])
    for i in range(hm_arr.shape[0]):
        for j in range(hm_arr.shape[1]):
            ax.text(j, i, f"{hm_arr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Z-score")
    ax.set_title("Feature Z-scores per Cluster", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_features.png"), bbox_inches="tight")
    plt.close()

    return uni, engine, scaler, cols, {
        "silhouette": sil, "k": k, "knee_k": knee_k,
        "backend": engine.backend,
        "k_scores": {int(kk): round(float(s), 3) for kk, s in zip(k_range, sils)},
        "n_fit_samples": int(len(fit_idx)),
    }


def label_personas(uni: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """基于簇中心的启发式签名 → 统一画像命名（各参考项目的画像体系的并集）。"""
    z = uni.groupby("cluster")[cols].mean()
    zscore = (z - uni[cols].mean()) / (uni[cols].std() + 1e-9)

    def get(c, col, default=0.0):
        return zscore.loc[c, col] if col in zscore.columns else default

    names, mapping = {}, {}
    used = set()
    ranked = sorted(z.index, key=lambda c: -get(c, "total_clicks") - get(c, "active_days"))
    for c in ranked:
        eng = get(c, "total_clicks") + get(c, "active_days") + get(c, "engagement_score")
        perf = get(c, "avg_quiz_score") + get(c, "accuracy")
        help_z = get(c, "help_requests") + get(c, "hint_usage")
        video = get(c, "video_clicks")
        quiz_rate = get(c, "quiz_attempts") - get(c, "active_days") * 0.3
        if c in used:
            continue
        if eng < -0.5 and perf < 0:
            name = "At-Risk / Disengaged"
        elif eng > 0.4 and perf > 0.3 and help_z < 0.3:
            name = "Independent Deep Learner"
        elif help_z > 0.3 or video > 0.5:
            name = "Guided Learner"
        elif quiz_rate > 0.2:
            name = "Exam-Driven Learner"
        else:
            name = None
        if name and name in used:
            # 同签名簇按投入度追加限定词区分
            qual = "High Engagement" if eng > 0 else "Low Engagement"
            name = f"{name} ({qual})"
        if name is None:
            perf_rank = "high" if perf > 0 else "low"
            eng_rank = "high" if eng > 0 else "low"
            name = f"{'High' if eng_rank == 'high' else 'Low'}-Engagement "
            name += f"{'Strong' if perf_rank == 'high' else 'Weak'}-Performance Learner"
        mapping[c] = name
        used.add(name)
    uni["persona"] = uni["cluster"].map(mapping)
    return uni


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
    loadings.round(3).to_csv(os.path.join(out_dir, "pca_loadings.csv"),
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
    score_df.to_csv(os.path.join(out_dir, "pca_component_scores.csv"), index=False,
                    encoding="utf-8-sig")
    return score_df, loadings, var, pc_names


# ----------------------------------------------------------------------
# 社会网络分析 SNA
# ----------------------------------------------------------------------
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
    for _, r in edge.iterrows():
        A[idx[r.student_id], idx[r.resource_id]] = r.weight
        A[idx[r.resource_id], idx[r.student_id]] = r.weight
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
    lm.to_csv(os.path.join(out_dir, "sna_learner_metrics.csv"), index=False,
              encoding="utf-8-sig")

    rm = pd.DataFrame({
        "resource_id": resources,
        "degree": deg[len(learners):],
        "degree_centrality": deg_c[len(learners):].round(3),
        "pagerank": pr[len(learners):].round(5),
        "course_id": [s.split("-R")[0] for s in resources],
    }).sort_values("pagerank", ascending=False)
    rm.to_csv(os.path.join(out_dir, "sna_resource_metrics.csv"), index=False,
              encoding="utf-8-sig")

    # 路径关联度：学生内时序上资源→资源的转换次数（top 边）
    ev = clicks.sort_values(["student_id", "timestamp"])
    ev["next_r"] = ev.groupby("student_id")["resource_id"].shift(-1)
    path_edges = (ev.dropna(subset=["next_r"])
                  .groupby(["resource_id", "next_r"]).size()
                  .reset_index(name="count").sort_values("count", ascending=False))
    path_edges = path_edges[path_edges["resource_id"] != path_edges["next_r"]]
    path_edges.head(top_k).to_csv(os.path.join(out_dir, "sna_top_paths.csv"),
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
    ax1.set_title("Resource Network (size ∝ PageRank, edges = top access paths)")
    ax1.axis("off")
    ax2.hist(deg_c[:len(learners)], bins=20, color=PALETTE[0], alpha=0.8)
    ax2.set(xlabel="Learner Degree Centrality", ylabel="# Learners",
            title="Learner Degree Centrality Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sna_network.png"), bbox_inches="tight")
    plt.close()

    log(f"  SNA 完成: {len(learners)} 学习者 / {len(resources)} 资源节点; "
        f"PageRank Top 资源: {rm.head(3)['resource_id'].tolist()}")
    return lm, rm, path_edges


# ----------------------------------------------------------------------
# 滞后序列分析 LSA
# ----------------------------------------------------------------------
def lsa_analysis(clicks: pd.DataFrame, out_dir: str):
    """滞后序列分析：点击流行为编码 → lag-1 转换频次 → 调整残差 z 分数
    → 行为转换图谱；并识别 刷课 / 跳步学习 异常模式。"""
    ev = clicks.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    ev["action"] = ev["action"].astype(str).str.lower()
    ev = ev[ev["action"].isin(LSA_ACTIONS)].sort_values(["student_id", "timestamp"])
    ev["prev"] = ev.groupby("student_id")["action"].shift(1)
    tr = (ev.dropna(subset=["prev"]).groupby(["prev", "action"]).size()
            .unstack(fill_value=0)
            .reindex(index=LSA_ACTIONS, columns=LSA_ACTIONS, fill_value=0))
    tr.to_csv(os.path.join(out_dir, "lsa_transition_counts.csv"), encoding="utf-8-sig")

    # 调整残差 z 分数（标准 LSA 显著性检验）
    N = tr.values.sum()
    row = tr.values.sum(1, keepdims=True)
    col = tr.values.sum(0, keepdims=True)
    E = row @ col / max(N, 1)
    denom = np.sqrt(E * (1 - row / max(N, 1)) * (1 - col / max(N, 1)) + 1e-12)
    Z = (tr.values - E) / denom
    z_df = pd.DataFrame(Z, index=tr.index, columns=tr.columns).round(2)
    z_df.to_csv(os.path.join(out_dir, "lsa_transition_zscores.csv"), encoding="utf-8-sig")

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
    plt.savefig(os.path.join(out_dir, "lsa_transition_graph.png"), bbox_inches="tight")
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
    anoms.to_csv(os.path.join(out_dir, "lsa_behavior_patterns.csv"), index=False,
                 encoding="utf-8-sig")
    log(f"  LSA 完成: 行为模式分布 {anoms['lsa_pattern'].value_counts().to_dict()}")
    return anoms, tr, z_df


# ======================================================================
# 五、预测层：成功预测 + 风险预警 + 画像可预测性
# ======================================================================
def run_predictions(uni: pd.DataFrame, cols: list[str], out_dir: str) -> dict:
    res = {}
    y = uni["success"].astype(int)
    X = uni[cols].fillna(uni[cols].median())

    # 1) LogisticRegression 成功预测（data-pipeline 思路）
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                              stratify=y)
    lr = LogisticRegression(max_iter=1000).fit(StandardScaler().fit_transform(X_tr), y_tr)
    proba = lr.predict_proba(StandardScaler().fit(X_tr).transform(X_te))[:, 1]
    res["logreg_acc"] = accuracy_score(y_te, proba > 0.5)
    res["logreg_auc"] = roc_auc_score(y_te, proba)
    coef = pd.Series(lr.coef_[0], index=cols).sort_values(key=abs, ascending=False)
    coef.head(10).to_csv(os.path.join(out_dir, "logreg_coefficients.csv"),
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
    imp.to_csv(os.path.join(out_dir, "rf_feature_importance.csv"), encoding="utf-8-sig")

    # 3) 由行为特征预测画像标签（MSc Thesis predictive_models 思路）
    le = LabelEncoder()
    y_p = le.fit_transform(uni["persona"])
    mask = y_p < np.unique(y_p).__len__()  # 全部样本
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
    from sklearn.metrics import r2_score, mean_absolute_error

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
    imp.to_csv(os.path.join(out_dir, "rf_grade_feature_importance.csv"),
               encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.head(12)[::-1].plot.barh(ax=ax, color=PALETTE[0])
    ax.set(title="RandomForest 特征重要性 — 最终成绩预测 Top 12",
           xlabel="Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rf_feature_importance.png"), bbox_inches="tight")
    plt.close()
    return res


# ----------------------------------------------------------------------
# 灰色 GM(1,1) 小样本趋势预测
# ----------------------------------------------------------------------
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
    ev["day"] = pd.to_datetime(ev["timestamp"]).dt.normalize()
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
    gm.to_csv(os.path.join(out_dir, "gm11_forecast.csv"), index=False,
              encoding="utf-8-sig")
    valid = gm[gm["trend"] != "数据不足"]
    log(f"  GM(1,1) 完成: 有效预测 {len(valid)}/{len(gm)} 人, "
        f"趋势分布 {valid['trend'].value_counts().to_dict()}, "
        f"模型精度分布 {valid['model_grade'].value_counts().to_dict()}")
    return gm


# ======================================================================
# 六、决策层：规则引擎队列 + 教师复核 + 内容推荐
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
        base = persona.split(" (")[0]  # 变体簇复用基础画像的描述与推荐
        recs.append({
            "persona": persona, "n_students": len(group),
            "share_pct": round(100 * len(group) / len(uni), 1),
            "description": PERSONA_DESC.get(base, ""),
            "content_recommendation": CONTENT_REC.get(
                base, "按簇中心特征定制内容策略"),
        })
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------
# AHP + 模糊综合评判：学习质量量化评分
# ----------------------------------------------------------------------
def _ahp_weights(A: np.ndarray):
    """AHP 求权：主特征向量法 + 一致性比率 CR 检验（RI(n=3)=0.58）。"""
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    lam = vals.real[k]
    n = A.shape[0]
    CI = (lam - n) / (n - 1)
    CR = CI / 0.58
    return w, CR


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
    """AHP 定权（学习行为/学习效果/互动参与三维度两两比较）+ 模糊综合评判
    → 每位学习者学习质量量化综合评分（0-100）。"""
    w, cr = _ahp_weights(AHP_MATRIX)
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
    qs.to_csv(os.path.join(out_dir, "learning_quality_scores.csv"), index=False,
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
    plt.savefig(os.path.join(out_dir, "quality_score_distribution.png"),
                bbox_inches="tight")
    plt.close()

    log(f"  学习质量评分: 均值={qs['quality_score'].mean():.1f}, "
        f"等级分布 {qs['fuzzy_best_level'].value_counts().to_dict()}")
    return qs, {"weights": dict(zip(AHP_DIMS, [float(x) for x in np.round(w, 3)])),
                "cr": round(cr, 3)}


# ======================================================================
# 六B、扩展分析：基础统计 / 时序模式 / Apriori / 场景对比 / 相关性 / SD 仿真
# ======================================================================
def basic_stats_analysis(uni: pd.DataFrame, out_dir: str) -> dict:
    """第一部分：人口属性频数统计 + 基础行为指标基线 + 静态属性×行为交叉对比。"""
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
    demo_df.to_csv(os.path.join(out_dir, "demographics_distribution.csv"),
                   index=False, encoding="utf-8-sig")
    core = {c: demo_df[demo_df["dimension"] == c].iloc[0]["category"]
            for c in demo_cols}
    log(f"  人口结构: 核心群体 {core}")

    # 2) 基础行为指标基线
    baseline_cols = [c for c in ["login_freq_per_week", "study_hours_per_week",
                                 "video_completion_rate", "submission_rate",
                                 "forum_posts", "total_clicks", "active_days",
                                 "avg_session_min"] if c in uni.columns]
    desc = uni[baseline_cols].describe().T[["mean", "50%", "min", "max", "std"]]
    desc.columns = ["均值", "中位数", "最小值", "最大值", "标准差"]
    desc.round(2).to_csv(os.path.join(out_dir, "behavior_baseline.csv"),
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
    cross.to_csv(os.path.join(out_dir, "group_behavior_comparison.csv"),
                 index=False, encoding="utf-8-sig")

    # 可视化：教育背景 × 行为差异（代表性交叉图）
    if "education" in uni.columns:
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
        plt.savefig(os.path.join(out_dir, "demographics_cross.png"), bbox_inches="tight")
        plt.close()
    return {"core_groups": core, "demo_df": demo_df, "cross": cross}


def temporal_analysis(clicks: pd.DataFrame, out_dir: str):
    """第二部分-1：登录时间分布（早/中/晚/深夜）、周度周期性、时间行为模式标记。"""
    ev = clicks.copy()
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
    plt.savefig(os.path.join(out_dir, "temporal_patterns.png"), bbox_inches="tight")
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
    pat_df.to_csv(os.path.join(out_dir, "temporal_student_patterns.csv"),
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
    """第二部分-3：学习路径关联规则挖掘（Apriori），并统计负向行为（只看短视频跳过慕课）。"""
    ev = clicks.copy()
    ev["action"] = ev["action"].astype(str).str.lower()
    labels = {"video": "看视频", "resource": "读资源", "quiz": "做测验",
              "forum": "论坛互动", "help": "求助"}
    trans = []
    for sid, g in ev.groupby("student_id"):
        items = {labels.get(a, a) for a in g["action"].unique()}
        if "resource_type" in g.columns:
            rts = g["resource_type"].dropna().unique().tolist()
            rt_map = {"mooc": "慕课长视频", "micro": "微课", "short": "短视频"}
            items |= {rt_map.get(r, r) for r in rts}
        if items:
            trans.append(frozenset(items))
    rules = _apriori(trans, min_support, min_conf)
    rules.to_csv(os.path.join(out_dir, "apriori_rules.csv"), index=False,
                 encoding="utf-8-sig")

    # 负向行为：只看短视频、未接触慕课长视频
    used = {}
    for t in trans:
        for it in t:
            used.setdefault(it, set()).add(id(t))
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
    ax.set(xlabel="Lift", title="学习路径关联规则 Top10（Apriori）")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "apriori_rules.png"), bbox_inches="tight")
    plt.close()
    summary = {"short_only_pct": round(100 * short_only / max(len(trans), 1), 1),
               "top_rules": top.head(3).to_dict("records")}
    log(f"  Apriori 规则 {len(rules)} 条; 只看短视频跳过完整慕课占比: "
        f"{summary['short_only_pct']}%")
    return rules, summary


def scene_comparison(clicks: pd.DataFrame, out_dir: str):
    """第二部分-4：慕课长视频 / 微课 / 短视频三类场景的完播率、倍速使用率、拖拽行为对比。"""
    vid = clicks[clicks["action"] == "video"].copy()
    if "resource_type" not in vid.columns or vid["resource_type"].dropna().empty:
        log("  场景对比: 点击流缺少 resource_type 字段，跳过")
        return pd.DataFrame(), pd.DataFrame(), {}
    vid = vid.dropna(subset=["resource_type"])
    rt_names = {"mooc": "慕课长视频", "micro": "微课", "short": "短视频"}
    vid["场景"] = vid["resource_type"].map(rt_names)
    tbl = vid.groupby("场景").agg(
        播放次数=("completed", "count"),
        完播率=("completed", "mean"),
        倍速使用率=("speed_used", lambda s: (s > 1).mean()),
        平均拖拽次数=("seek_count", "mean"),
    ).round(3)
    tbl["播放占比"] = (tbl["播放次数"] / tbl["播放次数"].sum()).round(3)
    tbl.to_csv(os.path.join(out_dir, "scene_comparison.csv"), encoding="utf-8-sig")

    # 逐学生完播率（供相关分析使用）
    per_stu = vid.groupby(["student_id", "场景"])["completed"].mean().unstack(fill_value=np.nan)
    per_stu["video_completion_rate"] = vid.groupby("student_id")["completed"].mean().round(3)
    per_stu["preferred_scene"] = vid.groupby("student_id")["场景"] \
        .agg(lambda s: s.value_counts().idxmax())
    per_stu = per_stu.reset_index()
    per_stu.to_csv(os.path.join(out_dir, "scene_per_student.csv"), index=False,
                   encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [("完播率", "completed"), ("倍速使用率", "speed"), ("平均拖拽次数", "seek")]
    plot_tbl = tbl.reset_index()
    axes[0].bar(plot_tbl["场景"], plot_tbl["完播率"], color=PALETTE[:3])
    axes[0].set(title="各场景完播率")
    axes[1].bar(plot_tbl["场景"], plot_tbl["倍速使用率"], color=PALETTE[:3])
    axes[1].set(title="各场景倍速使用率")
    axes[2].bar(plot_tbl["场景"], plot_tbl["平均拖拽次数"], color=PALETTE[:3])
    axes[2].set(title="各场景平均拖拽次数")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scene_comparison.png"), bbox_inches="tight")
    plt.close()
    log(f"  场景对比: {tbl[['完播率', '倍速使用率', '平均拖拽次数']].to_dict('index')}")
    return tbl, per_stu, tbl[["完播率", "倍速使用率", "平均拖拽次数"]].to_dict("index")


def correlation_analysis(uni: pd.DataFrame, out_dir: str):
    """第三部分-1：行为指标与最终成绩/结课认证的皮尔逊相关分析（含显著性）。"""
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
    corr.to_csv(os.path.join(out_dir, "correlation_results.csv"), index=False,
                encoding="utf-8-sig")
    g = corr[corr["target"] == "final_grade"].sort_values("pearson_r", key=abs,
                                                          ascending=False)
    top = g.head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE[1] if r > 0 else PALETTE[2] for r in top["pearson_r"]]
    ax.barh(top["indicator"][::-1], top["pearson_r"][::-1], color=colors[::-1])
    ax.set(xlabel="Pearson r（与最终成绩）", title="行为指标与学业效果相关性 Top10")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_top_factors.png"), bbox_inches="tight")
    plt.close()
    log(f"  相关性 Top3（成绩）: {top.head(3)[['indicator', 'pearson_r']].to_dict('records')}")
    return corr, top.head(5)


def heterogeneity_analysis(uni: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """第三部分-2：性别/地区/教育背景是否调节"学习时长-成绩"相关性（异质性分析）。"""
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
    het.to_csv(os.path.join(out_dir, "heterogeneity_analysis.csv"), index=False,
               encoding="utf-8-sig")
    log(f"  异质性分析: 各群体学习时长-成绩相关系数范围 "
        f"[{het['r_时长-成绩'].min():.3f}, {het['r_时长-成绩'].max():.3f}]")
    return het


def engagement_tier_analysis(uni: pd.DataFrame, out_dir: str):
    """第三部分-3：学习投入度低/中/高三分组，量化投入对学习效果的提升幅度。"""
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
    tbl.to_csv(os.path.join(out_dir, "engagement_tier_analysis.csv"), encoding="utf-8-sig")
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
    plt.savefig(os.path.join(out_dir, "engagement_tiers.png"), bbox_inches="tight")
    plt.close()
    log(f"  投入分层: 高投入组较低投入组 完成率 +{tbl['完成率提升幅度'].iloc[2]}%, "
        f"成绩 +{tbl['成绩提升幅度'].iloc[2]} 分")
    return uni, tbl


def performance_model_extended(uni: pd.DataFrame, cols: list[str], out_dir: str):
    """第四部分-1：RF/GBDT 学业表现预测（成绩等级 + 是否结课），准确率/召回率 + Top5 关键行为，
    并输出高辍学风险学生早期预警名单。"""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import recall_score, precision_score

    uni = uni.copy()
    X = uni[cols].fillna(uni[cols].median())
    uni["grade_band"] = pd.cut(uni["final_grade"].clip(lower=0), [-0.1, 60, 70, 85, 100.01],
                               labels=["D", "C", "B", "A"])
    uni["certified"] = (uni["completion_rate"] >= 0.75).astype(int)
    has_band = uni["grade_band"].notna()
    X_tr, X_te, u_tr, u_te = train_test_split(X[has_band], uni[has_band],
                                              test_size=0.3, random_state=SEED)

    res, models = {}, {}
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
    best = max(["RandomForest", "GradientBoosting"],
               key=lambda m: res[f"{m}_grade_acc"])
    res["best_model"] = best

    # Top5 关键行为特征（以最佳模型的等级分类为准）
    imp = pd.Series(models[best].feature_importances_, index=cols).sort_values(
        ascending=False)
    res["top5_features"] = imp.head(5).round(3).to_dict()
    fig, ax = plt.subplots(figsize=(9, 5))
    imp.head(10)[::-1].plot.barh(ax=ax, color=PALETTE[0])
    ax.set(title=f"学业结果预测特征重要性 Top10（{best} 成绩等级模型）")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_feature_importance.png"), bbox_inches="tight")
    plt.close()

    # 早期预警：结课概率最低的学生（高辍学风险）
    proba = rf_cert.predict_proba(X)[:, list(rf_cert.classes_).index(1)]
    ew = uni[["student_id", "persona", "region", "completion_rate", "final_grade"]].copy()
    ew["cert_prob"] = proba.round(3)
    ew["risk_level"] = pd.cut(proba, [0, 0.2, 0.4, 1], labels=["高", "中", "低"])
    ew = ew.sort_values("cert_prob").head(50)
    ew.to_csv(os.path.join(out_dir, "early_warning_students.csv"), index=False,
              encoding="utf-8-sig")
    log(f"  扩展建模: 最佳模型={best} 成绩等级 acc={res[best + '_grade_acc']:.3f} "
        f"recall={res[best + '_grade_recall']:.3f}; 认证 acc={res['cert_acc']:.3f}; "
        f"Top5 关键行为: {list(res['top5_features'])}")
    return res, ew


def sd_simulation(out_dir: str, weeks: int = 52) -> dict:
    """第四部分-3：系统动力学（SD）行为演化仿真。
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
    plt.savefig(os.path.join(out_dir, "sd_simulation.png"), bbox_inches="tight")
    plt.close()
    pd.DataFrame({k: v["final_engagement"] for k, v in res.items()},
                 index=["final_engagement"]).T.to_csv(
        os.path.join(out_dir, "sd_simulation_results.csv"), encoding="utf-8-sig")
    best = max(res, key=lambda k: res[k]["final_engagement"])
    log(f"  SD 仿真: 52 周后最优策略 = {best}（投入水平 {res[best]['final_engagement']}）")
    return {"results": res, "best_strategy": best}


def generate_optimization_suggestions(uni, recs, scene_tbl, corr_top, sd_info,
                                      early_warning, out_dir) -> pd.DataFrame:
    """第五部分：基于全部分析结果输出可落地的平台优化策略。"""
    rows = []
    # 1) 画像差异化推送与干预
    for _, r in recs.iterrows():
        base = r["persona"].split(" (")[0]
        extra_map = {
            "At-Risk / Disengaged": "推送轻量化短视频学习内容 + 每日微任务签到，提升留存率",
            "Exam-Driven Learner": "考前冲刺包 + 间隔重复记忆卡片推送",
            "Guided Learner": "微课 + 即时反馈测验组合推送，逐步过渡到自主学习",
            "Independent Deep Learner": "开放项目与进阶资源优先推送，邀请担任社区助教",
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
                         "辅导员 48 小时内一对一介入；预警名单见 early_warning_students.csv"})
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
    sug.to_csv(os.path.join(out_dir, "optimization_suggestions.csv"), index=False,
               encoding="utf-8-sig")
    log(f"  优化建议: 输出 {len(sug)} 条落地策略")
    return sug


# ======================================================================
# 七、输出层：报告
# ======================================================================
def _df_markdown(df: pd.DataFrame) -> list[str]:
    """DataFrame 转 Markdown 表格；缺少 tabulate 时降级为纯文本表格。"""
    try:
        return df.to_markdown(index=False).splitlines()
    except ImportError:
        return df.to_string(index=False).splitlines()


def write_report(uni, queues, recs, metrics, out_dir, extras=None) -> str:
    extras = extras or {}
    lines = [
        "# 学习者画像分析报告",
        f"生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"## 数据概览",
        f"- 学生总数: {len(uni)}，数据来源: {uni['source'].value_counts().to_dict()}",
        f"- 聚类数 K = {metrics['k']}，Silhouette = {metrics['silhouette']:.3f}",
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
            cs = ", ".join(f"{r.indicator}(r={r.pearson_r})"
                           for r in ex["corr_top"].itertuples())
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
            lines.append(f"- 优化建议: {ex['n_suggestions']} 条（详见 optimization_suggestions.csv）")
    lines += _df_markdown(recs)
    lines += ["", "## 干预队列统计", ""]
    qcount = queues["queue"].value_counts()
    lines += [f"- {q}: {n} 人" for q, n in qcount.items()]
    lines += ["", "## 需要教师复核的学生（review/support 队列示例，前 20）", ""]
    sub = queues[queues["queue"].isin(["support", "review"])].head(20)
    lines += _df_markdown(sub)
    report = "\n".join(lines)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    return report


# ======================================================================
# 主流程
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="统一学习行为分析与学习者画像管线")
    ap.add_argument("--data-dir", default="data", help="真实数据扫描目录")
    ap.add_argument("--out-dir", default="out", help="输出目录")
    ap.add_argument("--k", type=int, default=0,
                    help="聚类数（0 = 自动：肘部拐点 + 轮廓系数综合选择）")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                    help="计算设备：auto 自动探测 GPU（PyTorch CUDA / RAPIDS cuML），"
                         "gpu 强制 GPU（无后端则回退 CPU），cpu 强制 CPU")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    global log_handle
    log_handle = open(os.path.join(args.out_dir, "pipeline_log.jsonl"), "a",
                      encoding="utf-8")
    setup_cjk_font()
    dev = detect_device(args.device)
    if dev["backend"] == "cpu":
        log("  计算设备: CPU（sklearn 多核并行，--device gpu 可启用 GPU 加速）")
        if dev["pref"] == "gpu":
            log("  警告: 未检测到可用 GPU 后端（需 PyTorch+CUDA 或 RAPIDS cuML），已回退 CPU")
    else:
        log(f"  计算设备: GPU[{dev.get('device_name') or 'CUDA'}]，"
            f"KMeans / 轮廓系数 / 质心分配使用 {dev['backend']} 后端加速")

    log("=" * 60)
    log("  统一学习行为分析管线启动")
    log("=" * 60)

    log("[1/9] 数据层：多源加载与统一（含人口属性）")
    uni = build_unified_dataset(args)
    clicks = build_clickstream(args.data_dir)

    log("[2/9] 特征层：EDA 与清洗完成")
    cols = select_cluster_columns(uni)
    eda_summary(uni, args.out_dir)

    log("[3/9] 分析层：命名主成分 + K-Means 聚类画像")
    pc_scores, loadings, pc_var, pc_names = pca_named_components(uni, cols, args.out_dir)
    uni = uni.merge(pc_scores, on="student_id", how="left")
    uni, engine, scaler, cols, metrics = cluster_personas(uni, cols, args.k, args.out_dir)
    uni = label_personas(uni, cols)

    log("[4/9] 挖掘层：SNA + LSA + 时序模式 + Apriori 路径挖掘 + 场景对比")
    sna_learners, sna_resources, sna_paths = sna_analysis(clicks, args.out_dir)
    anoms, lsa_tr, lsa_z = lsa_analysis(clicks, args.out_dir)
    uni = uni.merge(anoms, on="student_id", how="left")
    uni["lsa_pattern"] = uni["lsa_pattern"].fillna("无点击流数据")
    basic_placeholder = None
    pat_df, time_dist = temporal_analysis(clicks, args.out_dir)
    uni = uni.merge(pat_df, on="student_id", how="left")
    rules, apriori_summary = apriori_path_mining(clicks, args.out_dir)
    scene_tbl, scene_per_stu, scene_dict = scene_comparison(clicks, args.out_dir)
    if not scene_per_stu.empty:
        keep_cols = ["student_id", "video_completion_rate", "preferred_scene"]
        uni = uni.merge(scene_per_stu[keep_cols], on="student_id", how="left")
    basic = basic_stats_analysis(uni, args.out_dir)

    log("[5/9] 关联层：皮尔逊相关 + 异质性分析 + 投入度分层")
    corr_df, corr_top = correlation_analysis(uni, args.out_dir)
    het = heterogeneity_analysis(uni, args.out_dir)
    uni, tier_tbl = engagement_tier_analysis(uni, args.out_dir)

    log("[6/9] 预测层：RF/GBDT 学业表现 + 风险预警 + GM(1,1) 趋势预测")
    metrics.update(rf_performance_model(uni, cols, args.out_dir))
    model_ext, early_warning = performance_model_extended(uni, cols, args.out_dir)
    metrics.update(run_predictions(uni, cols, args.out_dir))
    gm = gm11_forecast(clicks, args.out_dir)

    log("[7/9] 评价层：AHP 模糊综合评判 + 系统动力学仿真")
    quality, ahp_info = ahp_fuzzy_scores(uni, args.out_dir)
    uni = uni.merge(quality, on="student_id", how="left")
    sd_info = sd_simulation(args.out_dir)

    log("[8/9] 决策层：干预队列 + 内容推荐 + 优化建议")
    queues = rule_queues(uni)
    recs = generate_recommendations(uni)
    suggestions = generate_optimization_suggestions(
        uni, recs, scene_tbl, corr_top, sd_info, early_warning, args.out_dir)

    log("[9/9] 输出层：导出结果与报告")
    uni.to_csv(os.path.join(args.out_dir, "learner_profiles.csv"), index=False,
               encoding="utf-8-sig")
    queues.to_csv(os.path.join(args.out_dir, "queues.csv"), index=False,
                  encoding="utf-8-sig")
    recs.to_csv(os.path.join(args.out_dir, "persona_recommendations.csv"), index=False,
                encoding="utf-8-sig")
    # 教师复核记录（闭环起点，初始均为 pending）
    review = queues[queues["queue"].isin(["support", "review"])].copy()
    review["review_key"] = review["student_id"] + "|" + review["suggested_action"]
    review["decision"] = "pending"
    review.to_csv(os.path.join(args.out_dir, "teacher_reviews.csv"), index=False,
                  encoding="utf-8-sig")
    # 汇总新方法结果供报告使用
    flat_z = lsa_z.stack().sort_values(key=abs, ascending=False)
    top_trans = [f"{a}→{b}(z={z})" for (a, b), z in flat_z.head(5).items()]
    extras = {
        "pc_names": pc_names,
        "pc_var": [f"{v:.1f}%" for v in pc_var],
        "sna_top_resources": sna_resources.head(5)["resource_id"].tolist(),
        "lsa_patterns": uni["lsa_pattern"].value_counts().to_dict(),
        "lsa_top_transitions": top_trans,
        "gm_trends": gm[gm["trend"] != "数据不足"]["trend"].value_counts().to_dict(),
        "ahp_weights": ahp_info["weights"],
        "ahp_cr": ahp_info["cr"],
        "quality_mean": round(quality["quality_score"].mean(), 1),
        "quality_levels": quality["fuzzy_best_level"].value_counts().to_dict(),
        # 五大扩展模块摘要
        "core_groups": basic["core_groups"],
        "time_patterns": time_dist,
        "apriori_top": apriori_summary["top_rules"],
        "short_only_pct": apriori_summary["short_only_pct"],
        "scene": scene_dict,
        "corr_top": corr_top,
        "tier_lift": f"完成率 +{tier_tbl['完成率提升幅度'].iloc[2]}%, "
                     f"成绩 +{tier_tbl['成绩提升幅度'].iloc[2]} 分",
        "model_ext": model_ext,
        "early_warning_count": int((early_warning["risk_level"] == "高").sum()),
        "sd_best": sd_info["best_strategy"],
        "sd_final": sd_info["results"][sd_info["best_strategy"]]["final_engagement"],
        "n_suggestions": len(suggestions),
    }
    report = write_report(uni, queues, recs, metrics, args.out_dir, extras)

    log("\n" + "=" * 60)
    log("  LEARNER PERSONA SUMMARY")
    log("=" * 60)
    for _, r in recs.iterrows():
        log(f"\n  {r['persona']}  (n={r['n_students']}, {r['share_pct']}%)")
        log(f"  Profile : {r['description']}")
        log(f"  Content : {r['content_recommendation']}")

    log("\n输出文件已保存至 out/:")
    for f in sorted(os.listdir(args.out_dir)):
        log(f"  - {f}")
    log_handle.close()
    print("\n✓ 管线执行完成。完整报告见 out/report.md")


if __name__ == "__main__":
    main()
