#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — 统一学习行为分析与学习者画像管线（一体化入口）

整合了 7 个开源学习分析项目的方法论，形成一条完整管线：

  数据层  多源适配：学生汇总表 CSV / 交互日志 JSONL / 答题事件流 / 代码提交
          —— 内置各类数据源的合成生成器，扫描 data/ 目录自动加载真实数据
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
    python main.py                    # 使用合成数据 + data/ 目录中可发现的真实数据
    python main.py --n 600 --k 4      # 指定合成样本量与聚类数
    python main.py --data-dir data --out-dir out
"""

import argparse
import ast
import glob
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, silhouette_score,
                             adjusted_rand_score, f1_score)
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
# 一、数据层：合成数据生成器（各参考项目的数据形态）
# ======================================================================
def synthetic_oulad(n: int, seed: int = SEED) -> pd.DataFrame:
    """源形态 A：学生级行为汇总表（OULAD 风格， Learner-Persona-Detection）。"""
    rng = np.random.default_rng(seed)
    types = rng.choice([0, 1, 2, 3], size=n, p=[0.25, 0.30, 0.25, 0.20])
    params = {
        0: dict(total_clicks=(600, 1400), active_days=(50, 90), avg_session_min=(35, 70),
                forum_posts=(8, 25), resource_clicks=(150, 400), video_clicks=(50, 150),
                quiz_attempts=(1, 3), avg_quiz_score=(72, 95), assignment_submit_lag=(1, 5),
                help_requests=(0, 4), login_freq_per_week=(4, 7)),
        1: dict(total_clicks=(400, 900), active_days=(35, 70), avg_session_min=(20, 45),
                forum_posts=(3, 12), resource_clicks=(60, 180), video_clicks=(120, 350),
                quiz_attempts=(2, 5), avg_quiz_score=(60, 80), assignment_submit_lag=(0, 3),
                help_requests=(4, 12), login_freq_per_week=(3, 6)),
        2: dict(total_clicks=(200, 600), active_days=(20, 45), avg_session_min=(15, 35),
                forum_posts=(0, 5), resource_clicks=(30, 100), video_clicks=(20, 80),
                quiz_attempts=(4, 9), avg_quiz_score=(55, 78), assignment_submit_lag=(-1, 1),
                help_requests=(1, 6), login_freq_per_week=(1, 4)),
        3: dict(total_clicks=(10, 200), active_days=(3, 20), avg_session_min=(3, 18),
                forum_posts=(0, 2), resource_clicks=(5, 40), video_clicks=(2, 30),
                quiz_attempts=(0, 3), avg_quiz_score=(25, 58), assignment_submit_lag=(-5, 0),
                help_requests=(0, 3), login_freq_per_week=(0, 2)),
    }
    records = []
    for t in types:
        p = params[t]
        rec = {}
        for k, (lo, hi) in p.items():
            rec[k] = rng.integers(lo, hi) if isinstance(lo, int) and isinstance(hi, int) \
                else rng.uniform(lo, hi)
        rec["latent_type"] = t
        records.append(rec)
    df = pd.DataFrame(records)
    df["student_id"] = [f"OULAD-{i:04d}" for i in range(n)]
    df["source"] = "oulad_synthetic"
    return df


def synthetic_interaction(n: int, seed: int = SEED) -> pd.DataFrame:
    """源形态 B：个性化学习系统的学习者指标（ai-personalized-learning 形态）。"""
    rng = np.random.default_rng(seed + 1)
    df = pd.DataFrame({
        "student_id": [f"INT-{i:04d}" for i in range(n)],
        "accuracy": rng.uniform(0.3, 0.98, n),
        "avg_response_time": rng.uniform(8, 120, n),
        "hesitation_rate": rng.uniform(0.05, 0.6, n),
        "engagement_score": rng.uniform(0.2, 1.0, n),
        "hint_usage": rng.integers(0, 15, n),
        "source": "interaction_logs",
    })
    # 与 OULAD 特征对齐的一部分字段（模拟真实系统间的重叠）
    df["avg_quiz_score"] = (df["accuracy"] * 100).round(1)
    df["help_requests"] = df["hint_usage"]
    df["quiz_attempts"] = rng.integers(1, 10, n)
    df["active_days"] = (df["engagement_score"] * 90).astype(int)
    df["total_clicks"] = (df["engagement_score"] * 1500).astype(int)
    return df


def synthetic_chatbot_events(n_students: int, seed: int = SEED) -> pd.DataFrame:
    """源形态 C：答题/聊天机器人事件流（MSc Thesis 形态）。
    action_type: 1=答对 2=答错 3=跳过 9=升级 10=降级"""
    rng = np.random.default_rng(seed + 2)
    rows = []
    base = datetime(2026, 9, 1)
    for i in range(n_students):
        uid = f"CHAT-{i:04d}"
        n_events = rng.integers(40, 400)
        level, t = 1, 0
        for _ in range(n_events):
            t += int(rng.integers(30, 3 * 86400))
            act = rng.choice([1, 2, 3, 9, 10], p=[0.5, 0.28, 0.12, 0.05, 0.05])
            if act == 9:
                level += 1
            elif act == 10 and level > 1:
                level -= 1
            rows.append({"student_id": uid, "timestamp": base + timedelta(seconds=t),
                         "action_type": int(act), "question_id": f"Q{rng.integers(1, 300)}",
                         "response_time": float(rng.uniform(5, 180))})
    return pd.DataFrame(rows)


def events_to_features(events: pd.DataFrame) -> pd.DataFrame:
    """事件流 → 学生级特征（MSc Thesis 的特征工程思路）。"""
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["timestamp"]).dt.date
    span = (ev["date"].max() - ev["date"].min()).days + 1
    g = ev.groupby("student_id")
    feat = pd.DataFrame({
        "total_events": g.size(),
        "success_rate": g.apply(
            lambda d: (d["action_type"] == 1).mean(), include_groups=False),
        "skip_rate": g.apply(
            lambda d: (d["action_type"] == 3).mean(), include_groups=False),
        "avg_response_time": g["response_time"].mean(),
        "active_days": g["date"].nunique(),
        "unique_days_ratio": g["date"].nunique() / max(span, 1),
        "avg_interactions_per_day": g.size() / g["date"].nunique(),
        "level_ups": g.apply(
            lambda d: (d["action_type"] == 9).sum(), include_groups=False),
        "level_downs": g.apply(
            lambda d: (d["action_type"] == 10).sum(), include_groups=False),
    }).reset_index()
    feat["quiz_attempts"] = feat["total_events"]
    feat["avg_quiz_score"] = feat["success_rate"] * 100
    feat["help_requests"] = feat["level_downs"]
    feat["source"] = "chatbot_events"
    return feat


def synthetic_code_submissions(n: int, seed: int = SEED) -> pd.DataFrame:
    """源形态 D：编程课代码提交（user2code2vec 形态）—— 生成代码文本再提特征。"""
    rng = np.random.default_rng(seed + 3)
    skill = rng.uniform(0.1, 1.0, n)
    rows = []
    templates = [
        "def solve(x):\n{body}\n    return result\n",
        "class Task:\n    def __init__(self):\n{body}\n",
    ]
    for i, s in enumerate(skill):
        n_lines = int(5 + s * 40)
        n_funcs = int(1 + s * 6)
        body_lines = []
        for _ in range(n_lines):
            r = rng.random()
            if r < s * 0.25:
                body_lines.append("    # " + rng.choice(["handle edge case", "optimize loop",
                                                        "validate input"]))
            elif r < 0.6:
                body_lines.append(f"    value_{rng.integers(0, 99)} = x * {rng.integers(1, 10)}")
            else:
                body_lines.append("    pass")
        code = templates[i % 2].format(body="\n".join(body_lines))
        code = code.replace("solve", "solve", 1)
        for fi in range(1, n_funcs):
            code += f"\ndef helper_{fi}(x):\n    return x\n"
        rows.append({"student_id": f"CODE-{i:04d}", "code": code, "grade": s * 100})
    return pd.DataFrame(rows)


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


def synthetic_clickstream(n_students: int, seed: int = SEED) -> pd.DataFrame:
    """源形态 E：点击流事件（SNA / LSA / GM(1,1) 的输入）。
    action ∈ video/resource/quiz/forum/help；内置三种行为模式：
    normal（混合浏览）/ skim（刷课：高占比连续视频）/ skip（跳步：未学先测）。
    学生 ID 与 OULAD 合成源对齐（OULAD-{i:04d}），便于特征合并。"""
    rng = np.random.default_rng(seed + 4)
    rows = []
    base = datetime(2026, 9, 7)
    n_courses = 5
    for i in range(n_students):
        uid = f"OULAD-{i:04d}"
        mode = "normal" if i % 5 < 3 else ("skim" if i % 5 == 3 else "skip")
        course = f"C{i % n_courses}"
        t, n_ev = 0, int(rng.integers(40, 120))
        for _ in range(n_ev):
            if mode == "skim":
                act = "video" if rng.random() < 0.8 else rng.choice(LSA_ACTIONS)
            elif mode == "skip":
                act = "quiz" if rng.random() < 0.6 else \
                    rng.choice(["video", "quiz", "forum"], p=[0.3, 0.4, 0.3])
            else:
                act = rng.choice(LSA_ACTIONS, p=[0.35, 0.30, 0.15, 0.12, 0.08])
            t += int(rng.integers(1800, 30 * 3600))
            rows.append({"student_id": uid, "timestamp": base + timedelta(seconds=t),
                         "action": act,
                         "resource_id": f"{course}-R{rng.integers(1, 13)}",
                         "course_id": course})
    return pd.DataFrame(rows)


def load_clickstream(data_dir: str, n_students: int) -> pd.DataFrame:
    """优先加载 data/clickstream.csv（须含 student_id/timestamp/action 列），
    否则使用内置合成点击流。"""
    path = os.path.join(data_dir, "clickstream.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df = normalize_columns(df)
            if {"student_id", "timestamp", "action"}.issubset(df.columns):
                df["resource_id"] = df.get("resource_id", "R0")
                df["course_id"] = df.get("course_id", "C0")
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                log(f"  已加载真实点击流 clickstream.csv  shape={df.shape}")
                return df
            log("  clickstream.csv 缺少必需列（student_id/timestamp/action），改用合成数据")
        except Exception as e:
            log(f"  读取 clickstream.csv 失败: {e}，改用合成数据")
    return synthetic_clickstream(n_students)


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
    """合并所有数据源 → 统一特征表（union 列，缺失由 clean_features 填充）。"""
    frames = [synthetic_oulad(args.n, SEED), synthetic_interaction(max(args.n // 2, 50), SEED),
              events_to_features(synthetic_chatbot_events(max(args.n // 4, 30), SEED)),
              code_to_features(synthetic_code_submissions(max(args.n // 4, 30), SEED))]
    frames += load_real_data(args.data_dir)
    parts = []
    for df in frames:
        keep = ["student_id", "source"] + [c for c in ALL_FEATURES if c in df.columns]
        extra = [c for c in ["latent_type", "success_status", "overall_score", "grade",
                             "learner_type", "study_hours", "attendance_rate", "code_calls",
                             "completion_rate", "final_grade"]
                 if c in df.columns]
        parts.append(df[keep + extra])
    uni = pd.concat(parts, ignore_index=True, sort=False)
    # 派生标签：优先使用真实标签列，否则按分数阈值派生 success
    if "success_status" in uni.columns:
        uni["success"] = uni["success_status"].astype(str).str.lower().isin(
            ["successful", "success", "1", "true", "pass"])
    else:
        score = uni["avg_quiz_score"]
        if "overall_score" in uni.columns:
            score = score.fillna(uni["overall_score"])
        uni["success"] = score >= np.nanmedian(score)
    # RF 学业表现预测的目标变量：若真实数据未提供则按行为派生
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
    log(f"\n统一数据集: {uni.shape[0]} 名学生 × {uni.shape[1]} 列, "
        f"来源分布: {uni['source'].value_counts().to_dict()}")
    return uni


# ======================================================================
# 四、分析层：EDA + 聚类画像
# ======================================================================
def select_cluster_columns(uni: pd.DataFrame, min_fill: float = 0.6) -> list[str]:
    fill_ratio = uni[ALL_FEATURES].notna().mean()
    cols = fill_ratio[fill_ratio >= min_fill].index.tolist()
    log(f"  聚类特征（缺失率 ≤ {1 - min_fill:.0%}）: {cols}")
    return cols


def eda_summary(uni: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    desc = uni[ALL_FEATURES].describe().T.round(2)
    desc["missing"] = uni[ALL_FEATURES].isna().sum()
    desc.to_csv(os.path.join(out_dir, "eda_summary.csv"), encoding="utf-8-sig")
    return desc


def cluster_personas(uni: pd.DataFrame, cols: list[str], k: int, out_dir: str):
    """肘部法 + 轮廓系数选 K → KMeans → PCA / 雷达 / 热力图（当前仓库管线风格）。"""
    X_raw = uni[cols].values.astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # --- 选 K ---
    k_range = range(2, 8)
    inertias, sils = [], []
    for kk in k_range:
        km = KMeans(n_clusters=kk, random_state=SEED, n_init=10).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))
    if k <= 0:
        k = list(k_range)[int(np.argmax(sils))]
        log(f"  自动选择 K = {k}（轮廓系数 {max(sils):.3f}）")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Selecting Optimal Number of Clusters", fontsize=14, fontweight="bold")
    axes[0].plot(list(k_range), inertias, "o-", color=PALETTE[0])
    axes[0].set(xlabel="K", ylabel="Inertia", title="Elbow Method")
    axes[1].plot(list(k_range), sils, "s-", color=PALETTE[1])
    axes[1].axvline(k, color=PALETTE[2], ls="--", label=f"K = {k}")
    axes[1].set(xlabel="K", ylabel="Silhouette", title="Silhouette Analysis")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elbow_silhouette.png"), bbox_inches="tight")
    plt.close()

    # --- 最终聚类 ---
    km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    if "latent_type" in uni.columns and uni["latent_type"].notna().any():
        m = uni["latent_type"].notna().values
        ari = adjusted_rand_score(uni.loc[m, "latent_type"], labels[m])
    else:
        ari = float("nan")
    log(f"  K-Means(K={k})  Silhouette = {sil:.3f}  ARI(vs latent) = {ari:.3f}")
    uni["cluster"] = labels

    # --- PCA 散点 ---
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100
    proj = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    proj["cluster"] = labels
    proj.to_csv(os.path.join(out_dir, "pca_projection.csv"), index=False)
    log(f"  PCA 方差解释率: PC1={var[0]:.1f}%, PC2={var[1]:.1f}%")
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(k):
        m = labels == c
        ax.scatter(X_pca[m, 0], X_pca[m, 1], c=PALETTE[c % len(PALETTE)],
                   label=f"Cluster {c}", alpha=0.75, s=55, edgecolors="white", linewidths=0.4)
    ax.set(xlabel=f"PC1 ({var[0]:.1f}%)", ylabel=f"PC2 ({var[1]:.1f}%)",
           title="Learner Personas — PCA Projection")
    ax.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_clusters.png"), bbox_inches="tight")
    plt.close()

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
        ncol = min(k, 3)
        nrow = int(np.ceil(k / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow),
                                 subplot_kw=dict(polar=True), squeeze=False)
        fig.suptitle("Behavioral Radar — Learner Personas", fontsize=14, fontweight="bold")
        for idx, ax in enumerate(axes.flat):
            if idx >= k:
                ax.remove()
                continue
            vals = norm.loc[idx].tolist() + norm.loc[idx].tolist()[:1]
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

    return uni, km, scaler, cols, {"silhouette": sil, "ari": ari, "k": k}


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
    rf_full = RandomForestClassifier(n_estimators=200, random_state=SEED)
    rf_early = RandomForestClassifier(n_estimators=200, random_state=SEED)
    cv_full = cross_val_score(rf_full, X, y, cv=5, scoring="roc_auc")
    cv_early = cross_val_score(rf_early, X[early_cols], y, cv=5, scoring="roc_auc")
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
    cv_p = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=SEED),
                           X, y_p, cv=5, scoring="accuracy")
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
    rf_g = RandomForestRegressor(n_estimators=200, random_state=SEED).fit(X_tr, y_tr["final_grade"])
    pred_g = rf_g.predict(X_te)
    res["rf_grade_r2"] = r2_score(y_te["final_grade"], pred_g)
    res["rf_grade_mae"] = mean_absolute_error(y_te["final_grade"], pred_g)
    # 课程完成率（回归）
    rf_c = RandomForestRegressor(n_estimators=200, random_state=SEED).fit(X_tr, y_tr["completion_rate"])
    pred_c = rf_c.predict(X_te)
    res["rf_completion_r2"] = r2_score(y_te["completion_rate"], pred_c)
    # 是否完成（分类：完成率 ≥ 0.75）
    thr = 0.75
    rf_f = RandomForestClassifier(n_estimators=200, random_state=SEED)
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
    """针对每位学习者的周活跃度序列做 GM(1,1) 趋势预测，
    弥补机器学习模型对小样本群体预测差的短板。"""
    ev = clicks.copy()
    ev["week"] = pd.to_datetime(ev["timestamp"]).dt.isocalendar().week.astype(int)
    weekly = ev.groupby(["student_id", "week"]).size().unstack(fill_value=0)
    rows = []
    for sid, series in weekly.iterrows():
        x0 = series.values.astype(float)
        if len(x0) < min_points:
            rows.append({"student_id": sid, "n_weeks": len(x0), "a": np.nan,
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
        rows.append({"student_id": sid, "n_weeks": len(x0), "a": round(a, 4),
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
# 七、输出层：报告
# ======================================================================
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
        f"- GM(1,1) 周活跃趋势: {extras.get('gm_trends', {})}",
        "",
        "## 学习质量评价（AHP + 模糊综合评判）",
        f"- AHP 权重: {extras.get('ahp_weights', {})}, 一致性 CR = {extras.get('ahp_cr', '-')}",
        f"- 评分均值: {extras.get('quality_mean', '-')}, "
        f"等级分布: {extras.get('quality_levels', {})}",
        "",
        "## 学习者画像",
    ]
    lines += recs.to_markdown(index=False).splitlines()
    lines += ["", "## 干预队列统计", ""]
    qcount = queues["queue"].value_counts()
    lines += [f"- {q}: {n} 人" for q, n in qcount.items()]
    lines += ["", "## 需要教师复核的学生（review/support 队列示例，前 20）", ""]
    sub = queues[queues["queue"].isin(["support", "review"])].head(20)
    lines += sub.to_markdown(index=False).splitlines()
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
    ap.add_argument("--n", type=int, default=400, help="OULAD 合成样本量")
    ap.add_argument("--k", type=int, default=4, help="聚类数（0 = 自动选择）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    global log_handle
    log_handle = open(os.path.join(args.out_dir, "pipeline_log.jsonl"), "a",
                      encoding="utf-8")
    setup_cjk_font()

    log("=" * 60)
    log("  统一学习行为分析管线启动")
    log("=" * 60)

    log("[1/8] 数据层：多源加载与统一")
    uni = build_unified_dataset(args)
    clicks = load_clickstream(args.data_dir, args.n)

    log("[2/8] 特征层：EDA 与清洗完成")
    cols = select_cluster_columns(uni)
    eda_summary(uni, args.out_dir)

    log("[3/8] 分析层：命名主成分 + K-Means 聚类画像")
    pc_scores, loadings, pc_var, pc_names = pca_named_components(uni, cols, args.out_dir)
    uni = uni.merge(pc_scores, on="student_id", how="left")
    uni, km, scaler, cols, metrics = cluster_personas(uni, cols, args.k, args.out_dir)
    uni = label_personas(uni, cols)

    log("[4/8] 挖掘层：社会网络分析 SNA + 滞后序列分析 LSA")
    sna_learners, sna_resources, sna_paths = sna_analysis(clicks, args.out_dir)
    anoms, lsa_tr, lsa_z = lsa_analysis(clicks, args.out_dir)
    uni = uni.merge(anoms, on="student_id", how="left")
    uni["lsa_pattern"] = uni["lsa_pattern"].fillna("无点击流数据")

    log("[5/8] 预测层：RF 学业表现 + 风险预警 + GM(1,1) 趋势预测")
    metrics.update(rf_performance_model(uni, cols, args.out_dir))
    metrics.update(run_predictions(uni, cols, args.out_dir))
    gm = gm11_forecast(clicks, args.out_dir)

    log("[6/8] 评价层：AHP + 模糊综合评判")
    quality, ahp_info = ahp_fuzzy_scores(uni, args.out_dir)
    uni = uni.merge(quality, on="student_id", how="left")

    log("[7/8] 决策层：干预队列与内容推荐")
    queues = rule_queues(uni)
    recs = generate_recommendations(uni)

    log("[8/8] 输出层：导出结果与报告")
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
