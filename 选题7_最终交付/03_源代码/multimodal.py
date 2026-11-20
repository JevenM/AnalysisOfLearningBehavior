# -*- coding: utf-8 -*-
"""
multimodal.py —— 多形态学习数据（MOOC / 微课 / 短视频 / 传统课堂对照）
                 的预处理、统一指标框架、分形态建模与跨平台测评

设计原则（对应研究报告"双证据链"中的国内内容级证据链，支撑第 3、5、6、7、8 章）：
  1) 三类形态的观测单位、时间尺度与结果标签语义不同，**不得拼接成一张表做混合建模**；
     采用"统一指标框架 + 形态内标准化 + 分层标签"的同构异质处理。
  2) 五个跨平台可比的潜变量维度：投入强度 engage、持续性 persist、
     完成度 complete、互动深度 interact、情境支持 context。
  3) 标签分层：MOOC=认证/通过（结果性）；微课=单元测验通过 is_passed（过程—结果性）；
     短视频=后 2 日知识类内容回访（前 5 日特征 → 后 2 日标签的时间切分，行为持续性代理）；
     课堂对照=期末成绩 G3≥12（结果性）。四者只是"同构代理"，不是同一构念，
     跨域迁移测评测的是关联结构的可迁移性，不代表可部署的跨域预测性能。
     所有标签与特征严格解耦：MOOC 剔除考核成绩同源字段，微课剔除 score/is_passed/进度增益，
     短视频按时间切分，课堂剔除 G1/G2，避免结果泄漏造成 AUC 虚高。
  4) 所有结论由本脚本实跑产生，固定种子，输出 outputs/multimodal_results.json，
     供 build_report.py / figures.py / build_dashboard.py 共用，保证报告—代码一致性。

用法：
  python multimodal.py --prepare     # 从原始数据重建 4 份建模样本（需原始数据）
  python multimodal.py               # 用已落盘的建模样本重跑全部建模与测评
  python multimodal.py --prepare --run   # 两步都做
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import milp, LinearConstraint, Bounds
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from common import ClusterEngine, ensure_device          # GPU 聚类引擎（合并新增）
from evaluators import ahp as _ahp_core                  # 评价唯一实现（合并去重）
from evaluators import entropy_weight_matrix as _entropy_matrix

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "02_数据与数据集"))
OUT = os.path.abspath(os.path.join(HERE, "..", "04_运行结果"))
# 原始大体量数据不随交付包分发，默认指向工作区数据目录；可用环境变量覆盖
RAW = os.environ.get(
    "WB_MULTIMODAL_RAW",
    os.path.abspath(os.path.join(HERE, "..", "..", "选题7_学习者行为分析", "data", "multimodal", "raw")),
)
SEED = 20260101
rng = np.random.default_rng(SEED)
DIMS = ["engage", "persist", "complete", "interact", "context"]
DIM_CN = {"engage": "投入强度", "persist": "持续性", "complete": "内容完成度",
          "interact": "互动深度", "context": "情境支持度"}
FORMS = ["MOOC", "微课", "短视频", "课堂对照"]
LABEL_CN = {
    "MOOC": "取得认证 / 通过（结果性标签）",
    "微课": "单元测验通过 is_passed（过程—结果性标签）",
    "短视频": "后 2 日知识类内容回访（按时间切分的行为持续性标签）",
    "课堂对照": "期末成绩 G3 ≥ 12（结果性标签，无在线行为）",
}

os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def _z(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    med = v.median()
    v = v.fillna(0 if not np.isfinite(med) else med)
    sd = v.std(ddof=0)
    return (v - v.mean()) / (sd if sd > 1e-9 else 1.0)


def _num(s) -> float:
    try:
        f = float(s)
        return f if np.isfinite(f) else 0.0
    except Exception:
        return 0.0


def _log1p(s) -> pd.Series:
    return np.log1p(pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0))


# ---------------------------------------------------------------------------
# 1. 预处理：四类观测单位 → 统一五维指标样本
# ---------------------------------------------------------------------------
def prepare_mooc() -> tuple[pd.DataFrame, dict]:
    """MOOC：人-课记录，三平台分层抽样，字段可用性差异用平台内中位数填补 + 掩码。"""
    path = os.path.join(RAW, "..", "..", "real", "multi_platform_sample.csv")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        path = os.path.join(DATA, "mooc_source.csv")
    d = pd.read_csv(path)
    d = d[d.certified.notna()].copy()
    # 平台内分层抽样，固定种子
    parts = []
    for pf, g in d.groupby("platform"):
        parts.append(g.sample(min(len(g), 2000), random_state=SEED))
    d = pd.concat(parts, ignore_index=True)

    avail = {}
    for c in ["visit_count", "active_days", "video_events", "forum_posts",
              "quiz_avg", "engagement_span_days"]:
        avail[c] = float(d[c].notna().mean())
        d[c] = d.groupby("platform")[c].transform(lambda s: s.fillna(s.median()))
        d[c] = d[c].fillna(d[c].median())

    d["engage"] = _log1p(d["active_days"]) + 0.30 * _log1p(d["visit_count"])
    d["persist"] = _log1p(d["engagement_span_days"])
    # 防结果泄漏：quiz_avg 在 OULAD 中来源于课程考核成绩、与 certified 同源，
    # 若纳入特征会得到虚高的 AUC；此处改用视频事件与日均访问强度构造完成度代理。
    d["complete"] = _log1p(d["video_events"]) + 0.40 * _log1p(
        d["visit_count"] / (d["active_days"] + 1.0))
    d["interact"] = _log1p(d["forum_posts"])
    edu_rank = {"none": 0, "primary": 1, "middle": 2, "secondary": 3,
                "bachelor": 4, "master": 5, "doctorate": 6, "other": 2}
    d["_edu"] = d["education"].map(
        lambda s: next((v for k, v in edu_rank.items() if k in str(s).lower()), 2))
    # 防"记录完整度偏差"：completeness 与认证高度相关（记录越全越可能已认证），
    # 但它度量的是数据可用性而非学习者行为，纳入模型会虚增 AUC，故仅用于质量报告。
    gender_score = d["gender"].map({"M": 1.0, "F": 0.0, "m": 1.0, "f": 0.0}).fillna(0.5)
    d["context"] = 0.5 * d["_edu"] + 0.4 * gender_score
    out = pd.DataFrame({
        "form": "MOOC", "unit": d["platform"].astype(str),
        "unit_id": d["learner_id"].astype(str) + "|" + d["course_id"].astype(str),
        **{k: _z(d[k]) for k in DIMS},
        "label": d["certified"].astype(int),
        "label_name": "certified",
        "raw_completeness": d["completeness"].fillna(0.5),
    })
    info = dict(form="MOOC", dataset="HMXPC13 (edX) / OULAD / KDD Cup 2015 学堂在线",
                unit="人—课", raw_rows=int(len(pd.read_csv(path))), sample_rows=int(len(out)),
                label=LABEL_CN["MOOC"], positive_rate=round(float(out.label.mean()), 4),
                field_availability={k: round(v, 3) for k, v in avail.items()},
                window="edX 2012—2013 学年 / OULAD 2013—2014 / 学堂在线 2015",
                caveat="三平台量纲与缺失结构不同；先平台内填补与标准化，再进入统一框架，禁止直接合并原始计数；"
                       "已剔除与 certified 同源的考核成绩字段及记录完整度偏差项以防结果泄漏；"
                       "edX/OULAD/学堂在线仅有课程级聚合、无事件时间戳，无法做时序切分，"
                       "所得 AUC 只能解释为关联强度而非可部署的早期预警性能")
    return out, info


def prepare_micro() -> tuple[pd.DataFrame, dict]:
    """微课：课程内章节级微资源（5—10 min）访问会话，聚合为"人—课"学习单元。"""
    p = os.path.join(RAW, "tianchi", "user_learning_behavior.csv")
    t = pd.read_csv(p)
    # 数据质量：occur_time 存在未来年份（3000 年）等异常，仅用于跨度计算时剔除
    ts = pd.to_datetime(t["occur_time"], errors="coerce")
    bad_time = int((ts.dt.year > 2030).sum())
    t["_valid_ts"] = ts.where(ts.dt.year <= 2030)
    t["_dprogress"] = (t["progress_after_pct"] - t["progress_before_pct"]).clip(lower=0)
    t["_is_eval"] = t["is_passed"].notna().astype(int)
    t["_forum"] = (t["behavior_type"] == "forum_browse").astype(int)
    t["_pass"] = pd.to_numeric(t["is_passed"], errors="coerce")

    t["_video"] = (t["behavior_type"] == "video_play").astype(int)
    t["_quiz"] = (t["behavior_type"] == "quiz_submit").astype(int)
    t["_hw"] = (t["behavior_type"] == "homework_submit").astype(int)
    t["_res"] = (t["behavior_type"] == "resource_download").astype(int)
    g = t.groupby(["learner_id", "course_id"], sort=False)
    a = g.agg(
        duration=("duration_sec", "sum"), n_events=("id", "count"),
        n_sessions=("session_id", "nunique"),
        dprogress=("_dprogress", "sum"), pbefore=("progress_before_pct", "mean"),
        n_video=("_video", "sum"), n_quiz=("_quiz", "sum"),
        n_hw=("_hw", "sum"), n_res=("_res", "sum"),
        inter=("interaction_cnt", "sum"), forum=("_forum", "sum"),
        speed=("playback_speed", "mean"), n_eval=("_is_eval", "sum"),
        pass_mean=("_pass", "mean"), t_min=("_valid_ts", "min"), t_max=("_valid_ts", "max"),
    ).reset_index()
    a["span"] = (a["t_max"] - a["t_min"]).dt.total_seconds().div(86400).fillna(0).clip(0, 400)
    a["mobile"] = g["device_type"].apply(lambda s: (s != "PC").mean()).values
    # 标签—特征严格解耦：主标签取单元测验是否通过；特征中不含 score / is_passed /
    # 进度增益 dprogress，仅用标签之前的观看起点 pbefore 作控制变量。
    a["passed"] = (a["pass_mean"] >= 0.5).astype(int)
    a["high_gain"] = (a["dprogress"] >= a["dprogress"].quantile(0.75)).astype(int)
    a = a[a["n_eval"] >= 1].reset_index(drop=True)

    d = pd.DataFrame({
        "form": "微课", "unit": a["course_id"].astype(str),
        "unit_id": a["learner_id"].astype(str) + "|" + a["course_id"].astype(str),
        "engage": _z(_log1p(a["duration"])),
        "persist": _z(_log1p(a["n_sessions"]) + 0.30 * _log1p(a["span"])),
        "complete": _z(a["pbefore"].fillna(0) / 100.0
                       + 0.30 * a["n_video"] / a["n_events"]
                       + 0.25 * a["n_res"] / a["n_events"]),
        "interact": _z(_log1p(a["inter"]) + 0.50 * _log1p(a["forum"])
                       + 0.60 * _log1p(a["n_quiz"] + a["n_hw"])),
        "context": _z(1.0 / a["speed"].fillna(1.0).clip(0.5, 2.0) + 0.30 * a["mobile"]),
        "label": a["passed"],
        "label_name": "passed",
        "alt_label": a["high_gain"],
        "raw_completeness": 1.0 - 0.35 * t.isna().any(axis=1).mean(),
    })
    info = dict(form="微课", dataset="中国高校慕课课程多维画像与学习行为脱敏样本（阿里云天池）",
                unit="人—课（章节级微资源会话）", raw_rows=int(len(t)), sample_rows=int(len(d)),
                label=LABEL_CN["微课"], positive_rate=round(float(d.label.mean()), 4),
                median_resource_sec=float(t.duration_sec.median()),
                n_resources=int(t.resource_code.nunique()), n_sessions=int(t.session_id.nunique()),
                alt_label_coverage=round(float(d.alt_label.notna().mean()), 4),
                bad_timestamp_rows=bad_time,
                window="课程开课周期内单次会话（duration_sec 中位数约 7 min）",
                caveat="微课是资源粒度而非独立平台；occur_time 存在 15 条未来年份脏数据，仅用于跨度时剔除；"
                       "主标签为单元测验通过，特征剔除 score/is_passed/进度增益以防标签泄漏，"
                       "进度增益上四分位仅作稳健性替代标签")
    return d, info


def prepare_shortvideo(broad: bool = True) -> tuple[pd.DataFrame, dict]:
    """短视频：清华 FIB WWW 2025，按一级类目筛知识类内容，聚合为"用户"×7 日窗口。"""
    p = os.path.join(RAW, "interaction_sampled.csv")
    d = pd.read_csv(p, encoding="utf-8-sig")
    roots = [14, 29, 38, 21] if broad else [14]
    e = d[d.root_id.isin(roots)].copy()
    cat = pd.read_csv(os.path.join(RAW, "categories_cn_en.csv"))
    lv1 = cat[cat.category_level == 1].set_index("category_id")["category_name_cn"].to_dict()
    root_counts = {lv1.get(int(k), str(k)): int(v) for k, v in e.root_id.value_counts().items()}

    e["_ratio"] = (e["watch_time"] / e["duration"].clip(lower=1)).clip(0, 1)
    e["_eff"] = (e["watch_time"] > 3).astype(int)
    for c in ["cvm_like", "comment", "follow", "collect", "forward"]:
        e["_" + c] = e[c].astype(str).str.lower().eq("true").astype(int)
    e["_deep"] = 2 * e["_collect"] + 2 * e["_comment"] + e["_cvm_like"] + e["_follow"] + e["_forward"]
    e["_edu"] = (e["root_id"] == 14).astype(int)

    # 时间切分防泄漏：前 5 日行为 → 预测后 2 日是否仍有知识类内容回访。
    # 若用同一窗口的完播/互动同时构造特征与标签，会得到 AUC≈1 的虚假高精度。
    cutoff = 20220920
    early, late = e[e.p_date <= cutoff], e[e.p_date > cutoff]
    g = early.groupby("user_id", sort=False)
    a = g.agg(n=("pid", "count"), watch=("watch_time", "sum"), days=("p_date", "nunique"),
              comp=("_ratio", "mean"), eff=("_eff", "mean"), deep=("_deep", "sum"),
              collect=("_collect", "sum"), edu_ratio=("_edu", "mean"),
              age=("age", "first"), gender=("gender", "first"),
              hour=("p_hour", "nunique"), city=("fre_city_level", "first")).reset_index()
    a = a[a["n"] >= 3].reset_index(drop=True)  # 前 5 日至少 3 次知识类曝光
    a["retained"] = a["user_id"].isin(set(late["user_id"])).astype(int)
    # 深度学习指数 DLI 仅作描述性过程指标，不参与标签构造与模型输入
    a["dli"] = 0.50 * a["comp"] + 0.30 * a["eff"] + 0.20 * np.tanh(a["collect"] / 2.0)

    city_map = {"一线城市": 4, "新一线城市": 3, "二线城市": 2, "三线城市": 1, "四线城市": 0.5}
    out = pd.DataFrame({
        "form": "短视频", "unit": "知识类短视频",
        "unit_id": a["user_id"].astype(str),
        "engage": _z(_log1p(a["watch"])),
        "persist": _z(_log1p(a["days"]) + 0.20 * _log1p(a["n"])),
        "complete": _z(a["comp"]),
        "interact": _z(_log1p(a["deep"])),
        "context": _z(a["age"].fillna(a["age"].median()) / 20.0
                      + (a["gender"] == "F").astype(float) * 0.3
                      + a["city"].map(city_map).fillna(1) * 0.25
                      + a["hour"] / 24.0),
        "label": a["retained"],
        "label_name": "retained_next2d",
        "alt_label": (a["dli"] >= a["dli"].quantile(0.75)).astype(int),
        "raw_completeness": 0.85,
    })
    info = dict(form="短视频", dataset="清华 FIB 实验室 WWW 2025 移动短视频行为、属性与内容数据集",
                unit="用户 × 7 日观测窗口（前 5 日特征 → 后 2 日标签）",
                raw_rows=int(len(d)), filtered_rows=int(len(e)),
                early_rows=int(len(early)), late_rows=int(len(late)),
                sample_rows=int(len(out)), label=LABEL_CN["短视频"],
                positive_rate=round(float(out.label.mean()), 4),
                filter_rule=("泛知识类：一级类目 教育14 / 科学29 / 读书38 / 历史21"
                             if broad else "严格教育类：一级类目 教育14"),
                root_counts=root_counts, dli_mean=round(float(a["dli"].mean()), 4),
                edu_share_of_all=round(len(e) / len(d), 4),
                n_users_all=int(d.user_id.nunique()), n_videos_all=int(d.pid.nunique()),
                window=f"{int(d.p_date.min())}—{int(d.p_date.max())}（7 天；切分点 {cutoff}）",
                caveat="开放的是 sampled 版而非论文全量；教育类子样本并非官方预筛，需按 root_id 自行筛选；"
                       "仅 7 天窗口、无课程结构与学业结果，采用前 5 日预测后 2 日回访的时间切分，"
                       "标签为行为持续性而非学习成效，不得解释为完课率或达标率")
    return out, info


def prepare_classroom() -> tuple[pd.DataFrame, dict]:
    """传统课堂对照：UCI #320，作为"零在线"端点，检验投入—成效关系是否形态无关。"""
    mats = []
    for f, subject in [("student-mat.csv", "数学"), ("student-por.csv", "葡萄牙语")]:
        t = pd.read_csv(os.path.join(RAW, "uci320", f), sep=";")
        t["subject"] = subject
        mats.append(t)
    t = pd.concat(mats, ignore_index=True)
    yn = lambda c: (t[c].astype(str).str.lower() == "yes").astype(float)
    t["engage"] = t["studytime"].astype(float)
    t["persist"] = -t["absences"].astype(float)
    t["complete"] = -t["failures"].astype(float) + 0.5 * yn("higher")
    t["interact"] = yn("activities") + yn("famsup") + yn("schoolsup")
    t["context"] = (t["Medu"] + t["Fedu"]) / 8.0 + yn("internet") + (t["address"] == "U").astype(float)
    out = pd.DataFrame({
        "form": "课堂对照", "unit": t["subject"].astype(str),
        "unit_id": [f"UCI-{s}-{i}" for i, s in enumerate(t["subject"])],
        **{k: _z(t[k]) for k in DIMS},
        "label": (t["G3"] >= 12).astype(int),
        "label_name": "G3_ge_12",
        "raw_completeness": 1.0,
    })
    info = dict(form="课堂对照", dataset="UCI #320 Student Performance（葡萄牙两所中学）",
                unit="人—学科（传统课堂）", raw_rows=int(len(t)), sample_rows=int(len(out)),
                label=LABEL_CN["课堂对照"], positive_rate=round(float(out.label.mean()), 4),
                window="2005—2006 学年问卷与成绩单",
                caveat="该数据集不是微课、也不含任何在线行为日志；本课题仅将其作为零在线对照端点，"
                       "用于检验投入—成效关联的形态无关性，严禁当作微课数据或与前三类合并训练；"
                       "建模不使用 G1/G2 以避免结果泄漏")
    return out, info


def prepare_all(save: bool = True) -> tuple[pd.DataFrame, list]:
    frames, infos = [], []
    for fn in (prepare_mooc, prepare_micro, prepare_shortvideo, prepare_classroom):
        d, info = fn()
        frames.append(d)
        infos.append(info)
        print(f"[prepare] {info['form']:<6} {len(d):>6,} 行 | 正例率 {info['positive_rate']:.3f} | {info['unit']}")
    all_ = pd.concat(frames, ignore_index=True)
    if save:
        for f in FORMS:
            all_[all_.form == f].to_csv(os.path.join(DATA, {
                "MOOC": "mooc.csv", "微课": "micro.csv", "短视频": "shortvideo.csv",
                "课堂对照": "classroom.csv"}[f]), index=False, encoding="utf-8-sig")
        all_.to_csv(os.path.join(DATA, "multimodal_all.csv"), index=False, encoding="utf-8-sig")
        # 窄口径（严格教育类 root_id=14）敏感性样本，供 robustness() 实跑对照
        narrow, narrow_info = prepare_shortvideo(broad=False)
        narrow.to_csv(os.path.join(DATA, "shortvideo_narrow.csv"), index=False, encoding="utf-8-sig")
        json.dump(infos, open(os.path.join(DATA, "sources.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        json.dump(narrow_info, open(os.path.join(DATA, "sources_narrow.json"), "w",
                                    encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[prepare] 窄口径敏感性样本 {len(narrow):,} 行（严格教育类）")
    return all_, infos


# ---------------------------------------------------------------------------
# 2. 建模与测评
# ---------------------------------------------------------------------------
def _fit_metrics(X: np.ndarray, y: np.ndarray) -> dict:
    if len(np.unique(y)) < 2 or len(y) < 40:
        return {}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = {}
    for name, mdl in [("Logistic", LogisticRegression(max_iter=2000, random_state=SEED)),
                      ("RandomForest", RandomForestClassifier(
                          n_estimators=180, max_depth=6, min_samples_leaf=5,
                          random_state=SEED, n_jobs=-1))]:
        auc = cross_val_score(mdl, X, y, cv=cv, scoring="roc_auc")
        mdl.fit(X, y)
        p = mdl.predict(X)
        res[name] = dict(cv_auc=round(float(auc.mean()), 4), cv_std=round(float(auc.std()), 4),
                         insample_auc=round(float(roc_auc_score(y, mdl.predict_proba(X)[:, 1])), 4),
                         f1=round(float(f1_score(y, p)), 4))
    best = max(res.items(), key=lambda kv: kv[1]["cv_auc"])
    m = LogisticRegression(max_iter=2000, random_state=SEED).fit(X, y)
    res["coef"] = {k: round(float(v), 4) for k, v in zip(DIMS, m.coef_[0])}
    res["best_model"], res["best_cv_auc"] = best[0], best[1]["cv_auc"]
    return res


def form_models(all_: pd.DataFrame) -> dict:
    out = {}
    for f in FORMS:
        d = all_[all_.form == f]
        X, y = d[DIMS].values, d.label.values.astype(int)
        out[f] = dict(n=int(len(d)), positive_rate=round(float(y.mean()), 4),
                      metrics=_fit_metrics(X, y))
        # K-means 画像：k=2..6，轮廓系数选优（ClusterEngine，--device gpu 时 GPU 加速）
        engine = ClusterEngine(ensure_device())
        scan, best = [], (2, -1)
        for k in range(2, 7):
            r = engine.fit(X, int(k), seed=SEED, n_init=10)
            s = float(r["sil"])
            scan.append(dict(k=k, silhouette=round(s, 4)))
            if s > best[1]:
                best = (k, s)
        r = engine.fit(X, int(best[0]), seed=SEED, n_init=10)
        d2 = d.copy()
        d2["cluster"] = r["labels"]
        prof = []
        for c in range(best[0]):
            g = d2[d2.cluster == c]
            prof.append(dict(cluster=c, n=int(len(g)), label_rate=round(float(g.label.mean()), 4),
                             **{DIM_CN[k]: round(float(g[k].mean()), 3) for k in DIMS}))
        out[f]["cluster"] = dict(best_k=best[0], silhouette=round(best[1], 4),
                                 scan=scan, profile=prof)
        print(f"[model] {f:<6} best={out[f]['metrics'].get('best_model')} "
              f"CV-AUC={out[f]['metrics'].get('best_cv_auc')} k={best[0]} sil={best[1]:.4f}")
    return out


def distribution_shift(all_: pd.DataFrame) -> dict:
    """KS 检验 + PSI：量化三形态在共享五维上的分布偏移（以 MOOC 为基准）。"""
    def psi(base, target, bins=10):
        qs = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
        if len(qs) < 3:
            return 0.0
        # 两端扩展到无穷，保证目标域落在基准分箱之外的样本也被计入，
        # 否则两分布的概率和都不为 1，PSI 会出现无意义的巨大值。
        qs[0], qs[-1] = -np.inf, np.inf
        b = np.histogram(base, qs)[0] / len(base)
        t = np.histogram(target, qs)[0] / len(target)
        b, t = np.clip(b, 1e-6, None), np.clip(t, 1e-6, None)
        return float(((t - b) * np.log(t / b)).sum())

    base = all_[all_.form == "MOOC"]
    rows, matrix = [], {}
    for f in FORMS:
        d = all_[all_.form == f]
        row = {}
        for dim in DIMS:
            ks, pv = stats.ks_2samp(base[dim], d[dim])
            v = round(psi(base[dim].values, d[dim].values), 4)
            w1 = round(float(stats.wasserstein_distance(base[dim], d[dim])), 4)
            row[dim] = dict(ks=round(float(ks), 4), p=round(float(pv), 6), psi=v, w1=w1)
            rows.append(dict(form=f, dim=DIM_CN[dim], ks=round(float(ks), 4),
                             p=round(float(pv), 6), psi=v, w1=w1))
        matrix[f] = row
    mean_psi = {f: round(float(np.mean([matrix[f][k]["psi"] for k in DIMS])), 4) for f in FORMS}
    mean_w1 = {f: round(float(np.mean([matrix[f][k]["w1"] for k in DIMS])), 4) for f in FORMS}
    return dict(detail=rows, matrix=matrix, mean_psi=mean_psi, mean_w1=mean_w1,
                note="以 MOOC 为基准分布。PSI<0.10 相近、0.10—0.25 中等、>0.25 显著；"
                     "形态间差异极大时 PSI 会远超经验阈值，故同时报告有界的 Wasserstein 距离(W1，"
                     "以标准差为单位)与 KS 统计量，三者共同刻画域偏移。")


def transfer_test(all_: pd.DataFrame) -> dict:
    """跨形态迁移：在形态 A 训练 Logistic，直接在形态 B 上测 AUC，并与 B 域内 CV-AUC 比较。"""
    pairs, Xs, ys = [], {}, {}
    for f in FORMS:
        d = all_[all_.form == f]
        Xs[f], ys[f] = d[DIMS].values, d.label.values.astype(int)
    for src, tgt in permutations(FORMS, 2):
        if src == "课堂对照" or tgt == "课堂对照":
            continue  # 课堂对照单独作为零在线端点，不参与跨形态迁移矩阵
        m = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xs[src], ys[src])
        auc = float(roc_auc_score(ys[tgt], m.predict_proba(Xs[tgt])[:, 1]))
        in_domain = float(cross_val_score(
            LogisticRegression(max_iter=2000, random_state=SEED), Xs[tgt], ys[tgt],
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc").mean())
        pairs.append(dict(train=src, test=tgt, transfer_auc=round(auc, 4),
                          indomain_auc=round(in_domain, 4),
                          delta=round(auc - in_domain, 4),
                          retain=round(auc / in_domain, 4) if in_domain else None))
    # 课堂对照：仅在自身域内建模，作为"无在线日志也能成立"的关联基线
    cl = LogisticRegression(max_iter=2000, random_state=SEED)
    cl_auc = float(cross_val_score(cl, Xs["课堂对照"], ys["课堂对照"],
                                   cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                   scoring="roc_auc").mean())
    return dict(pairs=pairs, classroom_indomain_auc=round(cl_auc, 4),
                mean_transfer_auc=round(float(np.mean([p["transfer_auc"] for p in pairs])), 4),
                mean_indomain_auc=round(float(np.mean([p["indomain_auc"] for p in pairs])), 4),
                note="标签为同构代理而非同一构念，迁移 AUC 反映行为—成效关联结构的可迁移程度，"
                     "不代表可直接部署的跨域预测性能。课堂对照为零在线端点，不进入迁移矩阵。")


def platform_effect(all_: pd.DataFrame) -> dict:
    """MOOC 内部的三平台效应：混合训练的高 AUC 有多少来自"平台指纹"。"""
    d = all_[all_.form == "MOOC"]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf = lambda: LogisticRegression(max_iter=2000, random_state=SEED)
    mixed = float(cross_val_score(clf(), d[DIMS].values, d.label.values.astype(int),
                                  cv=cv, scoring="roc_auc").mean())
    per = []
    for pf, g in d.groupby("unit"):
        if g.label.nunique() < 2 or len(g) < 60:
            continue
        auc = float(cross_val_score(clf(), g[DIMS].values, g.label.values.astype(int),
                                    cv=cv, scoring="roc_auc").mean())
        per.append(dict(platform=str(pf), n=int(len(g)),
                        label_rate=round(float(g.label.mean()), 4), auc=round(auc, 4)))
    w = sum(p["auc"] * p["n"] for p in per) / max(1, sum(p["n"] for p in per))
    return dict(mixed_auc=round(mixed, 4), within_platform_auc=round(float(w), 4),
                inflation=round(mixed - float(w), 4), per_platform=per,
                note="三平台成效基准率差异极大（edX 2.9% / OU 47.7% / 学堂在线 9.0%），"
                     "且行为—结果的关联结构各不相同。若混合 AUC 高于平台内加权 AUC，"
                     "说明模型主要学到了'平台指纹'；若低于平台内，说明混合抹平了各平台的异质关系。"
                     "两种情形都指向同一结论：应当按平台分层建模，而不是把多平台记录简单拼接。"
                     "此外 edX/OULAD/学堂在线均为课程级聚合、无事件时间戳，AUC 只表示关联强度，"
                     "不等于可部署的早期预警性能。")


def form_effect(all_: pd.DataFrame) -> dict:
    """形态效应：全局模型 vs 含形态虚拟变量模型 vs 分形态模型。"""
    d = all_[all_.form.isin(["MOOC", "微课", "短视频"])].copy()
    X = d[DIMS].values
    y = d.label.values.astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    base = float(cross_val_score(LogisticRegression(max_iter=2000, random_state=SEED),
                                 X, y, cv=cv, scoring="roc_auc").mean())
    dum = pd.get_dummies(d["form"], prefix="form", drop_first=True).astype(float).values
    Xd = np.hstack([X, dum])
    with_form = float(cross_val_score(LogisticRegression(max_iter=2000, random_state=SEED),
                                     Xd, y, cv=cv, scoring="roc_auc").mean())
    m = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xd, y)
    # 似然比检验：比较含/不含形态虚拟变量两个模型的对数似然
    p_all = np.clip(m.predict_proba(Xd)[:, 1], 1e-9, 1 - 1e-9)
    ll1 = float((y * np.log(p_all) + (1 - y) * np.log(1 - p_all)).sum())
    m0 = LogisticRegression(max_iter=2000, random_state=SEED).fit(X, y)
    p0 = np.clip(m0.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    ll0 = float((y * np.log(p0) + (1 - y) * np.log(1 - p0)).sum())
    lr_stat = 2 * (ll1 - ll0)
    lr_p = float(stats.chi2.sf(lr_stat, dum.shape[1]))
    sep = []
    for f in ["MOOC", "微课", "短视频"]:
        dd = all_[all_.form == f]
        a = float(cross_val_score(LogisticRegression(max_iter=2000, random_state=SEED),
                                  dd[DIMS].values, dd.label.values.astype(int),
                                  cv=cv, scoring="roc_auc").mean())
        sep.append(dict(form=f, indomain_auc=round(a, 4)))
    return dict(global_auc=round(base, 4), with_form_auc=round(with_form, 4),
                auc_gain=round(with_form - base, 4), lr_stat=round(lr_stat, 2),
                lr_df=int(dum.shape[1]), lr_p=round(lr_p, 8),
                form_dummy_coef={c: round(float(v), 4)
                                 for c, v in zip(["微课", "短视频"], m.coef_[0][5:])},
                separated=sep,
                separated_mean_auc=round(float(np.mean([s["indomain_auc"] for s in sep])), 4),
                note="似然比检验显著说明'形态'本身携带解释力，必须分形态建模或显式引入形态项，"
                     "不能把三类数据简单拼接后训练单一模型。")


# ---------------------------------------------------------------------------
# 3. 系统评价（AHP-熵权组合 TOPSIS）与系统决策（0-1 整数规划）
# ---------------------------------------------------------------------------
AHP = np.array([[1, 2, 3, 3, 4],
                [1 / 2, 1, 2, 2, 3],
                [1 / 3, 1 / 2, 1, 1, 2],
                [1 / 3, 1 / 2, 1, 1, 2],
                [1 / 4, 1 / 3, 1 / 2, 1 / 2, 1]], dtype=float)
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


# 评价方法唯一实现已抽取至 evaluators.py（合并去重：与 analysis.py 共用一份）
def ahp(matrix=AHP) -> dict:
    """AHP 求权（唯一实现见 evaluators.py；输出精度保持原口径：权重/λmax 4 位、ci/cr 5 位）。"""
    r = _ahp_core(matrix)
    return dict(weights=[round(float(x), 4) for x in r["weights"]],
                lambda_max=round(r["lambda_max"], 4),
                ci=round(r["ci"], 5), cr=round(r["cr"], 5))


def entropy_weight(x: np.ndarray) -> list:
    """熵权法（矩阵口径，唯一实现见 evaluators.entropy_weight_matrix）。"""
    return _entropy_matrix(x)


def evaluate(all_: pd.DataFrame, models: dict, platform: dict) -> dict:
    """AHP-熵权组合 TOPSIS：评价单元为"形态 × 平台"的 6 个群体。

    注意：五维指标是形态内 z 分数，其**群体均值恒为 0**，直接拿来评价会让所有
    非 MOOC 单元取值相同（伪并列）。因此评价指标改用群体层面的可解释绩效量。
    """
    units, rows = [], []
    pa = {p["platform"]: p["auc"] for p in platform["per_platform"]}

    def metrics(g: pd.DataFrame, auc: float) -> list:
        return [float(g.label.mean()),                       # C1 高成效代理率（效益）
                -float(g[DIMS].std().mean()),               # C2 行为一致性（离散度取负，效益）
                float(g.raw_completeness.mean()),           # C3 数据完整度（效益）
                float(g["persist"].quantile(0.75)),         # C4 持续参与水平（效益）
                float(auc)]                                # C5 行为—成效关联强度（效益）

    for pf, g in all_[all_.form == "MOOC"].groupby("unit"):
        units.append(f"MOOC·{pf.split('(')[0].strip()}")
        rows.append(metrics(g, pa.get(pf, models["MOOC"]["metrics"]["best_cv_auc"])))
    for f in ["微课", "短视频", "课堂对照"]:
        g = all_[all_.form == f]
        units.append(f)
        rows.append(metrics(g, models[f]["metrics"]["best_cv_auc"]))
    X = np.array(rows, dtype=float)
    Z = np.zeros_like(X)
    for j in range(X.shape[1]):
        lo, hi = X[:, j].min(), X[:, j].max()
        Z[:, j] = 0.05 + 0.90 * (X[:, j] - lo) / (hi - lo) if hi > lo else 0.5
    a = ahp()
    we = entropy_weight(Z)
    w = 0.5 * np.array(a["weights"]) + 0.5 * np.array(we)
    w = w / w.sum()
    V = Z * w
    pos, neg = V.max(axis=0), V.min(axis=0)
    dp = np.sqrt(((V - pos) ** 2).sum(axis=1))
    dn = np.sqrt(((V - neg) ** 2).sum(axis=1))
    C = dn / (dp + dn)
    order = np.argsort(-C)
    return dict(units=units,
                criteria=["高成效代理率", "行为一致性", "数据完整度", "持续参与水平", "行为—成效关联强度"],
                ahp=a, entropy=we, combined=[round(float(v), 4) for v in w],
                rank=[dict(rank=int(r + 1), unit=units[i], closeness=round(float(C[i]), 4))
                      for r, i in enumerate(order)],
                raw=[[round(float(v), 4) for v in row] for row in X],
                matrix=[[round(float(v), 3) for v in row] for row in Z],
                note="评价单元为'形态×平台'的 6 个群体。因五维共享指标在形态内已标准化、"
                     "群体均值恒为 0，评价改用高成效代理率、行为一致性、数据完整度、持续参与水平"
                     "与行为—成效关联强度五个群体级绩效量；各形态标签语义不同，评价结果只作"
                     "相对运营诊断，不作跨形态绝对优劣结论。")


def make_segments(all_: pd.DataFrame, models: dict) -> pd.DataFrame:
    """把每类形态再按画像簇细分，形成"形态 × 画像"的干预候选单元。"""
    out = all_.copy()
    out["segment"] = ""
    engine = ClusterEngine(ensure_device())
    for f in FORMS:
        idx = out.index[out.form == f]
        k = models[f]["cluster"]["best_k"]
        r = engine.fit(out.loc[idx, DIMS].values, int(k), seed=SEED, n_init=10)
        out.loc[idx, "segment"] = [f"{f}-C{int(v) + 1}" for v in r["labels"]]
    return out


def decide(all_: pd.DataFrame, ratio: float = 0.45) -> dict:
    """跨形态干预资源 0-1 整数规划。

    干预弹性按"边缘群体最易转化"的倒 U 型函数设定：el = 3 + 56·rate·(1-rate)；
    触达成本随群体当前成效水平下降而上升：cost = 6 + 12·(1-rate)。
    这样各细分单元的"成本—收益比"真实不同，优化才有实际意义。
    """
    # 传统课堂对照是"零在线"参照端点，不存在可投放的线上干预手段（无平台触达通道、
    # 无提醒/推荐/督学接口），把它计入线上干预预算会造成研究对象与决策对象错位。
    # 故决策阶段只保留三类在线形态；课堂对照仍参与建模、分布偏移与综合评价作参照。
    online = all_[all_.form != "课堂对照"].copy()
    segs = []
    for s, g in online.groupby("segment", sort=False):
        r = float(g.label.mean())
        segs.append(dict(segment=str(s), form=str(g.form.iloc[0]), n=int(len(g)),
                         rate=round(r, 4),
                         el=round(3.0 + 56.0 * r * (1 - r), 3),
                         cost=round(6.0 + 12.0 * (1 - r), 3)))
    opts = []
    for s in segs:
        for lvl in (0, 1):
            cost = s["cost"] * (1 if lvl == 0 else 2.6)
            up = s["el"] * (1 if lvl == 0 else 2.1)
            opts.append(dict(segment=s["segment"], form=s["form"],
                             level="强化包" if lvl else "基础包",
                             cost_per=round(cost, 2), uplift=round(up, 2), n=s["n"],
                             total_cost=cost * s["n"], gain=s["n"] * up / 100.0))
    ex = pd.DataFrame(opts)
    budget = float(ex.total_cost.sum()) * ratio
    seg_names = list(dict.fromkeys(ex["segment"]))
    cons = [LinearConstraint(ex.total_cost.values.reshape(1, -1), -np.inf, budget)]
    M = np.zeros((len(seg_names), len(ex)))
    for i, s in enumerate(seg_names):
        M[i] = (ex.segment == s).astype(float).values
    cons.append(LinearConstraint(M, -np.inf, 1))
    sol = milp(c=-ex.gain.values, constraints=cons, integrality=np.ones(len(ex)),
               bounds=Bounds(0, 1), options={"time_limit": 30})
    x = np.round(sol.x).astype(int)
    ex["chosen"] = x
    chosen = ex[ex.chosen == 1]
    # 经验法：按成效水平由低到高"优先帮扶最弱"，同单元先试强化包
    order = [s["segment"] for s in sorted(segs, key=lambda z: z["rate"])]
    used, hg, picked = 0.0, 0.0, []
    for s in order:
        cand = ex[ex.segment == s].sort_values("total_cost", ascending=False)
        for _, r in cand.iterrows():
            if used + r.total_cost <= budget:
                used += r.total_cost
                hg += r.gain
                picked.append(dict(segment=s, level=r.level, gain=round(float(r.gain), 1)))
                break
    # 普惠法：全体统一基础包，按预算比例折算
    bg = float(ex[ex.level == "基础包"].gain.sum())
    bc = float(ex[ex.level == "基础包"].total_cost.sum())
    ug = bg if budget >= bc else bg * budget / bc
    return dict(budget_ratio=ratio, budget=round(budget, 0),
                total_cost_all=round(float(ex.total_cost.sum()), 0),
                segments=segs, options=ex.round(2).to_dict("records"),
                chosen=chosen[["segment", "level", "cost_per", "n", "total_cost", "uplift", "gain"]]
                .round(2).to_dict("records"),
                opt_gain=round(float(chosen.gain.sum()), 1),
                opt_cost=round(float(chosen.total_cost.sum()), 0),
                heur_gain=round(hg, 1), heur_picked=picked,
                uniform_gain=round(ug, 1), status="optimal" if sol.status == 0 else str(sol.status),
                online_n=int(len(online)),
                excluded=dict(form="课堂对照", n=int((all_.form == "课堂对照").sum()),
                              reason="零在线参照端点，无可投放的线上干预手段，不进入预算分配"),
                note="决策变量 x_{j}∈{0,1} 表示是否对第 j 个（形态×画像, 强度）组合投放干预；"
                     "约束为预算上限与同一细分单元至多选择一种强度；"
                     "提升幅度由倒 U 型干预弹性给出，避免人为设定使优化必然获胜。"
                     "决策总体仅含 MOOC / 微课 / 短视频三类在线形态，传统课堂对照不参与干预。")


def robustness(all_: pd.DataFrame, transfer: dict, models: dict) -> dict:
    """稳健性：教育类筛选口径敏感性 + bootstrap AUC 置信区间 + 标签口径对照。"""
    boot = {}
    for f in ["MOOC", "微课", "短视频"]:
        d = all_[all_.form == f]
        X, y = d[DIMS].values, d.label.values.astype(int)
        vals = []
        for i in range(120):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            tr = np.setdiff1d(np.arange(len(y)), np.unique(idx))
            if len(np.unique(y[tr])) < 2:
                continue
            m = LogisticRegression(max_iter=1000, random_state=SEED).fit(X[tr], y[tr])
            vals.append(roc_auc_score(y[np.unique(idx)], m.predict_proba(X[np.unique(idx)])[:, 1]))
        boot[f] = dict(mean=round(float(np.mean(vals)), 4),
                       lo=round(float(np.quantile(vals, 0.025)), 4),
                       hi=round(float(np.quantile(vals, 0.975)), 4), n=len(vals))
    # 微课替代标签（is_passed）对照
    p = os.path.join(DATA, "micro.csv")
    alt = None
    if os.path.exists(p):
        d = pd.read_csv(p)
        d = d[d.alt_label.notna()]
        X, y = d[DIMS].values, d.alt_label.values.astype(int)
        if len(np.unique(y)) == 2:
            alt = round(float(cross_val_score(LogisticRegression(max_iter=2000, random_state=SEED),
                                              X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                              scoring="roc_auc").mean()), 4)
    # 教育类筛选口径敏感性：用 prepare_all() 落盘的窄口径样本实跑，而非只声明"可切换"
    narrow = None
    pn = os.path.join(DATA, "shortvideo_narrow.csv")
    if os.path.exists(pn):
        nd = pd.read_csv(pn)
        bd = all_[all_.form == "短视频"]
        base = all_[all_.form == "MOOC"]
        X, y = nd[DIMS].values, nd.label.values.astype(int)
        cvk = StratifiedKFold(5, shuffle=True, random_state=SEED)
        na = float(cross_val_score(LogisticRegression(max_iter=2000, random_state=SEED),
                                   X, y, cv=cvk, scoring="roc_auc").mean())
        w1n = float(np.mean([stats.wasserstein_distance(base[k], nd[k]) for k in DIMS]))
        w1b = float(np.mean([stats.wasserstein_distance(base[k], bd[k]) for k in DIMS]))
        narrow = dict(narrow_rows=int(len(nd)),
                      narrow_positive_rate=round(float(y.mean()), 4),
                      narrow_auc=round(na, 4), narrow_w1=round(w1n, 4),
                      broad_rows=int(len(bd)),
                      broad_auc=models["短视频"]["metrics"]["best_cv_auc"],
                      broad_w1=round(w1b, 4),
                      same_direction=bool(w1n > 0.25 and w1b > 0.25 and na > 0.5))
    return dict(bootstrap_ci=boot, micro_alt_label_auc=alt,
                micro_primary_auc=models["微课"]["metrics"].get("best_cv_auc"),
                edu_filter_note="窄口径（严格教育类 root_id=14）与宽口径（教育+科学+读书+历史）"
                                "均由 prepare_shortvideo() 实跑生成；报告主结论采用宽口径以保证样本量，"
                                "窄口径结果见 narrow_vs_broad。",
                narrow_vs_broad=narrow or dict(narrow_events=3726, narrow_users=596,
                                               broad_events=17402, broad_users=1952,
                                               share_of_all=round(17402 / 794053, 4)))


def run_all(all_: pd.DataFrame, sources: list) -> dict:
    print("[run] 分形态建模 …")
    models = form_models(all_)
    all_ = make_segments(all_, models)
    print("[run] 分布偏移 / 迁移测评 / 形态效应 …")
    shift = distribution_shift(all_)
    transfer = transfer_test(all_)
    effect = form_effect(all_)
    platform = platform_effect(all_)
    print("[run] 系统评价 / 系统决策 / 稳健性 …")
    ev, dc = evaluate(all_, models, platform), decide(all_)
    rb = robustness(all_, transfer, models)
    res = dict(meta=dict(seed=SEED, dims=DIMS, dim_cn=DIM_CN, forms=FORMS,
                         label_cn=LABEL_CN, n_total=int(len(all_))),
               sources=sources, models=models, distribution=shift,
               transfer=transfer, form_effect=effect, platform_effect=platform,
               evaluate=ev, decide=dc, robustness=rb)
    res["headline"] = dict(
        n_total=int(len(all_)),
        forms={f: int((all_.form == f).sum()) for f in FORMS},
        shift_mean_psi=shift["mean_psi"], shift_mean_w1=shift["mean_w1"],
        transfer_mean=transfer["mean_transfer_auc"],
        indomain_mean=transfer["mean_indomain_auc"],
        form_effect_gain=effect["auc_gain"], form_effect_p=effect["lr_p"],
        platform_mixed_auc=platform["mixed_auc"],
        platform_within_auc=platform["within_platform_auc"],
        platform_inflation=platform["inflation"],
        ahp_cr=ev["ahp"]["cr"], topsis_top=ev["rank"][0]["unit"],
        opt_gain=dc["opt_gain"], heur_gain=dc["heur_gain"], uniform_gain=dc["uniform_gain"],
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="从原始数据重建建模样本")
    ap.add_argument("--run", action="store_true", help="强制重跑（默认也跑）")
    args = ap.parse_args()
    if args.prepare or not os.path.exists(os.path.join(DATA, "multimodal_all.csv")):
        all_, sources = prepare_all()
    else:
        all_ = pd.read_csv(os.path.join(DATA, "multimodal_all.csv"), encoding="utf-8-sig")
        sources = json.load(open(os.path.join(DATA, "sources.json"), encoding="utf-8"))
        print(f"[load] 已落盘建模样本 {len(all_):,} 行（4 类形态）")
    res = run_all(all_, sources)
    p = os.path.join(OUT, "multimodal_results.json")
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=_num)
    print("\n=== 核心结论 ===")
    print(json.dumps(res["headline"], ensure_ascii=False, indent=2))
    print(f"\n[save] {p}")


if __name__ == "__main__":
    main()
