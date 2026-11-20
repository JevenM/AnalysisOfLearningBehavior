# -*- coding: utf-8 -*-
"""
profile_modeling.py —— 六维学习者画像建模（地区 / 专业背景 / 学习习惯 / 兴趣爱好 /
                     短视频用户属性 / 天池微课标签），对应报告第 5.2 节「系统分析」
《系统工程原理》实践项目 选题7

回应研究问题：线上学习数据（登录时间、选课数量与类别、访问、观看等显性/隐性行为）
是否蕴含使用者的兴趣爱好、学习习惯、专业背景等信息，能否据此刻画群体画像并做
地区/背景差异细分。

六个维度全部基于真实数据，无仿真（课程链 4 维 + 国内内容级 2 维，分层呈现不混算）：
  1) 地区差异      : OULAD real_learners.csv 的 region（13 类英国地区）
  2) 专业背景      : OULAD education（5 类学历）+ discipline（7 个课程模块/学科方向）
                     以及 course_num（选课门数，真实派生）
  3) 学习习惯      : OULAD 行为节律（首活滞后、活跃密度、单次时长、资源覆盖）
                     + 短视频 p_hour 观看时段偏好（6,654 用户真实时段）
  4) 兴趣爱好      : 短视频 root_id 一级类目（38 类）用户偏好画像
                     + 教育类内容偏好子群（root_id=14）行为对照
  5) 短视频用户属性: 城市等级 / 城乡 / 性别 / 年龄 / 观看时段（国内数据，6,654 用户）
  6) 天池微课标签  : 倍速 / 设备 / OS / 网络 / 预习 / 学习状态（国内数据，脱敏样本）

口径与伦理边界（写入结果 meta，报告须如实引用）：
  - 地区为英国开放大学学习者地域（locale），非国籍，不外推到其它教育体系。
  - "专业背景"以入学学历与所选课程模块作代理，数据集无"专业"自陈字段。
  - "兴趣"由内容观看占比推断，属行为偏好代理，非心理测量；短视频平台内容偏好
    不等同学科兴趣，仅作方法示范与人群描述，不用于个体评价。
  - 学习习惯中"登录时间"在 OULAD 仅有首活滞后与日粒度，短视频有小时级 p_hour，
    二者不可直接合并，分层呈现。
  - 群体差异均为观察性关联，非因果；不做个体 ranking 或惩戒。

输出：
  04_运行结果/profile_results.json
  05_图表与看板/figures/fig17_region.png ~ fig20_interest.png
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATA_DIR, OUT_DIR                    # 路径常量（合并去重）
from common import apply_fig_style

apply_fig_style(dpi=180)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = DATA_DIR
RUN = OUT_DIR
FIG = os.path.join(ROOT, "05_图表与看板", "figures")
# 细粒度画像输入表（由 build_profile_inputs.py 从真实原始数据派生，随交付包分发，离线可复现）
SV_PROFILE = os.path.join(DATA, "shortvideo_profile.csv")     # 短视频用户级 6,654 人
SV_CATWATCH = os.path.join(DATA, "shortvideo_cat_watch.csv")  # 短视频类目观看汇总 37 类
TC_PROFILE = os.path.join(DATA, "tianchi_profile.csv")        # 天池事件级 5,000 条
# 兼容：若派生表缺失，回退到工作区原始数据（不随交付分发，仅本地复现用）
RAW_SV = os.path.abspath(os.path.join(ROOT, "..", "选题7_学习者行为分析",
                                      "data", "multimodal", "raw", "interaction_sampled.csv"))
RAW_CAT = os.path.abspath(os.path.join(ROOT, "..", "选题7_学习者行为分析",
                                       "data", "multimodal", "raw", "categories_cn_en.csv"))
SEED = 20260101

C = {"blue": "#2E75B6", "teal": "#2A9D8F", "amber": "#E9A23B", "red": "#C0392B",
     "slate": "#64748B", "light": "#D9E2F0", "grid": "#E2E8F0"}


def _r(x, n=4):
    try:
        v = float(x)
        return None if not np.isfinite(v) else round(v, n)
    except (TypeError, ValueError):
        return None


def _p(x):
    """p 值保留有效数字，避免极小值被 round 成 0（统计上应写作 <1e-6）。"""
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        if v < 1e-6:
            return "<1e-6"
        return float(f"{v:.3g}")
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------
# 工具：分组描述 + 差异检验
# ----------------------------------------------------------------------------
def group_table(df, by, label="certificate"):
    g = df.groupby(by, observed=True)
    rows = []
    for key, sub in g:
        rows.append({
            "group": str(key),
            "n": int(len(sub)),
            "share": _r(len(sub) / len(df)),
            "pass_rate": _r(sub[label].mean()),
            "visits": _r(sub["visit_count"].mean(), 2),
            "active_days": _r(sub["active_days"].mean(), 2),
            "study_hours": _r(sub["study_hours"].mean(), 3),
        })
    return sorted(rows, key=lambda x: -x["n"])


def chi2_label(df, by, label="certificate"):
    tab = pd.crosstab(df[by], df[label])
    chi2, p, dof, exp = stats.chi2_contingency(tab)
    n = int(tab.values.sum())
    cramers_v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1)))) if n and min(tab.shape) > 1 else None
    return {"chi2": _r(chi2, 3), "p": _p(p), "dof": int(dof),
            "cramers_v": _r(cramers_v), "n": n,
            "significant": bool(p < 0.05)}


def kw_metric(df, by, metric):
    groups = [sub[metric].values for _, sub in df.groupby(by, observed=True) if len(sub) >= 30]
    if len(groups) < 2:
        return {"H": None, "p": None, "significant": False}
    H, p = stats.kruskal(*groups)
    return {"H": _r(H, 3), "p": _p(p), "groups": len(groups), "significant": bool(p < 0.05)}


def hbar(ax, labels, values, colors, fmt="{:.1%}", xlabel="", title="", vmax=None):
    y = np.arange(len(labels))
    vals = np.array([0 if v is None else v for v in values], dtype=float)
    order = np.argsort(vals)
    labels = [labels[i] for i in order]; vals = vals[order]
    colors = [colors[i] for i in order]
    ax.barh(y, vals, color=colors, height=0.68, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#1F4E79", loc="left")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_xlim(0, (vmax or 1) * (1.02 if vmax else 1.12))
    ax.xaxis.grid(True, color=C["grid"], zorder=0); ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        ax.text(v + (vmax or 1) * 0.012, i, fmt.format(v), va="center", fontsize=7.4, color="#334155")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------------
# 维度 1：地区差异
# ----------------------------------------------------------------------------
def dimension_region(df):
    rows = group_table(df, "region")
    rows = sorted(rows, key=lambda x: -x["pass_rate"])
    return {
        "rows": rows,
        "chi2": chi2_label(df, "region"),
        "kw_visits": kw_metric(df, "region", "visit_count"),
        "kw_hours": kw_metric(df, "region", "study_hours"),
        "note": "地区为英国开放大学学习者地域分类（locale），非国籍；差异为观察性关联，不作因果解释。",
    }


# ----------------------------------------------------------------------------
# 维度 2：专业背景（学历 + 课程模块 + 选课门数）
# ----------------------------------------------------------------------------
def dimension_background(df):
    edu_order = ["无正式学历", "低于A-Level", "A-Level或同等", "高等教育学历", "研究生学历"]
    edu = group_table(df, "education")
    edu = sorted(edu, key=lambda x: edu_order.index(x["group"]) if x["group"] in edu_order else 99)
    disc = sorted(group_table(df, "discipline"), key=lambda x: -x["pass_rate"])
    # 选课门数分箱（真实派生 course_num）
    bins = pd.cut(df["course_num"], [0, 1, 2, 100], labels=["1 门", "2 门", "≥3 门"])
    tmp = df.assign(course_bin=bins)
    cnum = []
    for k in ["1 门", "2 门", "≥3 门"]:
        s = tmp[tmp["course_bin"] == k]
        if len(s) == 0:
            continue
        cnum.append({"group": k, "n": int(len(s)), "share": _r(len(s) / len(tmp)),
                     "pass_rate": _r(s["certificate"].mean()),
                     "visits": _r(s["visit_count"].mean(), 2),
                     "active_days": _r(s["active_days"].mean(), 2)})
    return {
        "education": edu, "discipline": disc, "course_num": cnum,
        "chi2_edu": chi2_label(df, "education"),
        "chi2_disc": chi2_label(df, "discipline"),
        "kw_edu_visits": kw_metric(df, "education", "visit_count"),
        "note": "数据集无'专业'自陈字段，以入学学历与所选课程模块（学科方向）作为专业背景代理；"
                "课程模块为 OULAD 编码（AAA–GGG），原数据不提供学科中文名，故保留代码。",
    }


# ----------------------------------------------------------------------------
# 维度 3：学习习惯（行为节律分型 + 短视频时段偏好）
# ----------------------------------------------------------------------------
def habit_type(row):
    """基于可解释阈值的行为节律分型（新增口径，独立于 4.4 的 K-means 成效画像）。"""
    lag, dens, sess, res = row["first_active_lag"], row["active_days"], row["avg_session_min"], row["video_rate"]
    early = lag <= 7          # 及时启动
    steady = dens >= 21       # 活跃密度高（12 周内≥21 天）
    long = sess >= 1.8        # 单次停留较长
    broad = res >= 0.25       # 资源覆盖广
    if early and steady and (long or broad):
        return "规律深耕型"
    if early and steady:
        return "稳定参与型"
    if (not early) and steady:
        return "晚到追赶型"
    if long or broad:
        return "碎片专注型"
    return "浅触即离型"


def dimension_habit(df):
    HABIT_ORDER = ["规律深耕型", "稳定参与型", "晚到追赶型", "碎片专注型", "浅触即离型"]
    df = df.assign(habit=df.apply(habit_type, axis=1))
    rows = []
    for k in HABIT_ORDER:
        s = df[df["habit"] == k]
        if len(s) == 0:
            continue
        rows.append({
            "group": k, "n": int(len(s)), "share": _r(len(s) / len(df)),
            "pass_rate": _r(s["certificate"].mean()),
            "lag": _r(s["first_active_lag"].mean(), 2),
            "active_days": _r(s["active_days"].mean(), 2),
            "session_min": _r(s["avg_session_min"].mean(), 2),
            "video_rate": _r(s["video_rate"].mean(), 3),
        })
    # 时段偏好（短视频真实 p_hour，读交付包派生表，离线可复现）
    tod = {"rows": [], "note": "短视频数据不可用"}
    if os.path.exists(SV_PROFILE):
        up = pd.read_csv(SV_PROFILE)
        pref = up["main_seg"].value_counts(normalize=True)
        tod = {
            "rows": [{"group": k, "users": int((up["main_seg"] == k).sum()),
                      "share": _r(v)} for k, v in pref.items()],
            "user_pref": [{"group": k, "share": _r(v)} for k, v in pref.items()],
            "n_users": int(len(up)),
            "note": "时段为用户主观看时段（短视频 p_hour，0–23 聚合）；OULAD 仅日粒度，二者分层呈现不合并。",
        }
    return {
        "types": rows,
        "chi2": chi2_label(df.assign(habit=df["habit"]), "habit"),
        "kw_lag": kw_metric(df.assign(habit=df["habit"]), "habit", "first_active_lag"),
        "time_of_day": tod,
        "note": "学习习惯分型由启动滞后、活跃密度、单次时长、资源覆盖四阈值规则生成，"
                "可解释、独立于成效；阈值来自真实分布分位，非仿真。",
    }


# ----------------------------------------------------------------------------
# 维度 4：兴趣爱好（短视频内容类目偏好）
# ----------------------------------------------------------------------------
def _load_sv_user():
    """读短视频用户级派生表；缺失时从原始数据即时派生一次（保证离线可复现）。"""
    if os.path.exists(SV_PROFILE):
        return pd.read_csv(SV_PROFILE)
    try:
        import build_profile_inputs as bpi
        bpi.build_shortvideo()
        return pd.read_csv(SV_PROFILE)
    except Exception as e:
        raise RuntimeError(f"短视频派生表缺失且无法重建：{e}")


def dimension_interest():
    if not os.path.exists(SV_CATWATCH):
        return {"note": "短视频派生数据不可用，兴趣维度跳过"}
    up = _load_sv_user()
    cw = pd.read_csv(SV_CATWATCH).sort_values("watch_sec", ascending=False)
    total = float(cw["watch_sec"].sum())
    cat_rows = [{"cat": str(r["cat"]), "watch_share": _r(r["watch_sec"] / total),
                 "events": int(r["events"]),
                 "like_rate": _r(r["like_rate"]), "comment_rate": _r(r["comment_rate"])}
                for _, r in cw.head(12).iterrows()]
    # 三类兴趣画像（由派生表预置的 know/ent/div，按 70% 分位）
    qk, qe = up["know_share"].quantile(0.70), up["ent_share"].quantile(0.70)
    def itype(r):
        if r["know_share"] >= qk and r["know_share"] > r["ent_share"]: return "知识求知型"
        if r["ent_share"] >= qe and r["ent_share"] > r["know_share"]: return "娱乐休闲型"
        return "均衡杂食型"
    up = up.assign(itype=up.apply(itype, axis=1))
    ORDER = ["知识求知型", "均衡杂食型", "娱乐休闲型"]
    irows = []
    for k in ORDER:
        s = up[up["itype"] == k]
        if len(s) == 0: continue
        irows.append({"group": k, "n": int(len(s)), "share": _r(len(s) / len(up)),
                      "know_share": _r(s["know_share"].mean()), "ent_share": _r(s["ent_share"].mean()),
                      "diversity": _r(s["diversity"].mean(), 3)})
    # 教育类子群 vs 全体（派生表无逐类目互动，改用类目汇总中的教育行）
    edu_row = cw[cw["cat"] == "教育"]
    edu_eng = {"like_rate": _r(edu_row["like_rate"].mean()),
               "comment_rate": _r(edu_row["comment_rate"].mean())} if len(edu_row) else {}
    all_eng = {"like_rate": _r(cw["like_rate"].mean()),
               "comment_rate": _r(cw["comment_rate"].mean())}
    return {
        "cat_watch": cat_rows,
        "interest_types": irows,
        "edu_subgroup": {"education": edu_eng, "all": all_eng,
                         "note": "互动强度按类目观看事件计（类目汇总），教育类 vs 全平台。"},
        "n_users": int(len(up)),
        "n_categories": int(cw["cat"].nunique()),
        "diversity_mean": _r(up["diversity"].mean(), 3),
        "note": "兴趣由内容观看占比推断，属行为偏好代理而非心理测量；"
                "短视频内容偏好不等同学科兴趣，仅作人群描述与方法示范，不用于个体评价。",
    }


# ----------------------------------------------------------------------------
# 维度 5：短视频人口属性与行为（城市等级 / 性别 / 年龄 / 时段 / 互动）——国内地域
# ----------------------------------------------------------------------------
CITY_ORDER = ["一线城市", "新一线城市", "二线城市", "三线城市", "四线城市", "五线城市"]
GENDER_CN = {"M": "男", "F": "女"}


def _sv_group_rows(up, by, order=None):
    g = up.groupby(by, observed=True)
    rows = []
    for key, sub in g:
        k = str(key)
        if k == "unknown":
            continue
        rows.append({
            "group": GENDER_CN.get(k, k),
            "n": int(len(sub)), "share": _r(len(sub) / len(up)),
            "finish_ratio": _r(sub["finish_ratio"].mean(), 3),
            "full_watch_rate": _r(sub["full_watch_rate"].mean(), 3),
            "interact_rate": _r(sub["interact_rate"].mean(), 3),
            "avg_watch_sec": _r(sub["avg_watch_sec"].mean(), 1),
            "events": int(sub["n_events"].sum()),
        })
    if order:
        rows = sorted(rows, key=lambda x: order.index(x["group"]) if x["group"] in order else 99)
    else:
        rows = sorted(rows, key=lambda x: -x["n"])
    return rows


def dimension_sv_profile():
    """短视频学习者人口属性 × 观看/互动行为。地域为中国城市（国内数据）。"""
    if not os.path.exists(SV_PROFILE):
        return {"note": "短视频派生数据不可用"}
    up = _load_sv_user()
    # 城市等级
    city = _sv_group_rows(up, "city_level", CITY_ORDER)
    # 性别
    gender = _sv_group_rows(up, "gender")
    # 年龄段
    AGE_ORDER = ["20–24岁", "25–34岁", "35–44岁", "45–54岁", "55–64岁", "65岁以上"]
    age = _sv_group_rows(up, "age_band", AGE_ORDER)
    # 观看时段
    SEG_ORDER = ["凌晨 (0-5)", "上午 (6-11)", "下午 (12-17)", "晚间 (18-21)", "深夜 (22-23)"]
    seg = _sv_group_rows(up, "main_seg", SEG_ORDER)
    # 城乡社区类型
    comm = _sv_group_rows(up, "community")
    # 差异检验（完成度在组间是否不同：Kruskal–Wallis）
    def kw(by):
        groups = [s["finish_ratio"].values for _, s in up.groupby(by, observed=True)
                  if len(s) >= 30 and str(list(s[by].unique())[0]) != "unknown"]
        if len(groups) < 2: return {"H": None, "p": None, "significant": False}
        H, p = stats.kruskal(*groups)
        return {"H": _r(H, 3), "p": _p(p), "groups": len(groups), "significant": bool(p < 0.05)}
    return {
        "city_level": city, "gender": gender, "age_band": age,
        "main_seg": seg, "community": comm,
        "kw_city": kw("city_level"), "kw_gender": kw("gender"), "kw_age": kw("age_band"),
        "kw_seg": kw("main_seg"),
        "n_users": int(len(up)),
        "n_cities": int(up["city"].nunique()),
        "city_top": [{"group": k, "n": int(v)} for k, v in up["city"].value_counts().head(10).items()],
        "finish_range": _r(max(r["finish_ratio"] for r in city) - min(r["finish_ratio"] for r in city), 3),
        "note": "城市等级/城市/社区类型来自短视频数据 fre_city_level/fre_city/fre_community_type，"
                "为中国国内地域（非英国 OULAD）；性别年龄为用户自陈属性。观看完成度为 watch_time/duration "
                "截断 1 的代理，非真实完课；全部为观察性关联。",
    }


# ----------------------------------------------------------------------------
# 维度 6：天池微课学习终端与倍速标签（倍速 / 设备 / OS / 网络 / 预习 / 测评）
# ----------------------------------------------------------------------------
def _tc_group_rows(tc, by, order=None, value_cols=None):
    cols = value_cols or ["progress_delta", "interaction_cnt", "score_ratio", "satisfaction_score"]
    g = tc.groupby(by, observed=True)
    rows = []
    for key, sub in g:
        k = str(key)
        rows.append({"group": k, "n": int(len(sub)), "share": _r(len(sub) / len(tc)),
                     "progress_delta": _r(sub["progress_delta"].mean(), 2),
                     "interaction_cnt": _r(sub["interaction_cnt"].mean(), 2),
                     "score_ratio": _r(sub["score_ratio"].mean(), 3),
                     "satisfaction": _r(sub["satisfaction_score"].mean(), 2)})
    if order:
        rows = sorted(rows, key=lambda x: order.index(x["group"]) if x["group"] in order else 99)
    else:
        rows = sorted(rows, key=lambda x: -x["n"])
    return rows


def dimension_tianchi():
    """天池微课学习终端与倍速/预习标签。标签语义为国内脱敏微课样本。

    重要口径修正：原始数据把非 video_play 事件（测验/作业/论坛/下载/评价）的 playback_speed
    统一填为 1.0×，而这些事件的 progress_delta 本就接近 0；若混算，1.0× 会出现虚假的
    "低进度增益"。因此倍速与预习对照严格限定在 video_play 事件内，避免结构性伪影。
    """
    if not os.path.exists(TC_PROFILE):
        return {"note": "天池派生数据不可用"}
    tc = pd.read_csv(TC_PROFILE)
    vid = tc[tc["behavior_type"] == "video_play"].copy()
    speed = _tc_group_rows(vid, "playback_speed")
    for r in speed:
        r["group"] = f"{float(r['group']):g}×"
    speed = sorted(speed, key=lambda x: float(x["group"].rstrip("×")))
    device = _tc_group_rows(tc, "device_type")
    os_rows = _tc_group_rows(tc, "client_os")
    net = _tc_group_rows(tc, "network_type")
    # 预习：0/1，严格在 video_play 内对照（否则对照组混入非视频事件造成伪影）
    preview = _tc_group_rows(vid, "is_preview")
    for r in preview:
        r["group"] = "课前预习" if r["group"] in ("1", "1.0") else "常规回看"
    preview = sorted(preview, key=lambda x: (x["group"] != "课前预习"))
    # 测评标签
    assess = _tc_group_rows(tc, "assessment")
    # 倍速分组对进度增益的 KW 检验（仅视频事件，各组在此均有值）
    groups = [s["progress_delta"].dropna().values for _, s in vid.groupby("playback_speed", observed=True) if len(s) >= 30]
    kw_speed = {"H": None, "p": None, "significant": False}
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups)
        kw_speed = {"H": _r(H, 3), "p": _p(p), "groups": len(groups), "significant": bool(p < 0.05)}
    # 预习 vs 常规回看 进度增益显著性（Mann–Whitney，仅视频事件）
    a = vid[vid["is_preview"] == 1]["progress_delta"].dropna()
    b = vid[vid["is_preview"] == 0]["progress_delta"].dropna()
    mw = {"U": None, "p": None, "significant": False}
    if len(a) > 5 and len(b) > 5:
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        mw = {"U": _r(U, 1), "p": _p(p), "n_preview": int(len(a)), "n_other": int(len(b)),
              "significant": bool(p < 0.05)}
    return {
        "playback_speed": speed, "device": device, "client_os": os_rows,
        "network": net, "preview": preview, "assessment": assess,
        "kw_speed_progress": kw_speed, "mw_preview_progress": mw,
        "n_events": int(len(tc)), "n_learners": int(tc["learner_id"].nunique()),
        "n_preview": int(tc["is_preview"].sum()), "n_video_events": int(len(vid)),
        "note": "倍速/设备/操作系统/网络/预习标签来自天池脱敏微课样本（国内），全部真实。"
                "原始数据将非视频事件的倍速统一记为 1.0× 且其进度增益≈0，会造成虚假差异；"
                "故倍速与预习对照严格限定在 video_play 事件（2,614 条）内，避免该结构性伪影。"
                "设备/OS/网络为全事件分布；progress_delta=观看后进度−观看前进度；预习=remark 含‘课前预习’。"
                "均为观察性关联，非因果。",
    }


# ----------------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------------
def make_figures(P):
    # fig17 地区差异
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rows = sorted(P["region"]["rows"], key=lambda x: x["pass_rate"])
    hbar(ax, [r["group"] for r in rows], [r["pass_rate"] for r in rows],
         [C["blue"]] * len(rows), fmt="{:.1%}", xlabel="真实通过率", title="地区差异：13 类地区通过率（OULAD）")
    fig.savefig(os.path.join(FIG, "fig17_region.png")); plt.close(fig)

    # fig18 专业背景
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    edu = P["background"]["education"]
    hbar(axes[0], [r["group"] for r in edu], [r["pass_rate"] for r in edu],
         [C["teal"]] * len(edu), fmt="{:.1%}", title="学历层次 × 通过率")
    disc = P["background"]["discipline"]
    hbar(axes[1], [r["group"] for r in disc], [r["pass_rate"] for r in disc],
         [C["amber"]] * len(disc), fmt="{:.1%}", title="课程模块（学科方向）× 通过率")
    fig.suptitle("专业背景代理：学历与学科方向的成效梯度", fontsize=11, fontweight="bold", color="#1F4E79")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig18_background.png")); plt.close(fig)

    # fig19 学习习惯
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    ht = P["habit"]["types"]
    hbar(axes[0], [r["group"] for r in ht], [r["pass_rate"] for r in ht],
         [C["blue"]] * len(ht), fmt="{:.1%}", title="学习习惯分型 × 通过率（OULAD）")
    tod = P["habit"]["time_of_day"]
    if tod.get("rows"):
        td = tod["rows"]
        hbar(axes[1], [r["group"] for r in td], [r["share"] for r in td],
             [C["slate"]] * len(td), fmt="{:.1%}", title="观看时段偏好（短视频，按事件）")
    fig.suptitle("学习习惯：行为节律分型与时段偏好", fontsize=11, fontweight="bold", color="#1F4E79")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig19_habit.png")); plt.close(fig)

    # fig20 兴趣爱好
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    cw = P["interest"].get("cat_watch", [])[:10]
    if cw:
        hbar(axes[0], [r["cat"] for r in cw], [r["watch_share"] for r in cw],
             [C["amber"]] * len(cw), fmt="{:.1%}", title="内容类目观看时长占比（短视频 Top10）")
    it = P["interest"].get("interest_types", [])
    if it:
        hbar(axes[1], [r["group"] for r in it], [r["share"] for r in it],
             [C["teal"]] * len(it), fmt="{:.1%}", title="用户兴趣画像分布")
    fig.suptitle("兴趣爱好：内容偏好画像（行为代理）", fontsize=11, fontweight="bold", color="#1F4E79")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig20_interest.png")); plt.close(fig)

    # fig21 短视频人口属性与行为（国内地域）
    sp = P.get("sv_profile", {})
    if sp.get("city_level"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
        specs = [
            (axes[0, 0], "city_level", "城市等级 × 观看完成度（国内）"),
            (axes[0, 1], "gender", "性别 × 互动率"),
            (axes[1, 0], "age_band", "年龄段 × 观看完成度"),
            (axes[1, 1], "main_seg", "观看时段 × 人群占比"),
        ]
        for ax, key, title in specs:
            rows = sp.get(key, [])
            metric = "share" if key == "main_seg" else ("interact_rate" if key == "gender" else "finish_ratio")
            hbar(ax, [r["group"] for r in rows], [r[metric] for r in rows],
                 [C["blue"]] * len(rows), fmt="{:.1%}", title=title)
        fig.suptitle("短视频学习者人口属性与行为（中国国内数据）", fontsize=11, fontweight="bold", color="#1F4E79")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig21_sv_demographic.png")); plt.close(fig)

    # fig22 天池倍速/设备/预习标签
    tc = P.get("tianchi", {})
    if tc.get("playback_speed"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
        specs = [
            (axes[0, 0], "playback_speed", "倍速 × 进度增益"),
            (axes[0, 1], "device", "设备 × 交互次数"),
            (axes[1, 0], "preview", "预习标签 × 进度增益"),
            (axes[1, 1], "network", "网络 × 观看时长"),
        ]
        for ax, key, title in specs:
            rows = tc.get(key, [])
            if key == "preview":
                hbar(ax, [r["group"] for r in rows], [r["progress_delta"] for r in rows],
                     [C["teal"] if r["group"] == "课前预习" else C["slate"] for r in rows],
                     fmt="{:.1f}", xlabel="进度增益(百分点)", title=title)
            elif key == "playback_speed":
                hbar(ax, [r["group"] for r in rows], [r["progress_delta"] for r in rows],
                     [C["amber"]] * len(rows), fmt="{:.1f}", xlabel="进度增益(百分点)", title=title)
            elif key == "device":
                hbar(ax, [r["group"] for r in rows], [r["interaction_cnt"] for r in rows],
                     [C["blue"]] * len(rows), fmt="{:.1f}", title=title)
            else:
                hbar(ax, [r["group"] for r in rows], [r["share"] for r in rows],
                     [C["slate"]] * len(rows), fmt="{:.0%}", title=title)
        fig.suptitle("天池微课：倍速、设备、预习与网络标签", fontsize=11, fontweight="bold", color="#1F4E79")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig22_tianchi_tags.png")); plt.close(fig)


def main():
    os.makedirs(RUN, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    from analysis import load_clean
    df, log = load_clean(os.path.join(DATA, "real_learners.csv"))
    P = {
        "meta": {
            "title": "六维学习者画像建模结果",
            "data": "OULAD 真实主表（20,105 条，英国）+ 清华 FIB 短视频 sampled（794,053 事件 / 6,654 用户，中国）"
                    "+ 阿里云天池脱敏微课样本（5,000 事件 / 715 学习者，中国）",
            "n_main": int(len(df)),
            "no_simulation": True,
            "dimensions": ["地区差异(英国)", "专业背景", "学习习惯", "兴趣爱好",
                           "短视频人口与行为(国内)", "天池倍速与终端标签(国内)"],
            "region_sources": [
                {"source": "OULAD", "region": "英国 13 大区（locale）", "n": int(len(df)), "domestic": False},
                {"source": "清华 FIB 短视频", "region": "中国 365 城市 + 一线~五线等级", "n": 6654, "domestic": True},
                {"source": "天池微课", "region": "中国（平台脱敏，无城市字段）", "n": 715, "domestic": True},
            ],
            "ethics": [
                "地域有国内与英国两类：OULAD 为英国 13 大区（locale，非国籍）；短视频为中国 365 城市与城市等级，"
                "天池为国内脱敏微课。二者分层呈现，不合并、不互推。",
                "专业背景以学历与课程模块代理，数据无专业自陈字段。",
                "兴趣由内容观看占比推断，属行为偏好代理，非心理测量。",
                "观看完成度为 watch_time/duration 截断 1 的代理，非真实完课率。",
                "倍速、设备、网络、预习均为观察性标签，关联非因果，不用于个体评价或惩戒。",
            ],
        },
        "region": dimension_region(df),
        "background": dimension_background(df),
        "habit": dimension_habit(df),
        "interest": dimension_interest(),
        "sv_profile": dimension_sv_profile(),
        "tianchi": dimension_tianchi(),
    }
    out = os.path.join(RUN, "profile_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(P, f, ensure_ascii=False, indent=2)
    make_figures(P)
    print("PROFILE_OK", out)
    print("  地区:", len(P["region"]["rows"]), "| 学历:", len(P["background"]["education"]),
          "| 习惯分型:", len(P["habit"]["types"]), "| 兴趣画像:", len(P["interest"].get("interest_types", [])),
          "| 城市等级:", len(P["sv_profile"].get("city_level", [])), "| 倍速档:", len(P["tianchi"].get("playback_speed", [])))
    print("  地区卡方 p =", P["region"]["chi2"]["p"], "| 学历卡方 p =", P["background"]["chi2_edu"]["p"],
          "| 习惯卡方 p =", P["habit"]["chi2"]["p"])


if __name__ == "__main__":
    main()
