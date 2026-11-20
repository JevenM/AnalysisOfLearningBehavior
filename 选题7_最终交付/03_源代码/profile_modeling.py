# -*- coding: utf-8 -*-
"""
profile_modeling.py —— 内容级学习者画像建模（观看时段 / 兴趣爱好 /
                     短视频用户属性 / 天池微课标签）
《系统工程原理》实践项目 选题7

本脚本只做"内容级"画像：数据来自六类真实数据集中的国内内容源
（清华 FIB 短视频、阿里天池微课）。课程级行为（含 OULAD）统一由
up_ 统一管线（up_build_unified → up_report）在六源合并口径下分析，
本脚本不再单独拆分任何一个课程数据集。

四个维度全部基于真实数据，无仿真：
  1) 观看时段偏好  : 短视频 p_hour 主观看时段（6,654 用户真实时段）
  2) 兴趣爱好      : 短视频 root_id 一级类目（38 类）用户偏好画像
                     + 教育类内容偏好子群行为对照
  3) 短视频用户属性: 城市等级 / 城乡 / 性别 / 年龄 / 观看时段（国内数据，6,654 用户）
  4) 天池微课标签  : 倍速 / 设备 / OS / 网络 / 预习 / 学习状态（国内数据，脱敏样本）

口径与伦理边界（写入结果 meta，报告须如实引用）：
  - "兴趣"由内容观看占比推断，属行为偏好代理，非心理测量；短视频平台内容偏好
    不等同学科兴趣，仅作方法示范与人群描述，不用于个体评价。
  - 群体差异均为观察性关联，非因果；不做个体 ranking 或惩戒。

输出：
  04_运行结果/profile_results.json
  05_图表与看板/figures/fig20_interest.png / fig21_sv_demographic.png / fig22_tianchi_tags.png
"""
from __future__ import annotations

import json
import os

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
    # x 上限自适应：显式 vmax 给定时按原口径，否则取数据最大值的 1.12 倍——
    # 修复非百分比指标（进度增益/交互次数）被硬编码 1.12 裁剪、标签溢出画布的问题
    xmax = float(vmax) * 1.02 if vmax is not None else max(float(vals.max()) * 1.12, 1e-9)
    ax.set_xlim(0, xmax)
    ax.xaxis.grid(True, color=C["grid"], zorder=0); ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        ax.text(v + xmax * 0.012, i, fmt.format(v), va="center", fontsize=7.4, color="#334155")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------------
# 维度 1：观看时段偏好（短视频真实 p_hour，读交付包派生表，离线可复现）
# ----------------------------------------------------------------------------
def dimension_time_of_day():
    tod = {"rows": [], "note": "短视频派生数据不可用"}
    if os.path.exists(SV_PROFILE):
        up = pd.read_csv(SV_PROFILE)
        pref = up["main_seg"].value_counts(normalize=True)
        tod = {
            "rows": [{"group": k, "users": int((up["main_seg"] == k).sum()),
                      "share": _r(v)} for k, v in pref.items()],
            "user_pref": [{"group": k, "share": _r(v)} for k, v in pref.items()],
            "n_users": int(len(up)),
            "note": "时段为用户主观看时段（短视频 p_hour，0–23 聚合），国内真实数据。",
        }
    return tod


# ----------------------------------------------------------------------------
# 维度 2：兴趣爱好（短视频内容类目偏好）
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
# 维度 3：短视频人口属性与行为（城市等级 / 性别 / 年龄 / 时段 / 互动）——国内地域
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
                "为中国国内地域；性别年龄为用户自陈属性。观看完成度为 watch_time/duration "
                "截断 1 的代理，非真实完课；全部为观察性关联。",
    }


# ----------------------------------------------------------------------------
# 维度 4：天池微课学习终端与倍速标签（倍速 / 设备 / OS / 网络 / 预习 / 测评）
# ----------------------------------------------------------------------------
def _tc_group_rows(tc, by, order=None, value_cols=None):
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
                "故倍速与预习对照严格限定在 video_play 事件内，避免该结构性伪影。"
                "设备/OS/网络为全事件分布；progress_delta=观看后进度−观看前进度；预习=remark 含‘课前预习’。"
                "均为观察性关联，非因果。",
    }


# ----------------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------------
def make_figures(P):
    # fig20 兴趣爱好 + 观看时段
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    cw = P["interest"].get("cat_watch", [])[:10]
    if cw:
        hbar(axes[0], [r["cat"] for r in cw], [r["watch_share"] for r in cw],
             [C["blue"]] * len(cw), fmt="{:.1%}", title="内容类目观看占比 Top10")
    td = P.get("time_of_day", {}).get("rows", [])
    if td:
        hbar(axes[1], [r["group"] for r in td], [r["share"] for r in td],
             [C["slate"]] * len(td), fmt="{:.1%}", title="观看时段偏好（按用户主时段）")
    fig.suptitle("兴趣爱好与观看时段：内容偏好画像（行为代理）", fontsize=11, fontweight="bold", color="#1F4E79")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig20_interest.png")); plt.close(fig)

    # fig21 短视频人口属性与行为（国内地域）
    sp = P.get("sv_profile", {})
    if sp.get("city_level"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
        specs = [
            ("city_level", "finish_ratio", "城市等级 × 观看完成度"),
            ("gender", "interact_rate", "性别 × 互动率"),
            ("age_band", "avg_watch_sec", "年龄段 × 平均单次观看时长"),
            ("main_seg", "share", "观看时段 × 用户占比"),
        ]
        for ax, (key, metric, title) in zip(axes.flat, specs):
            rows = sp.get(key, [])
            if not rows:
                ax.axis("off"); continue
            fmt = "{:.1%}" if metric in ("finish_ratio", "interact_rate", "share") else "{:.0f}s"
            hbar(ax, [r["group"] for r in rows], [r[metric] for r in rows],
                 [C["blue"]] * len(rows), fmt=fmt, title=title)
        fig.suptitle("短视频学习者人口属性与行为（中国国内数据）", fontsize=11, fontweight="bold", color="#1F4E79")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig21_sv_demographic.png")); plt.close(fig)

    # fig22 天池倍速/设备/预习标签
    tc = P.get("tianchi", {})
    if tc.get("playback_speed"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
        specs = [
            ("playback_speed", "progress_delta", "倍速档 × 进度增益"),
            ("device", "share", "设备类型分布"),
            ("preview", "progress_delta", "预习 × 进度增益"),
            ("network", "share", "网络类型分布"),
        ]
        for ax, (key, metric, title) in zip(axes.flat, specs):
            rows = tc.get(key, [])
            if not rows:
                ax.axis("off"); continue
            fmt = "{:.1%}" if metric == "share" else "{:.3f}"
            hbar(ax, [r["group"] for r in rows], [r[metric] for r in rows],
                 [C["slate"]] * len(rows), fmt=fmt, title=title)
        fig.suptitle("天池微课：倍速、设备、预习与网络标签", fontsize=11, fontweight="bold", color="#1F4E79")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig22_tianchi_tags.png")); plt.close(fig)


def main():
    os.makedirs(RUN, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    P = {
        "meta": {
            "no_simulation": True,
            "dimensions": ["观看时段偏好", "兴趣爱好",
                           "短视频人口与行为(国内)", "天池倍速与终端标签(国内)"],
            "region_sources": [
                {"source": "清华 FIB 短视频", "region": "中国 365 城市 + 一线~五线等级", "n": 6654, "domestic": True},
                {"source": "天池微课", "region": "中国（平台脱敏，无城市字段）", "n": 715, "domestic": True},
            ],
            "ethics": [
                "兴趣由内容观看占比推断，属行为偏好代理，非心理测量。",
                "观看完成度为 watch_time/duration 截断 1 的代理，非真实完课率。",
                "倍速、设备、网络、预习均为观察性标签，关联非因果，不用于个体评价或惩戒。",
            ],
        },
        "time_of_day": dimension_time_of_day(),
        "interest": dimension_interest(),
        "sv_profile": dimension_sv_profile(),
        "tianchi": dimension_tianchi(),
    }
    out = os.path.join(RUN, "profile_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(P, f, ensure_ascii=False, indent=2)
    make_figures(P)
    print("PROFILE_OK", out)
    print("  时段偏好:", len(P["time_of_day"].get("rows", [])), "| 兴趣画像:", len(P["interest"].get("interest_types", [])),
          "| 城市等级:", len(P["sv_profile"].get("city_level", [])), "| 倍速档:", len(P["tianchi"].get("playback_speed", [])))


if __name__ == "__main__":
    main()
