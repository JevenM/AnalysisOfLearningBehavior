# -*- coding: utf-8 -*-
"""从原始多形态数据派生"细粒度画像输入表"，使交付包自包含。

背景：profile_modeling.py 早期直接读工作区外部原始路径，不利于交付复现。
本脚本一次性把两份真实原始表中的细字段聚合/整理为交付包内 CSV：
  1) 短视频用户级画像表  shortvideo_profile.csv（6,654 用户 × 人口属性 + 行为 + 兴趣）
  2) 天池事件级标签表    tianchi_profile.csv  （5,000 事件 × 倍速/设备/OS/网络/预习/测评）

全部来自真实公开数据（清华 FIB WWW2025 短视频 sampled；阿里云天池脱敏微课样本），
无仿真、无随机生成；仅做确定性聚合与字段重命名。
原始路径不存在时给出明确报错，不产生占位数据。
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "02_数据与数据集")
RAW = os.path.abspath(os.path.join(ROOT, "..", "选题7_学习者行为分析",
                                   "data", "multimodal", "raw"))
RAW_SV = os.path.join(RAW, "interaction_sampled.csv")
RAW_CAT = os.path.join(RAW, "categories_cn_en.csv")
RAW_TC = os.path.join(RAW, "tianchi", "user_learning_behavior.csv")

INTERACT = ["cvm_like", "click", "comment", "follow", "collect", "forward", "hate"]


def age_band(a):
    if a < 25: return "20–24岁"
    if a < 35: return "25–34岁"
    if a < 45: return "35–44岁"
    if a < 55: return "45–54岁"
    if a < 65: return "55–64岁"
    return "65岁以上"


def hour_seg(h):
    if h < 6: return "凌晨 (0-5)"
    if h < 12: return "上午 (6-11)"
    if h < 18: return "下午 (12-17)"
    if h < 22: return "晚间 (18-21)"
    return "深夜 (22-23)"


def build_shortvideo():
    for p in (RAW_SV, RAW_CAT):
        if not os.path.exists(p):
            raise FileNotFoundError(f"短视频原始数据缺失：{p}")
    sv = pd.read_csv(RAW_SV)
    cat = pd.read_csv(RAW_CAT)
    r1 = cat[cat["category_level"] == 1][["category_id", "category_name_cn"]].drop_duplicates("category_id")
    name = dict(zip(r1["category_id"], r1["category_name_cn"]))
    sv["cat"] = sv["root_id"].map(name).fillna("其他")

    # 观看完成度：watch/duration 截断 1（重复观看不额外加分）；完整观看比例另列
    d = sv["duration"].replace(0, np.nan)
    sv["finish_ratio"] = (sv["watch_time"] / d).clip(0, 1).fillna(0)
    sv["full_watch"] = (sv["watch_time"] >= sv["duration"]).astype(float)
    sv["is_interact"] = (sv[INTERACT].sum(axis=1) > 0).astype(float)

    # 用户级：人口属性在用户内唯一（已核验）
    demo = sv.groupby("user_id").agg(
        city_level=("fre_city_level", "first"),
        city=("fre_city", "first"),
        community=("fre_community_type", "first"),
        gender=("gender", "first"),
        age=("age", "first"),
        n_events=("pid", "size"),
        avg_watch_sec=("watch_time", "mean"),
        finish_ratio=("finish_ratio", "mean"),
        full_watch_rate=("full_watch", "mean"),
        interact_rate=("is_interact", "mean"),
        like_rate=("cvm_like", "mean"),
        comment_rate=("comment", "mean"),
        collect_rate=("collect", "mean"),
    ).reset_index()
    demo["age_band"] = demo["age"].map(age_band)
    # 主观看时段（按事件数取众数）
    seg = sv.groupby(["user_id", "seg"]) if "seg" in sv else None
    sv["seg"] = sv["p_hour"].map(hour_seg)
    user_seg = sv.groupby("user_id")["seg"].agg(lambda x: x.value_counts().index[0])
    demo["main_seg"] = demo["user_id"].map(user_seg)

    # 兴趣向量：知识/娱乐占比 + 归一化熵 + 偏好类目
    piv = sv.pivot_table(index="user_id", columns="cat", values="watch_time",
                         aggfunc="sum", fill_value=0)
    frac = piv.div(piv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    know = [c for c in ["教育", "科学", "读书", "历史", "财经", "法律", "高新数码"] if c in frac]
    ent = [c for c in ["影视和短剧", "搞笑", "游戏", "音乐", "明星娱乐", "二次元"] if c in frac]
    p = frac.values
    entropy = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(axis=1)
    nc = (frac > 0).sum(axis=1).clip(lower=2)
    interest = pd.DataFrame({
        "know_share": frac[know].sum(axis=1).to_numpy(),
        "ent_share": frac[ent].sum(axis=1).to_numpy(),
        "diversity": np.asarray(entropy / np.log(nc)),
        "top_cat": frac.idxmax(axis=1).to_numpy(),
    }, index=frac.index)
    interest.index.name = "user_id"
    interest = interest.reset_index()
    demo = demo.merge(interest, on="user_id", how="left")
    demo = demo.rename(columns={"city_level": "city_level", "city": "city"})
    out = os.path.join(DATA, "shortvideo_profile.csv")
    demo.to_csv(out, index=False, encoding="utf-8-sig")
    print("SHORTVIDEO_PROFILE_OK", out, demo.shape)

    # 类目级观看时长汇总（保持兴趣维度原有口径，供建模脚本离线复现）
    cat_watch = sv.groupby("cat").agg(
        watch_sec=("watch_time", "sum"),
        events=("pid", "size"),
        like_rate=("cvm_like", "mean"),
        comment_rate=("comment", "mean"),
        collect_rate=("collect", "mean"),
    ).reset_index().sort_values("watch_sec", ascending=False)
    cat_out = os.path.join(DATA, "shortvideo_cat_watch.csv")
    cat_watch.to_csv(cat_out, index=False, encoding="utf-8-sig")
    print("SHORVIDEO_CAT_WATCH_OK", cat_out, cat_watch.shape)
    return demo


def build_tianchi():
    if not os.path.exists(RAW_TC):
        raise FileNotFoundError(f"天池原始数据缺失：{RAW_TC}")
    d = pd.read_csv(RAW_TC)
    d["is_preview"] = d["remark"].astype(str).str.contains("预习").astype(int)
    d["assessment"] = d["assessment_tag"].fillna("无标签")
    d["score_ratio"] = (d["score"] / d["full_score"].replace(0, np.nan)).clip(0, 1)
    d["progress_delta"] = d["progress_after_pct"] - d["progress_before_pct"]
    keep = ["learner_id", "session_id", "course_id", "behavior_type", "chapter_no",
            "device_type", "client_os", "network_type", "playback_speed",
            "is_preview", "remark", "assessment", "satisfaction_score",
            "interaction_cnt", "duration_sec", "score_ratio", "progress_delta", "is_passed"]
    d = d[keep].copy()
    out = os.path.join(DATA, "tianchi_profile.csv")
    d.to_csv(out, index=False, encoding="utf-8-sig")
    print("TIANCHI_PROFILE_OK", out, d.shape)
    return d


if __name__ == "__main__":
    sv = build_shortvideo()
    tc = build_tianchi()
    print("城市等级:", sv["city_level"].value_counts().to_dict())
    print("性别:", sv["gender"].value_counts().to_dict())
    print("预习事件:", int(tc["is_preview"].sum()), "| 倍速档:", sorted(tc["playback_speed"].unique()))
