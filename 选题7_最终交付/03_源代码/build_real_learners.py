# -*- coding: utf-8 -*-
"""
build_real_learners.py —— 从 OULAD 原始日志派生「学习者级真实特征主表」。

本脚本是本次重构的核心：把原来依赖教学仿真数据（learners.csv, N=8000）的
整条主线（清洗/画像/预测/评价/决策）替换为可核验的公开真实数据。

数据来源（Open University Learning Analytics Dataset, OULAD）：
    studentInfo.csv         人口属性 + 真实学业结果 final_result
    studentVle.csv          学生×资源×日期的点击日志（1065 万行）
    vle.csv                 资源 id → activity_type（区分论坛/内容/测验等）
    studentAssessment.csv   学生对测评的提交与得分
    assessments.csv         测评类型(TMA/CMA/Exam)、权重、截止日

口径与诚实性声明（重要，报告第 3 章引用）：
  * 全部字段来自上述真实公开表，无任何随机生成、无仿真样本。
  * 标签 certificate = (final_result ∈ {Pass, Distinction})，为真实学业结果，
    不是由输入行为特征派生，避免标签泄漏。
  * OULAD 不直接记录"学习时长"与"视频完播"。本表把
        study_hours     = 累计点击 × 单次点击等效投入(0.10 分钟) / 60
        avg_session_min = 累计点击 × 0.10 / 活跃天数
        video_rate      = 该生访问过的 oucontent 资源数 / 该课程 oucontent 资源总数
    作为【等效投入代理指标】，仅用于横向比较，不代表真实计时，报告中明确标注。
  * forum_posts 为论坛类(activity_type=forumng)点击数，是真实互动行为的代理。
  * quiz_avg / assign_rate / final_grade 直接来自真实测评得分与提交记录。
  * 只取 2013J / 2014J 两个标准 12 周春季学期，保证观察窗口一致、可比。

用法：
    python build_real_learners.py --oulad-dir <含 OULAD 原始 csv 的目录> \
                                  --out-dir <交付包/02_数据与数据集>

派生后的 real_learners.csv 随包交付，日常重跑 analysis.py / app.py 不需要原始数据。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

KEEP_PRES = ("2013J", "2014J")
CLICK_MIN = 0.10          # 单次点击等效投入（分钟）——折算假设，报告中披露
FORUM_TYPES = {"forumng"}          # 论坛互动类资源
CONTENT_TYPES = {"oucontent"}      # 教学内容类资源（视频/文档代理）

REGION_CN = {
    "Scotland": "苏格兰", "Wales": "威尔士", "Ireland": "爱尔兰",
    "London Region": "伦敦", "South Region": "南部", "South East Region": "东南",
    "South West Region": "西南", "East Anglian Region": "东盎格利亚",
    "East Midlands Region": "东米德兰兹", "West Midlands Region": "西米德兰兹",
    "North Region": "北部", "North Western Region": "西北", "Yorkshire Region": "约克郡",
}
EDU_CN = {
    "Lower Than A Level": "低于A-Level", "A Level or Equivalent": "A-Level或同等",
    "HE Qualification": "高等教育学历", "Post Graduate Qualification": "研究生学历",
    "No Formal quals": "无正式学历",
}
AGE_MID = {"0-35": 25.0, "35-55": 45.0, "55<=": 60.0}


def build(oulad_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    key = ["code_module", "code_presentation", "id_student"]

    # ---------- 1) 资源类型映射 ----------
    vle = pd.read_csv(os.path.join(oulad_dir, "vle.csv"),
                      usecols=["id_site", "code_module", "code_presentation", "activity_type"])
    vle["site_key"] = (vle["code_module"] + "|" + vle["code_presentation"] + "|"
                       + vle["id_site"].astype(str))
    atype = dict(zip(vle["site_key"], vle["activity_type"]))

    # 每门课 oucontent 资源总数（用于 video_rate 分母）
    content_total = (vle[vle["activity_type"].isin(CONTENT_TYPES)]
                     .groupby(["code_module", "code_presentation"])
                     .size().rename("content_total"))

    # ---------- 2) 点击日志分块聚合（433MB / 1065 万行） ----------
    daily_parts, site_parts = [], []
    reader = pd.read_csv(
        os.path.join(oulad_dir, "studentVle.csv"),
        usecols=key + ["id_site", "date", "sum_click"],
        chunksize=1_500_000,
    )
    for ch in reader:
        ch = ch[ch["code_presentation"].isin(KEEP_PRES)]
        ch = ch[(ch["date"] >= 0) & (ch["date"] < 84)].copy()
        if ch.empty:
            continue
        ch["site_key"] = (ch["code_module"] + "|" + ch["code_presentation"] + "|"
                          + ch["id_site"].astype(str))
        ch["atype"] = ch["site_key"].map(atype)
        ch["forum_click"] = np.where(ch["atype"].isin(FORUM_TYPES), ch["sum_click"], 0)
        ch["is_content"] = ch["atype"].isin(CONTENT_TYPES).astype(int)
        daily_parts.append(
            ch.groupby(key + ["date"], as_index=False)
              .agg(clicks=("sum_click", "sum"), forum=("forum_click", "sum")))
        cs = ch[ch["is_content"] == 1]
        if not cs.empty:
            site_parts.append(cs[key + ["id_site"]].drop_duplicates())

    daily = pd.concat(daily_parts, ignore_index=True)
    sites = pd.concat(site_parts, ignore_index=True).drop_duplicates()
    print("[派生] 有效学生-日记录", len(daily), "内容资源访问对", len(sites))

    # ---------- 3) 学生-课程级行为特征（全窗口，用于画像/评价） ----------
    beh = daily.groupby(key).agg(
        visit_count=("clicks", "sum"),          # 累计点击次数（真实）
        active_days=("date", "nunique"),        # 活跃天数（真实）
        forum_posts=("forum", "sum"),           # 论坛互动点击（真实代理）
        first_active_lag=("date", "min"),       # 首次参与滞后（真实）
    )
    # 前期行为（截至第 2 / 4 周）——用于无泄漏预测
    for w, cut in (("w2", 14), ("w4", 28)):
        sub = daily[daily["date"] < cut]
        g = sub.groupby(key).agg(**{f"early_{w}_clicks": ("clicks", "sum"),
                                    f"early_{w}_days": ("date", "nunique")})
        beh = beh.join(g)
    for c in [c for c in beh.columns if c.startswith("early_")]:
        beh[c] = beh[c].fillna(0)

    # 内容资源覆盖度 → video_rate（代理）
    cov = sites.groupby(key).size().rename("content_seen").to_frame()
    cov = cov.join(content_total, on=["code_module", "code_presentation"])
    cov["video_rate"] = (cov["content_seen"] / cov["content_total"]).clip(0, 1)

    # ---------- 4) 测评特征（真实得分/提交） ----------
    ass = pd.read_csv(os.path.join(oulad_dir, "assessments.csv"))
    sa = pd.read_csv(os.path.join(oulad_dir, "studentAssessment.csv"))
    m = sa.merge(ass[["id_assessment", "code_module", "code_presentation",
                      "assessment_type", "weight"]], on="id_assessment", how="left")
    m = m[m["code_presentation"].isin(KEEP_PRES)]
    mc = m[m["assessment_type"].isin(["TMA", "CMA"])].copy()   # 形成性测评
    req = mc.groupby(key).size().rename("req")
    sub = mc[mc["score"].notna()].groupby(key).size().rename("sub")
    assign = pd.concat([req, sub], axis=1).fillna(0)
    assign["assign_rate"] = np.where(assign["req"] > 0,
                                     assign["sub"] / assign["req"], 0.0).clip(0, 1)
    quiz = mc[mc["score"].notna()].groupby(key)["score"].mean().rename("quiz_avg")
    mg = m[m["score"].notna()].copy()
    mg["ws"] = mg["score"] * mg["weight"]
    fg = mg.groupby(key).agg(ws=("ws", "sum"), w=("weight", "sum"))
    fg["final_grade"] = np.where(fg["w"] > 0, fg["ws"] / fg["w"], np.nan)

    # ---------- 5) 人口属性 + 真实结果 ----------
    info = pd.read_csv(os.path.join(oulad_dir, "studentInfo.csv"))
    info = info[info["code_presentation"].isin(KEEP_PRES)].copy()
    base = info[key + ["gender", "region", "highest_education", "age_band",
                       "num_of_prev_attempts", "final_result"]].copy()

    df = (base.set_index(key)
          .join(beh).join(cov[["video_rate"]]).join(assign[["assign_rate"]])
          .join(quiz).join(fg[["final_grade"]]).reset_index())

    # ---------- 6) 折算与清洗（口径透明） ----------
    df["visit_count"] = df["visit_count"].fillna(0).astype(int)
    df["active_days"] = df["active_days"].fillna(0).astype(int)
    df["forum_posts"] = df["forum_posts"].fillna(0).astype(int)
    df["first_active_lag"] = df["first_active_lag"].fillna(84).astype(int)
    # 前 2/4 周无任何 VLE 记录的学生：语义上即"该窗口点击 0 次、活跃 0 天"，填 0。
    # （这类学生多为开学后很快退课者，填 0 符合真实行为含义，不引入虚构数据。）
    for c in ["early_w2_clicks", "early_w2_days", "early_w4_clicks", "early_w4_days"]:
        df[c] = df[c].fillna(0).astype(int)
    df["video_rate"] = df["video_rate"].fillna(0.0).clip(0, 1)
    df["assign_rate"] = df["assign_rate"].fillna(0.0).clip(0, 1)
    df["quiz_avg"] = df["quiz_avg"].fillna(0.0).clip(0, 100)
    df["final_grade"] = df["final_grade"].fillna(df["quiz_avg"]).clip(0, 100)
    # 活跃天数为 0 的退课学生：所有行为计 0，保证逻辑一致
    zero = df["active_days"] == 0
    df.loc[zero, ["visit_count", "forum_posts", "video_rate",
                  "assign_rate", "quiz_avg", "final_grade"]] = 0
    df["study_hours"] = (df["visit_count"] * CLICK_MIN / 60).round(2)
    df["avg_session_min"] = np.where(
        df["active_days"] > 0,
        df["visit_count"] * CLICK_MIN / df["active_days"].replace(0, np.nan), 0).round(2)
    df["avg_session_min"] = df["avg_session_min"].fillna(0)

    # ---------- 7) 映射为 analysis.py 既有契约 ----------
    df["course_id"] = (df["code_module"] + df["code_presentation"]).str.upper()
    df["course_name"] = df["course_id"]
    df["discipline"] = df["code_module"]
    df["region"] = df["region"].map(REGION_CN).fillna(df["region"])
    df["education"] = df["highest_education"].map(EDU_CN).fillna(df["highest_education"])
    df["gender"] = df["gender"].map({"M": "男", "F": "女"}).fillna(df["gender"])
    df["age"] = df["age_band"].map(AGE_MID).fillna(35.0)
    df["certificate"] = df["final_result"].isin(["Pass", "Distinction"]).astype(int)
    df["record_id"] = range(1, len(df) + 1)
    df["learner_id"] = "OU" + df["id_student"].astype(str)

    cols = ["record_id", "learner_id", "course_id", "course_name", "discipline",
            "region", "education", "gender", "age", "visit_count", "active_days",
            "study_hours", "avg_session_min", "forum_posts", "first_active_lag",
            "video_rate", "assign_rate", "quiz_avg", "final_grade", "certificate",
            "final_result", "early_w2_clicks", "early_w2_days",
            "early_w4_clicks", "early_w4_days"]
    out = df[cols].sort_values("record_id").reset_index(drop=True)
    out.to_csv(os.path.join(out_dir, "real_learners.csv"),
               index=False, encoding="utf-8-sig")

    meta = {
        "source": "OULAD (Open University Learning Analytics Dataset)",
        "source_tables": ["studentInfo.csv", "studentVle.csv", "vle.csv",
                          "studentAssessment.csv", "assessments.csv"],
        "presentations": list(KEEP_PRES),
        "rows": int(len(out)),
        "learners": int(df["id_student"].nunique()),
        "courses": int(out["course_id"].nunique()),
        "positive_rate": round(float(out["certificate"].mean()), 4),
        "label_definition": "certificate = final_result in {Pass, Distinction}，真实学业结果，非行为派生",
        "proxy_definitions": {
            "study_hours": f"累计点击 × {CLICK_MIN} 分钟/次 / 60，等效投入代理，非真实计时",
            "avg_session_min": f"累计点击 × {CLICK_MIN} / 活跃天数，等效代理",
            "video_rate": "访问过的 oucontent 资源数 / 该课程 oucontent 资源总数，内容覆盖代理",
            "forum_posts": "forumng 类资源点击数，互动行为代理",
        },
        "columns": cols,
    }
    with open(os.path.join(out_dir, "real_learners_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] 派生真实学习者主表:", os.path.join(out_dir, "real_learners.csv"))
    print("     记录", meta["rows"], "学习者", meta["learners"],
          "课程", meta["courses"], "真实通过率", meta["positive_rate"])
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oulad-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    build(args.oulad_dir, args.out_dir)


if __name__ == "__main__":
    main()
