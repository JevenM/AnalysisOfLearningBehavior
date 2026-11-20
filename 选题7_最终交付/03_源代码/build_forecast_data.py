# -*- coding: utf-8 -*-
"""
build_forecast_data.py —— 从 OULAD 原始日志派生「周活跃序列」与「早期预警面板」。

原始 OULAD（Open University Learning Analytics Dataset）不随交付包分发；
本脚本把公开源表中的信息压缩为两张可复现的小表，供 forecast.py 离线实跑。

用法：
    python build_forecast_data.py --oulad-dir <含 studentVle.csv / studentInfo.csv 的目录> \
                                  --out-dir <交付包/02_数据与数据集>

输出：
    weekly_activity.csv   周次 × 学期的人均点击与活跃率（系统预测章的时间序列）
    early_warning.csv     学生级 2/4/6 周累计行为 + 学业标签（早期预警时点识别）

说明：仅取 2013J / 2014J 两个 12 周春季学期，保证周次口径一致；
date>=0 且 week<12，排除开课前访问与超出学期窗口的记录。
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

KEEP_PRES = ("2013J", "2014J")
WEEKS = 12
CUTOFFS = {"w2": 14, "w4": 28, "w6": 42}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oulad-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    vle_path = os.path.join(args.oulad_dir, "studentVle.csv")
    info_path = os.path.join(args.oulad_dir, "studentInfo.csv")

    cols = ["code_module", "code_presentation", "id_student", "date", "sum_click"]
    vle = pd.read_csv(vle_path, usecols=cols)
    vle = vle[vle["code_presentation"].isin(KEEP_PRES)]
    vle = vle[(vle["date"] >= 0) & (vle["date"] < WEEKS * 7)].copy()
    vle["week"] = vle["date"] // 7
    print("filtered VLE rows:", len(vle))

    info = pd.read_csv(info_path)
    info = info[info["code_presentation"].isin(KEEP_PRES)].copy()
    enrolled = info.groupby("code_presentation")["id_student"].nunique()

    # ---- 1) 周活跃序列：按学期统计，再合并为总体序列 ----
    weekly = (vle.groupby(["code_presentation", "week"])
                .agg(clicks=("sum_click", "sum"), active=("id_student", "nunique"))
                .reset_index())
    weekly["enrolled"] = weekly["code_presentation"].map(enrolled)
    weekly["clicks_per_active"] = (weekly["clicks"] / weekly["active"]).round(4)
    weekly["active_rate"] = (weekly["active"] / weekly["enrolled"]).round(4)

    overall = (weekly.groupby("week")
               .agg(clicks=("clicks", "sum"), active=("active", "sum"),
                    enrolled=("enrolled", "max"))
               .reset_index())
    overall["code_presentation"] = "ALL"
    overall["clicks_per_active"] = (overall["clicks"] / overall["active"]).round(4)
    overall["active_rate"] = (overall["active"] / overall["enrolled"]).round(4)
    out_w = pd.concat([weekly, overall], ignore_index=True)
    out_w = out_w[out_w["week"] < WEEKS]
    out_w = out_w[["code_presentation", "week", "clicks", "active", "enrolled",
                   "clicks_per_active", "active_rate"]]
    out_w.to_csv(os.path.join(args.out_dir, "weekly_activity.csv"),
                 index=False, encoding="utf-8-sig")

    # ---- 2) 早期预警面板：截至第 2/4/6 周的累计点击与活跃天数 ----
    daily = (vle.groupby(["id_student", "date"], as_index=False)["sum_click"]
                .sum())
    panels = {}
    for name, cut in CUTOFFS.items():
        sub = daily[daily["date"] < cut]
        panels[name] = (sub.groupby("id_student")
                          .agg(**{f"{name}_clicks": ("sum_click", "sum"),
                                  f"{name}_days": ("date", "nunique")}))
    panel = info[["code_module", "code_presentation", "id_student", "gender",
                  "region", "highest_education", "age_band",
                  "num_of_prev_attempts", "studied_credits", "final_result"]].copy()
    for d in panels.values():
        panel = panel.join(d, on="id_student")
    for name in CUTOFFS:
        for col in (f"{name}_clicks", f"{name}_days"):
            panel[col] = panel[col].fillna(0)
    panel["label"] = panel["final_result"].isin(["Pass", "Distinction"]).astype(int)
    panel = panel.rename(columns={
        "code_module": "module", "code_presentation": "presentation",
        "gender": "gender", "region": "region",
        "highest_education": "education", "age_band": "age_band",
        "num_of_prev_attempts": "prev_attempts", "studied_credits": "credits",
    })
    panel.to_csv(os.path.join(args.out_dir, "early_warning.csv"),
                 index=False, encoding="utf-8-sig")

    print("weekly rows:", len(out_w), "warning rows:", len(panel),
          "positive rate:", round(panel["label"].mean(), 4))


if __name__ == "__main__":
    main()
