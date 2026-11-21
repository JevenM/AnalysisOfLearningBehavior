# -*- coding: utf-8 -*-
"""build_test_upload.py —— 生成建模控制台的测试上传文件（真实未使用数据）

对应课程要求（2）"教师现场更换输入数据重新运行"。默认六源合并总表只使用了
HarvardX-MITx Person Course 数据集 323,419 名真实注册用户中被随机抽样的 5 万人
（见 up_adapters.py load_harvardx_mitx 的 max_users=50000 上限）；本脚本从
**未参与建模的留出用户**（且真实行为事件数>0）中按固定种子抽取 8,000 人，
用与正式管线完全相同的聚合口径构造与 up_unified.csv 同 schema 的测试文件，
供答辩现场上传到控制台沙箱重跑。

数据性质：真实公开数据（HarvardX-MITx Person Course，edX 2012-2013，CC-BY 4.0），
不含任何合成样本。生成文件：02_数据与数据集/测试上传数据/test_upload_harvardx_holdout.csv

运行：python 03_源代码/build_test_upload.py [--n 8000]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT_DIR as RUN, SEED, log           # noqa: E402
from up_adapters import RAW_DIR                        # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(RAW_DIR), "测试上传数据")
OUT_NAME = "test_upload_harvardx_holdout.csv"
EDX_SOURCE = "real:harvardx_mitx_person_course"


def load_raw_harvardx() -> pd.DataFrame:
    """读取 HarvardX 原始数据（与 load_harvardx_mitx 同一文件与列）。"""
    base = os.path.join(RAW_DIR, "harvardx")
    path = os.path.join(base, "hmx_cleaned.csv")
    uid_col = "user_id"
    if not os.path.exists(path):
        path = os.path.join(base, "HMXPC13_DI_v2_5-14-14.csv")
        uid_col = "userid_DI"
    usecols = [uid_col, "course_id", "nevents", "ndays_act", "nplay_video",
               "nforum_posts", "nchapters", "grade", "certified", "viewed",
               "explored", "gender", "YoB", "education"]
    df = pd.read_csv(path, usecols=usecols).rename(columns={uid_col: "user_id"})
    return df

def main() -> int:
    ap = argparse.ArgumentParser(description="生成控制台测试上传文件（未参与建模的留出数据）")
    ap.add_argument("--n", type=int, default=8000, help="抽取人数（默认 8000）")
    a = ap.parse_args()

    # 1) 正式总表中已使用的 HarvardX 用户（EDX-<userid> 去前缀）
    uni = pd.read_csv(os.path.join(RUN, "up_unified.csv"),
                      usecols=["student_id", "source"])
    used = set(uni.loc[uni["source"] == EDX_SOURCE, "student_id"]
               .astype(str).str[4:])

    # 2) 原始数据的未入模、有真实行为（nevents>0）的留出用户
    df = load_raw_harvardx()
    eng = df.groupby("user_id")["nevents"].sum()
    unused_eng = [u for u in df["user_id"].unique()
                  if str(u) not in used and eng[u] > 0]
    log(f"未入模且有真实行为的留出用户: {len(unused_eng)} 人（已入模 {len(used)} 人）")
    if len(unused_eng) < a.n:
        raise SystemExit(f"留出用户不足 {a.n} 人，请调小 --n")
    pick = np.random.default_rng(SEED).choice(unused_eng, a.n, replace=False)
    sub = df[df["user_id"].isin(pick)]

    # 3) 聚合口径与 load_harvardx_mitx 完全一致（逐用户跨课程聚合）
    g = sub.groupby("user_id")
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
        "completion_rate": g[["viewed", "explored"]].max().mean(axis=1).values.round(3),
        "success_status": np.where(g["certified"].max().values == 1,
                                   "certified", "none"),
        "gender": g["gender"].first().map({"m": "男", "f": "女",
                                           "M": "男", "F": "女"}).values,
        "age_band": pd.cut(pd.to_numeric(g["YoB"].first(), errors="coerce"),
                           [0, 1975, 1985, 1995, 2004],
                           labels=["40+", "31-40", "26-30", "18-25"]).astype(str).values,
        "education": g["education"].first().map(edu_map).values,
        "source": EDX_SOURCE,
    })

    # 4) 对齐统一表列序（缺失列保持 NaN，管线按 fill≥0.6 规则自动选列），
    #    并按 build_unified_dataset 同口径派生 success 标签
    uni_cols = list(pd.read_csv(os.path.join(RUN, "up_unified.csv"),
                                nrows=0).columns)
    out = feat.reindex(columns=uni_cols)
    out["success"] = out["success_status"].astype(str).str.lower().isin(
        ["successful", "success", "certified", "pass", "1", "true"]).astype(int)
    out = out.drop_duplicates(subset="student_id")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_NAME)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"测试上传文件已生成: {out_path}")
    log(f"  规模 {len(out)} 行 × {len(out.columns)} 列; 认证率 "
        f"{out['success'].mean() * 100:.1f}%; 平均点击 "
        f"{out['total_clicks'].mean():.0f}; 活跃天数中位数 "
        f"{out['active_days'].median():.0f}")
    log("  用法见 06_支撑文档/运行说明.md『现场测试：更换输入数据完整流程』")
    return 0


if __name__ == "__main__":
    sys.exit(main())
