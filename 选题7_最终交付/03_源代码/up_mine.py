# -*- coding: utf-8 -*-
"""up_mine.py —— 统一管线阶段[4]：挖掘层（SNA / LSA / 时序模式 / Apriori / 场景对比 / 基础统计）

读 up_clicks.csv + up_learners.csv → 行为模式挖掘，并把逐学生模式列
（lsa_pattern / time_pattern / video_completion_rate / preferred_scene）合并回
up_learners.csv。
产物：up_sna_* / up_lsa_* / up_temporal_* / up_apriori_* / up_scene_* /
      up_demographics_* / up_behavior_baseline.csv / up_mining_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

from common import OUT_DIR, close_log, detect_device, device_banner, log, open_log, setup_cjk_font
from up_stages import apriori_path_mining, basic_stats_analysis, lsa_analysis, \
    scene_comparison, sna_analysis, temporal_analysis


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="统一管线阶段[4]：挖掘层")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None, help="计算设备：默认读取环境变量 WB_DEVICE（由 run_all.py 贯通），可显式覆盖")
    ap.add_argument("--out-dir", default=OUT_DIR, help="产物目录")
    args = ap.parse_args(argv)
    detect_device(args.device or os.environ.get("WB_DEVICE", "auto"))
    open_log(args.out_dir)
    try:
        setup_cjk_font()
        device_banner()
        log("[up 4] 挖掘层（SNA/LSA/时序/Apriori/场景）")
        clicks = pd.read_csv(os.path.join(args.out_dir, "up_clicks.csv"),
                             encoding="utf-8-sig")
        learners_path = os.path.join(args.out_dir, "up_learners.csv")
        uni = pd.read_csv(learners_path, encoding="utf-8-sig")

        # SNA
        sna_analysis(clicks, args.out_dir)
        rm = pd.read_csv(os.path.join(args.out_dir, "up_sna_resource_metrics.csv"))
        sna_top = rm.head(5)["resource_id"].tolist()

        # LSA
        anoms, tr, z_df = lsa_analysis(clicks, args.out_dir)
        lsa_patterns = anoms["lsa_pattern"].value_counts().to_dict()
        flat = z_df.stack().sort_values(key=abs, ascending=False)
        lsa_top = [{"transition": f"{a}→{b}", "z": float(z)}
                   for (a, b), z in flat.head(5).items()]

        # 时序模式
        pat_df, time_patterns = temporal_analysis(clicks, args.out_dir)

        # Apriori
        _, apriori_summary = apriori_path_mining(clicks, args.out_dir)

        # 场景对比
        tbl, per_stu, scene_dict = scene_comparison(clicks, args.out_dir)

        # 基础统计（人口属性 × 行为）
        bs = basic_stats_analysis(uni, args.out_dir)

        # 合并逐学生模式列回 up_learners.csv
        for m, name in [(anoms, "lsa_pattern"), (pat_df, "time_pattern")]:
            key = [c for c in m.columns if c != name]
            uni = uni.drop(columns=[name], errors="ignore") \
                .merge(m[["student_id", name]], on="student_id", how="left")
            uni[name] = uni[name].fillna("正常" if name != "time_pattern" else "常规型学习")
        if not per_stu.empty:
            uni = uni.drop(columns=["video_completion_rate", "preferred_scene"],
                           errors="ignore") \
                .merge(per_stu[["student_id", "video_completion_rate",
                                "preferred_scene"]], on="student_id", how="left")
        uni.to_csv(learners_path, index=False, encoding="utf-8-sig")

        summary = {
            "sna_top_resources": sna_top,
            "lsa_patterns": lsa_patterns,
            "lsa_top_transitions": lsa_top,
            "time_patterns": time_patterns,
            "apriori_top": apriori_summary.get("top_rules", []),
            "short_only_pct": apriori_summary.get("short_only_pct"),
            "scene": scene_dict,
            "core_groups": bs["core_groups"],
        }
        with open(os.path.join(args.out_dir, "up_mining_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log("[up 4] 完成")
        return 0
    except Exception as e:
        log(f"[up 4] 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    finally:
        close_log()


if __name__ == "__main__":
    sys.exit(main())
