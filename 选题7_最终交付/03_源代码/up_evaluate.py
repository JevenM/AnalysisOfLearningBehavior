# -*- coding: utf-8 -*-
"""up_evaluate.py —— 统一管线阶段[7]：评价层（AHP+模糊综合评判 / SD 系统动力学仿真）

读 up_learners.csv → AHP 定权（委托 evaluators 唯一实现）+ 模糊综合评判 →
学习质量评分合并回 up_learners.csv；SD 仿真四种平台策略 52 周演化。
产物：up_learning_quality_scores.csv / up_sd_simulation*.csv|png /
      up_eval_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import pandas as pd

from common import OUT_DIR, close_log, detect_device, device_banner, log, open_log, setup_cjk_font
from up_stages import ahp_fuzzy_scores, sd_simulation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="统一管线阶段[7]：评价层")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None, help="计算设备：默认读取环境变量 WB_DEVICE（由 run_all.py 贯通），可显式覆盖")
    ap.add_argument("--out-dir", default=OUT_DIR, help="产物目录")
    args = ap.parse_args(argv)
    detect_device(args.device or os.environ.get("WB_DEVICE", "auto"))
    open_log(args.out_dir)
    try:
        setup_cjk_font()
        device_banner()
        log("[up 7] 评价层（AHP+模糊评判 / SD 仿真）")
        learners_path = os.path.join(args.out_dir, "up_learners.csv")
        uni = pd.read_csv(learners_path, encoding="utf-8-sig")

        qs, ahp_info = ahp_fuzzy_scores(uni, args.out_dir)
        uni = uni.drop(columns=["dim_behavior", "dim_effect", "dim_interaction",
                                "fuzzy_best_level", "quality_score"],
                       errors="ignore") \
            .merge(qs, on="student_id", how="left")
        uni.to_csv(learners_path, index=False, encoding="utf-8-sig")

        sd = sd_simulation(args.out_dir)
        summary = {"ahp_weights": ahp_info["weights"], "ahp_cr": ahp_info["cr"],
                   "quality_mean": round(float(qs["quality_score"].mean()), 1),
                   "quality_levels": qs["fuzzy_best_level"].value_counts().to_dict(),
                   "sd_best": sd["best_strategy"],
                   "sd_final": sd["results"][sd["best_strategy"]]["final_engagement"],
                   "sd_results": sd["results"]}
        with open(os.path.join(args.out_dir, "up_eval_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log("[up 7] 完成")
        return 0
    except Exception as e:
        log(f"[up 7] 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    finally:
        close_log()


if __name__ == "__main__":
    sys.exit(main())
