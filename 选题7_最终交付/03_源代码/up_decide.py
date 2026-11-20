# -*- coding: utf-8 -*-
"""up_decide.py —— 统一管线阶段[8]：决策层（规则引擎干预队列 / 内容推荐 / 优化建议）

读 up_learners.csv 及上游产物 → 干预队列（含证据链与教师复核标记）→
分画像内容推荐 → 全链路优化建议。
产物：up_queues.csv / up_persona_recommendations.csv /
      up_optimization_suggestions.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import pandas as pd

from common import OUT_DIR, close_log, detect_device, device_banner, log, open_log, setup_cjk_font
from up_stages import generate_optimization_suggestions, generate_recommendations, \
    rule_queues


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="统一管线阶段[8]：决策层")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None, help="计算设备：默认读取环境变量 WB_DEVICE（由 run_all.py 贯通），可显式覆盖")
    ap.add_argument("--out-dir", default=OUT_DIR, help="产物目录")
    args = ap.parse_args(argv)
    detect_device(args.device or os.environ.get("WB_DEVICE", "auto"))
    open_log(args.out_dir)
    try:
        setup_cjk_font()
        device_banner()
        log("[up 8] 决策层（队列/推荐/建议）")
        uni = pd.read_csv(os.path.join(args.out_dir, "up_learners.csv"),
                          encoding="utf-8-sig")
        recs = generate_recommendations(uni)
        recs.to_csv(os.path.join(args.out_dir, "up_persona_recommendations.csv"),
                    index=False, encoding="utf-8-sig")
        queues = rule_queues(uni)
        queues.to_csv(os.path.join(args.out_dir, "up_queues.csv"), index=False,
                      encoding="utf-8-sig")

        # 汇集上游产物供建议生成
        scene_tbl = pd.DataFrame()
        scene_path = os.path.join(args.out_dir, "up_scene_comparison.csv")
        if os.path.exists(scene_path):
            scene_tbl = pd.read_csv(scene_path, index_col=0)
        corr_path = os.path.join(args.out_dir, "up_correlation_results.csv")
        corr_top = pd.DataFrame()
        if os.path.exists(corr_path):
            corr = pd.read_csv(corr_path)
            corr_top = corr[corr["target"] == "final_grade"] \
                .sort_values("pearson_r", key=abs, ascending=False).head(5)
        ew_path = os.path.join(args.out_dir, "up_early_warning_students.csv")
        early_warning = pd.read_csv(ew_path) if os.path.exists(ew_path) else \
            pd.DataFrame(columns=["risk_level"])
        sd_info = {"best_strategy": "-", "results": {}}
        eval_path = os.path.join(args.out_dir, "up_eval_summary.json")
        if os.path.exists(eval_path):
            with open(eval_path, encoding="utf-8") as f:
                ev = json.load(f)
            sd_info = {"best_strategy": ev.get("sd_best", "-"),
                       "results": ev.get("sd_results", {})}

        sug = generate_optimization_suggestions(uni, recs, scene_tbl, corr_top,
                                                sd_info, early_warning, args.out_dir)
        log(f"  干预队列分布: {queues['queue'].value_counts().to_dict()}")
        log("[up 8] 完成")
        return 0
    except Exception as e:
        log(f"[up 8] 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    finally:
        close_log()


if __name__ == "__main__":
    sys.exit(main())
