# -*- coding: utf-8 -*-
"""up_report.py —— 统一管线阶段[9]：输出层（汇总图文报告）

读全部上游产物 → 汇总 up_report.md。
产物：up_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import pandas as pd

from common import OUT_DIR, close_log, detect_device, device_banner, log, open_log, setup_cjk_font
from up_stages import write_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="统一管线阶段[9]：输出层")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None, help="计算设备：默认读取环境变量 WB_DEVICE（由 run_all.py 贯通），可显式覆盖")
    ap.add_argument("--out-dir", default=OUT_DIR, help="产物目录")
    args = ap.parse_args(argv)
    detect_device(args.device or os.environ.get("WB_DEVICE", "auto"))
    open_log(args.out_dir)
    try:
        setup_cjk_font()
        device_banner()
        log("[up 9] 输出层（汇总报告）")
        out = args.out_dir
        uni = pd.read_csv(os.path.join(out, "up_learners.csv"), encoding="utf-8-sig")
        queues = pd.read_csv(os.path.join(out, "up_queues.csv"), encoding="utf-8-sig")
        recs = pd.read_csv(os.path.join(out, "up_persona_recommendations.csv"),
                           encoding="utf-8-sig")
        with open(os.path.join(out, "up_metrics.json"), encoding="utf-8") as f:
            metrics = json.load(f)

        extras = {}
        mining_path = os.path.join(out, "up_mining_summary.json")
        if os.path.exists(mining_path):
            with open(mining_path, encoding="utf-8") as f:
                extras.update(json.load(f))
        assoc_path = os.path.join(out, "up_association_summary.json")
        if os.path.exists(assoc_path):
            with open(assoc_path, encoding="utf-8") as f:
                extras.update(json.load(f))
        eval_path = os.path.join(out, "up_eval_summary.json")
        if os.path.exists(eval_path):
            with open(eval_path, encoding="utf-8") as f:
                ev = json.load(f)
            extras["ahp_weights"] = ev.get("ahp_weights")
            extras["ahp_cr"] = ev.get("ahp_cr")
            extras["quality_mean"] = ev.get("quality_mean")
            extras["quality_levels"] = ev.get("quality_levels")
            extras["sd_best"] = ev.get("sd_best")
            extras["sd_final"] = ev.get("sd_final")
        sug_path = os.path.join(out, "up_optimization_suggestions.csv")
        if os.path.exists(sug_path):
            extras["n_suggestions"] = len(pd.read_csv(sug_path))
        ew_path = os.path.join(out, "up_early_warning_students.csv")
        if os.path.exists(ew_path):
            ew = pd.read_csv(ew_path)
            extras["early_warning_count"] = int((ew["risk_level"] == "高").sum())
        metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}

        report = write_report(uni, queues, recs, metrics, out, extras)
        log(f"  报告已生成: {os.path.join(out, 'up_report.md')} ({len(report)} 字符)")
        log("[up 9] 完成")
        return 0
    except Exception as e:
        log(f"[up 9] 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    finally:
        close_log()


if __name__ == "__main__":
    sys.exit(main())
