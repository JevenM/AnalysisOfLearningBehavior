# -*- coding: utf-8 -*-
"""up_predict.py —— 统一管线阶段[6]：预测层（RF/GBDT/LogReg/早期预警/GM(1,1)）

读 up_learners.csv + up_clicks.csv → 学业表现预测、成功/风险预测、画像标签
预测、早期预警名单、GM(1,1) 日活跃趋势。
产物：up_rf_* / up_logreg_* / up_early_warning_students.csv /
      up_gm11_forecast.csv / up_metrics.json（更新）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import pandas as pd

from common import OUT_DIR, close_log, detect_device, device_banner, log, open_log, setup_cjk_font
from up_stages import gm11_forecast, performance_model_extended, \
    rf_performance_model, run_predictions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="统一管线阶段[6]：预测层")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None, help="计算设备：默认读取环境变量 WB_DEVICE（由 run_all.py 贯通），可显式覆盖")
    ap.add_argument("--out-dir", default=OUT_DIR, help="产物目录")
    args = ap.parse_args(argv)
    detect_device(args.device or os.environ.get("WB_DEVICE", "auto"))
    open_log(args.out_dir)
    try:
        setup_cjk_font()
        device_banner()
        log("[up 6] 预测层（RF/GBDT/LogReg/预警/GM(1,1)）")
        uni = pd.read_csv(os.path.join(args.out_dir, "up_learners.csv"),
                          encoding="utf-8-sig")
        clicks = pd.read_csv(os.path.join(args.out_dir, "up_clicks.csv"),
                             encoding="utf-8-sig")
        with open(os.path.join(args.out_dir, "up_metrics.json"), encoding="utf-8") as f:
            metrics = json.load(f)
        cols = metrics["cluster_cols"]

        rf_performance_model(uni, cols, args.out_dir)
        ext, ew = performance_model_extended(uni, cols, args.out_dir)
        metrics.update({k: (float(v) if isinstance(v, (int, float, bool)) or
                            v is None else v) if not isinstance(v, (dict, str, list))
                        else v for k, v in ext.items()})
        metrics.update(run_predictions(uni, cols, args.out_dir))
        gm = gm11_forecast(clicks, args.out_dir)
        valid = gm[gm["trend"] != "数据不足"]
        metrics["gm_trends"] = valid["trend"].value_counts().to_dict() \
            if len(valid) else {}

        with open(os.path.join(args.out_dir, "up_metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        log("[up 6] 完成")
        return 0
    except Exception as e:
        log(f"[up 6] 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    finally:
        close_log()


if __name__ == "__main__":
    sys.exit(main())
