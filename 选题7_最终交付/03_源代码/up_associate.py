# -*- coding: utf-8 -*-
"""up_associate.py —— 统一管线阶段[5]：关联层（相关性 / 异质性 / 投入度分层）

读 up_learners.csv → 皮尔逊相关（含 p 值）、群体异质性分析、投入三分层。
产物：up_correlation_results.csv / up_heterogeneity_analysis.csv /
      up_engagement_tier_analysis.csv / up_association_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import pandas as pd

from common import OUT_DIR, close_log, detect_device, device_banner, log, open_log, setup_cjk_font
from up_stages import correlation_analysis, engagement_tier_analysis, \
    heterogeneity_analysis


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="统一管线阶段[5]：关联层")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None, help="计算设备：默认读取环境变量 WB_DEVICE（由 run_all.py 贯通），可显式覆盖")
    ap.add_argument("--out-dir", default=OUT_DIR, help="产物目录")
    args = ap.parse_args(argv)
    detect_device(args.device or os.environ.get("WB_DEVICE", "auto"))
    open_log(args.out_dir)
    try:
        setup_cjk_font()
        device_banner()
        log("[up 5] 关联层（相关/异质性/投入分层）")
        learners_path = os.path.join(args.out_dir, "up_learners.csv")
        uni = pd.read_csv(learners_path, encoding="utf-8-sig")

        corr, corr_top = correlation_analysis(uni, args.out_dir)
        heterogeneity_analysis(uni, args.out_dir)
        uni, tier_tbl = engagement_tier_analysis(uni, args.out_dir)
        uni.to_csv(learners_path, index=False, encoding="utf-8-sig")

        tier_lift = None
        if len(tier_tbl):
            tier_lift = (f"完成率 +{tier_tbl['完成率提升幅度'].iloc[-1]}%, "
                         f"成绩 +{tier_tbl['成绩提升幅度'].iloc[-1]} 分")
        summary = {"corr_top": corr_top.to_dict("records"),
                   "tier_lift": tier_lift}
        with open(os.path.join(args.out_dir, "up_association_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log("[up 5] 完成")
        return 0
    except Exception as e:
        log(f"[up 5] 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    finally:
        close_log()


if __name__ == "__main__":
    sys.exit(main())
