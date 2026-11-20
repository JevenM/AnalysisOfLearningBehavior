# -*- coding: utf-8 -*-
"""up_build_unified.py —— 统一管线阶段[1/2]：数据层 + 特征层

融合 raw/ 下六类真实公开数据集 → 统一特征表 + 真实事件流 + EDA 摘要。
产物：up_unified.csv / up_clicks.csv / up_eda_summary.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import pandas as pd

from common import OUT_DIR, close_log, detect_device, device_banner, log, open_log, setup_cjk_font
from up_adapters import build_clickstream, build_unified_dataset
from up_stages import eda_summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="统一管线阶段[1/2]：数据层+特征层")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None,
                    help="计算设备：默认读取环境变量 WB_DEVICE（由 run_all.py 贯通），可显式覆盖")
    ap.add_argument("--out-dir", default=OUT_DIR, help="产物目录")
    args = ap.parse_args(argv)
    detect_device(args.device or os.environ.get("WB_DEVICE", "auto"))
    os.makedirs(args.out_dir, exist_ok=True)
    open_log(args.out_dir)
    try:
        setup_cjk_font()
        device_banner()
        log("[up 1/2] 数据层+特征层")
        uni = build_unified_dataset()
        clicks = build_clickstream()
        uni_path = os.path.join(args.out_dir, "up_unified.csv")
        uni.to_csv(uni_path, index=False, encoding="utf-8-sig")
        clicks_path = os.path.join(args.out_dir, "up_clicks.csv")
        clicks.to_csv(clicks_path, index=False, encoding="utf-8-sig")
        log(f"  产物: {uni_path} ({uni.shape}), {clicks_path} ({clicks.shape})")
        eda_summary(uni, args.out_dir)
        log("[up 1/2] 完成")
        return 0
    except SystemExit as e:
        log(f"[up 1/2] 数据缺失: {e}")
        return 1
    except Exception as e:
        log(f"[up 1/2] 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    finally:
        close_log()


if __name__ == "__main__":
    sys.exit(main())
