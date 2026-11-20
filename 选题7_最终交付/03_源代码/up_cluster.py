# -*- coding: utf-8 -*-
"""up_cluster.py —— 统一管线阶段[3]：分析层（PCA 命名 + GPU 聚类 + 画像命名）

读 up_unified.csv → 命名主成分 → 肘部+轮廓融合选 K → ClusterEngine 聚类
（支持 GPU 加速）→ 画像命名与可视化。
产物：up_learners.csv / up_metrics.json / PCA/t-SNE/UMAP/雷达/热力图
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import pandas as pd

from common import OUT_DIR, close_log, detect_device, device_banner, log, open_log, setup_cjk_font
from up_stages import cluster_personas, label_personas, pca_named_components, \
    select_cluster_columns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="统一管线阶段[3]：分析层")
    ap.add_argument("--k", type=int, default=0,
                    help="聚类数（0 = 自动：肘部拐点 + 轮廓系数综合选择）")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None, help="计算设备：默认读取环境变量 WB_DEVICE（由 run_all.py 贯通），可显式覆盖")
    ap.add_argument("--out-dir", default=OUT_DIR, help="产物目录")
    args = ap.parse_args(argv)
    detect_device(args.device or os.environ.get("WB_DEVICE", "auto"))
    open_log(args.out_dir)
    try:
        setup_cjk_font()
        device_banner()
        log("[up 3] 分析层（PCA + 聚类画像）")
        uni = pd.read_csv(os.path.join(args.out_dir, "up_unified.csv"),
                          encoding="utf-8-sig")
        cols = select_cluster_columns(uni)
        pca_named_components(uni, cols, args.out_dir)
        uni, engine, scaler, cols, metrics = cluster_personas(
            uni, cols, args.k, args.out_dir)
        uni = label_personas(uni, cols)
        log(f"  画像分布: {uni['persona'].value_counts().to_dict()}")
        uni["cluster_cols"] = json.dumps(cols)   # 供下游阶段复用聚类特征列
        uni.to_csv(os.path.join(args.out_dir, "up_learners.csv"), index=False,
                   encoding="utf-8-sig")
        metrics["cluster_cols"] = cols
        with open(os.path.join(args.out_dir, "up_metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        log("[up 3] 完成")
        return 0
    except Exception as e:
        log(f"[up 3] 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    finally:
        close_log()


if __name__ == "__main__":
    sys.exit(main())
