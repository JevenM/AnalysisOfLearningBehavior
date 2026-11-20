# -*- coding: utf-8 -*-
"""run_all.py —— 选题7 一键运行入口（单命令复现整条分析链）。

用法（在交付包根目录执行，Python 3.9+）：
    python 03_源代码/run_all.py                 # 一键全链路：六源统一分析 → 图表 → 看板
    python 03_源代码/run_all.py --check         # 只体检：依赖、数据、结果文件、看板完整性
    python 03_源代码/run_all.py --skip-report   # 不重建 docx（Word 正占用时可用）

设计原则（对齐 GitHub 模板 main.py 的单入口 + 参数模式）：
  1. 默认完全离线可复现：所有派生数据随包，步骤只重算不重新下载；
  2. 每步独立子进程、失败即停并保留现场，打印 [i/n] 进度与耗时；
  3. --check 只读体检，不改任何文件。

分析口径（2026-11 重构）：全项目以"六类真实数据集合并总表"（up_ 统一管线，
66,644 名学习者）为主线，不单独拆分任何一个课程数据集（含 OULAD）；
内容级画像（时段/兴趣/短视频人口/天池标签）由 profile_modeling.py 补充；
多形态对比（慕课/微课/短视频/课堂）由 multimodal.py 提供跨场景证据。

数据依赖顺序（务必保持）：
  multimodal.py / profile_modeling.py（读 02_数据与数据集/ 随包派生表）
  → 统一管线 up_ 8 阶段（up_build_unified → up_cluster → up_mine →
    up_associate → up_predict → up_evaluate → up_decide → up_report，产物在
    04_运行结果/，聚类经 common.ClusterEngine 支持 --device gpu 加速）
  → 三个看板
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
SRC = HERE
DATA = os.path.join(PKG, "02_数据与数据集")
OUT = os.path.join(PKG, "04_运行结果")
FIG = os.path.join(PKG, "05_图表与看板", "figures")
DASH = os.path.join(PKG, "05_图表与看板")

PY = sys.executable


def _py(script: str, *args: str, cwd: str = SRC) -> int:
    cmd = [PY, os.path.join(SRC, script), *args]
    print(f"    $ {os.path.basename(script)} {' '.join(args)}".rstrip())
    return subprocess.run(cmd, cwd=cwd).returncode


def _up(name: str, *args: str) -> int:
    """统一管线阶段脚本：02_数据与数据集/raw/ 为输入，04_运行结果/ 落盘。"""
    return _py(name, *args)


# 报告插图清单（OULAD 单源图已删：fig04—fig10、fig15—fig19）
EXPECTED_FIGS = ["fig01", "fig02", "fig03", "fig11", "fig12", "fig13", "fig14",
                 "fig20", "fig21", "fig22"]


# ---------------------------------------------------------------- 体检
def check() -> int:
    print("=" * 68)
    print("体检模式：只读取、不修改任何文件")
    print("=" * 68)
    ok = True

    print("\n[1] Python 依赖")
    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib"):
        try:
            m = __import__(mod)
            print(f"    OK  {mod:<12} {getattr(m, '__version__', '')}")
        except Exception as e:
            ok = False
            print(f"    缺失 {mod:<12} {e}  ->  python -m pip install -r 03_源代码/requirements.txt")

    print("\n[2] 随包派生数据（离线复现所需）")
    for f in ("real_learners.csv", "multimodal_all.csv", "sources.json",
              "shortvideo_profile.csv", "tianchi_profile.csv", "shortvideo_cat_watch.csv"):
        p = os.path.join(DATA, f)
        good = os.path.isfile(p) and os.path.getsize(p) > 0
        ok &= good
        print(f"    {'OK ' if good else '缺失'} {f}")

    print("\n[3] 结果文件（JSON 可解析；统一管线产物为主线，必须齐全）")
    for f in ("multimodal_results.json", "profile_results.json"):
        p = os.path.join(OUT, f)
        try:
            json.load(open(p, encoding="utf-8"))
            print(f"    OK  {f}")
        except Exception as e:
            ok = False
            print(f"    异常 {f}: {e}")
    for f in ("up_learners.csv", "up_unified.csv", "up_metrics.json", "up_mining_summary.json",
              "up_association_summary.json", "up_eval_summary.json", "up_report.md"):
        p = os.path.join(OUT, f)
        good = os.path.isfile(p) and os.path.getsize(p) > 0
        ok &= good
        print(f"    {'OK ' if good else '缺失'} {f}")

    print("\n[4] 图表与看板")
    figs = [f for f in os.listdir(FIG)] if os.path.isdir(FIG) else []
    miss = [k for k in EXPECTED_FIGS if not any(x.startswith(k) for x in figs)]
    up_figs = [f for f in figs if f.startswith("up_")]
    ok &= not miss
    print(f"    报告插图 {len(figs) - len(up_figs)} 张，缺失编号: {miss if miss else '无'}")
    print(f"    统一管线图 {len(up_figs)} 张")
    for f in ("dashboard.html", "persona_dashboard.html", "profile_dashboard.html", "conclusion.md"):
        p = os.path.join(DASH, f)
        good = os.path.isfile(p) and os.path.getsize(p) > 10_000
        ok &= good
        print(f"    {'OK ' if good else '缺失'} {f}")

    print("\n" + "=" * 68)
    print("体检结论：" + ("全部通过，可执行一键运行" if ok else "存在缺失项，请按上方提示补齐后再运行"))
    print("=" * 68)
    return 0 if ok else 2


# ---------------------------------------------------------------- 一键运行
def run_all(args) -> int:
    t0 = time.time()
    print("\n" + "=" * 68)
    print("选题7 一键运行 · 单命令复现分析链（六源统一口径）")
    print(f"开始时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 68)

    steps = []  # (名称, 函数)

    # 1) 多形态对比与内容级画像（读 02_数据与数据集/ 随包派生表）
    steps.append(("四形态跨场景测评 → multimodal_results.json", lambda: _py("multimodal.py")))
    steps.append(("内容级画像（时段/兴趣/人口/天池）→ profile_results.json",
                  lambda: _py("profile_modeling.py")))

    # 2) 统一管线（主线）：六类真实数据集 → 聚类画像 → 挖掘/关联/预测/评价/决策/报告
    #    聚类走 ClusterEngine（--device gpu 时 GPU 加速）
    steps.append(("统一管线[1/8] 六类真实数据集融合 → up_unified.csv",
                  lambda: _up("up_build_unified.py")))
    steps.append(("统一管线[2/8] GPU 聚类画像 → up_learners.csv",
                  lambda: _up("up_cluster.py")))
    steps.append(("统一管线[3/8] 行为模式挖掘 SNA/LSA/Apriori → up_mining_summary.json",
                  lambda: _up("up_mine.py")))
    steps.append(("统一管线[4/8] 相关性/异质性/投入分层 → up_association_summary.json",
                  lambda: _up("up_associate.py")))
    steps.append(("统一管线[5/8] 预测与早期预警/GM(1,1) → up_metrics.json",
                  lambda: _up("up_predict.py")))
    steps.append(("统一管线[6/8] AHP+模糊评判/SD 仿真 → up_eval_summary.json",
                  lambda: _up("up_evaluate.py")))
    steps.append(("统一管线[7/8] 干预队列/推荐/优化建议 → up_queues.csv",
                  lambda: _up("up_decide.py")))
    steps.append(("统一管线[8/8] 汇总报告 → up_report.md",
                  lambda: _up("up_report.py")))

    # 3) 看板
    steps.append(("主看板（六源统一叙事）→ dashboard.html",
                  lambda: _py(os.path.join("..", "05_图表与看板", "build_dashboard.py"))))
    steps.append(("形态画像看板 → persona_dashboard.html",
                  lambda: _py(os.path.join("..", "05_图表与看板", "build_persona_dashboard.py"))))
    steps.append(("内容画像看板 → profile_dashboard.html",
                  lambda: _py(os.path.join("..", "05_图表与看板", "build_profile_dashboard.py"))))

    n = len(steps)
    for i, (name, fn) in enumerate(steps, 1):
        print(f"\n[{i}/{n}] {name}")
        s = time.time()
        rc = fn()
        if rc != 0:
            print(f"[失败] 步骤 {i} 退出码={rc}，已停止。修复后可重跑（已完成步骤会幂等覆盖）。")
            return rc
        print(f"[完成] 耗时 {time.time() - s:.1f}s")

    print("\n" + "=" * 68)
    print(f"一键运行完成：{n} 个步骤，总耗时 {time.time() - t0:.1f}s")
    print("=" * 68)

    # 4) 自动体检
    return check()


def main() -> int:
    ap = argparse.ArgumentParser(description="选题7 一键运行入口（六源统一口径）")
    ap.add_argument("--check", action="store_true", help="只体检，不修改文件")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                    help="计算设备：GPU 加速统一管线聚类（经环境变量 WB_DEVICE 贯通子进程）")
    ap.add_argument("--skip-report", action="store_true", help="预留：跳过可选的下游报告步骤")
    a = ap.parse_args()
    os.environ["WB_DEVICE"] = a.device          # 设备选择贯通全部子进程
    return check() if a.check else run_all(a)


if __name__ == "__main__":
    sys.exit(main())
