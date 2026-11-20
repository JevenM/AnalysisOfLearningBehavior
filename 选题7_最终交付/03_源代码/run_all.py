# -*- coding: utf-8 -*-
"""run_all.py —— 选题7 一键运行入口（单命令复现整条分析链）。

用法（在交付包根目录执行，Python 3.9+）：
    python 03_源代码/run_all.py                 # 一键全链路：重算所有结果 → 图表 → 看板 → 报告
    python 03_源代码/run_all.py --check         # 只体检：依赖、数据、结果文件、看板完整性
    python 03_源代码/run_all.py --web           # 全链路完成后启动交互式 Web 平台（默认 8765 端口）
    python 03_源代码/run_all.py --skip-report   # 不重建 docx（Word 正占用时可用）
    python 03_源代码/run_all.py --from-raw --oulad-dir <dir>   # 含"从 OULAD 原始表派生主表"的完整链
    python 03_源代码/run_all.py --port 9000      # 指定 Web 端口

设计原则（对齐 GitHub 模板 main.py 的单入口 + 参数模式）：
  1. 默认完全离线可复现：所有派生数据随包，步骤只重算不重新下载；
  2. 每步独立子进程、失败即停并保留现场，打印 [i/n] 进度与耗时；
  3. --from-raw 才走 443MB OULAD 原始表派生（需 --oulad-dir），默认跳过；
  4. --check 只读体检，不改任何文件。

数据依赖顺序（务必保持）：
  multimodal.py → app.py --batch（把多形态结果挂进 results.json）
  → forecast.py / fuzzy_eval.py / profile_modeling.py（各自读已落盘派生表）
  → demo_runs.py
  → 统一管线 up_ 8 阶段（合并新增；up_build_unified → up_cluster → up_mine →
    up_associate → up_predict → up_evaluate → up_decide → up_report，产物在
    04_运行结果/，聚类经 common.ClusterEngine 支持 --device gpu 加速，可 --skip-unified 跳过）
  → figures.py → 三个看板 → build_final_report.py
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
    """统一管线阶段脚本（合并新增）：02_数据与数据集/raw/ 为输入，04_运行结果/ 落盘。"""
    return _py(name, *args)


# ---------------------------------------------------------------- 体检
def check() -> int:
    print("=" * 68)
    print("体检模式：只读取、不修改任何文件")
    print("=" * 68)
    ok = True

    print("\n[1] Python 依赖")
    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "docx"):
        try:
            m = __import__(mod)
            print(f"    OK  {mod:<12} {getattr(m, '__version__', '')}")
        except Exception as e:
            ok = False
            print(f"    缺失 {mod:<12} {e}  ->  python -m pip install -r 03_源代码/requirements.txt")

    print("\n[2] 随包派生数据（离线复现所需）")
    for f in ("real_learners.csv", "weekly_activity.csv", "early_warning.csv",
              "multimodal_all.csv", "sources.json", "shortvideo_profile.csv",
              "tianchi_profile.csv", "shortvideo_cat_watch.csv"):
        p = os.path.join(DATA, f)
        good = os.path.isfile(p) and os.path.getsize(p) > 0
        ok &= good
        print(f"    {'OK ' if good else '缺失'} {f}")

    print("\n[3] 结果文件（JSON 可解析）")
    for f in ("results.json", "multimodal_results.json", "forecast_results.json",
              "fuzzy_results.json", "profile_results.json", "demo_runs/summary.json"):
        p = os.path.join(OUT, f)
        try:
            json.load(open(p, encoding="utf-8"))
            print(f"    OK  {f}")
        except Exception as e:
            ok = False
            print(f"    异常 {f}: {e}")

    print("\n[3b] 统一管线产物（合并新增；--skip-unified 或未运行时可缺失，不影响主链体检结论）")
    for f in ("up_learners.csv", "up_metrics.json", "up_mining_summary.json",
              "up_association_summary.json", "up_eval_summary.json", "up_report.md"):
        p = os.path.join(OUT, f)
        good = os.path.isfile(p) and os.path.getsize(p) > 0
        print(f"    {'OK ' if good else '缺失'} {f}")

    print("\n[4] 图表与看板")
    figs = [f for f in os.listdir(FIG)] if os.path.isdir(FIG) else []
    need = [f"fig{i:02d}" for i in range(1, 17)]
    miss = [k for k in need if not any(x.startswith(k) for x in figs)]
    ok &= not miss
    print(f"    插图 {len(figs)} 张，缺失编号: {miss if miss else '无'}")
    for f in ("dashboard.html", "persona_dashboard.html", "profile_dashboard.html"):
        p = os.path.join(DASH, f)
        good = os.path.isfile(p) and os.path.getsize(p) > 10_000
        ok &= good
        print(f"    {'OK ' if good else '缺失'} {f}")

    print("\n[5] 报告")
    for f in ("《系统工程原理》实践项目研究报告-选题7.docx", "《系统工程原理》实践项目研究报告-选题7.md"):
        p = os.path.join(PKG, "01_研究报告", f)
        good = os.path.isfile(p) and os.path.getsize(p) > 5_000
        ok &= good
        print(f"    {'OK ' if good else '缺失'} {f}")

    print("\n" + ("=" * 68))
    print("体检结论：" + ("全部通过，可执行一键运行" if ok else "存在缺失项，请按上方提示补齐后再运行"))
    print("=" * 68)
    return 0 if ok else 2


# ---------------------------------------------------------------- 一键运行
def run_all(args) -> int:
    t0 = time.time()
    print("\n" + "=" * 68)
    print("选题7 一键运行 · 单命令复现分析链")
    print(f"开始时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 68)

    steps = []  # (名称, 函数)

    # 0) 可选：从 OULAD 原始表重建主表
    if args.from_raw:
        if not args.oulad_dir or not os.path.isdir(args.oulad_dir):
            print("[ERROR] --from-raw 需要有效的 --oulad-dir <OULAD原始目录>")
            return 2
        steps.append(("从 OULAD 原始表派生真实主表",
                      lambda: _py("build_real_learners.py", "--oulad-dir", args.oulad_dir,
                                  "--out-dir", DATA)))
        steps.append(("从 OULAD 原始表派生预测面板",
                      lambda: _py("build_forecast_data.py", "--oulad-dir", args.oulad_dir,
                                  "--out-dir", DATA)))
    else:
        print("[提示] 默认使用随包派生数据（离线可复现）；如需从 443MB OULAD 原始表重建，加 --from-raw --oulad-dir <目录>")

    # 1) 核心结果链
    steps.append(("六源多形态测评 → multimodal_results.json", lambda: _py("multimodal.py")))
    steps.append(("OULAD 主线（清洗/画像/预测/评价/决策）→ results.json",
                  lambda: _py("app.py", "--batch")))
    steps.append(("周序列预测与早期预警 → forecast_results.json", lambda: _py("forecast.py")))
    steps.append(("模糊综合评价 → fuzzy_results.json", lambda: _py("fuzzy_eval.py")))
    steps.append(("六维画像 → profile_results.json", lambda: _py("profile_modeling.py")))
    steps.append(("三组改参重跑快照 → demo_runs/", lambda: _py("demo_runs.py")))

    # 1.5) 统一管线（合并新增）：六类真实数据源 → 聚类画像 → 挖掘/关联/预测/评价/决策/报告
    #      聚类走 ClusterEngine（--device gpu 时 GPU 加速）；--skip-unified 可跳过
    if not args.skip_unified:
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

    # 2) 图表 + 看板 + 报告
    steps.append(("报告插图 fig01—fig16 → figures/", lambda: _py("figures.py")))
    steps.append(("主看板 → dashboard.html", lambda: _py(os.path.join("..", "05_图表与看板", "build_dashboard.py"))))
    steps.append(("形态画像看板 → persona_dashboard.html",
                  lambda: _py(os.path.join("..", "05_图表与看板", "build_persona_dashboard.py"))))
    steps.append(("六维画像看板 → profile_dashboard.html",
                  lambda: _py(os.path.join("..", "05_图表与看板", "build_profile_dashboard.py"))))
    if not args.skip_report:
        steps.append(("研究报告 → docx / md",
                      lambda: _py(os.path.join("..", "01_研究报告", "build_final_report.py"))))

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

    # 3) 自动体检
    rc = check()

    # 4) 可选启动 Web
    if rc == 0 and args.web:
        print(f"\n[启动] 交互式平台 http://127.0.0.1:{args.port}  （Ctrl+C 停止）")
        os.chdir(SRC)
        return subprocess.run([PY, os.path.join(SRC, "app.py"),
                               "--port", str(args.port), "--no-browser"]).returncode
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="选题7 一键运行入口")
    ap.add_argument("--check", action="store_true", help="只体检，不修改文件")
    ap.add_argument("--web", action="store_true", help="全链路后启动交互式 Web 平台")
    ap.add_argument("--port", type=int, default=8765, help="Web 端口（默认 8765）")
    ap.add_argument("--skip-report", action="store_true", help="不重建 docx（Word 占用时）")
    ap.add_argument("--from-raw", action="store_true", help="从 OULAD 原始表重建主表")
    ap.add_argument("--oulad-dir", default="", help="OULAD 原始目录（配合 --from-raw）")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                    help="计算设备：GPU 加速统一管线聚类（经环境变量 WB_DEVICE 贯通子进程）")
    ap.add_argument("--skip-unified", action="store_true",
                    help="跳过统一管线（up_）阶段，仅跑原 OULAD 主链")
    a = ap.parse_args()
    os.environ["WB_DEVICE"] = a.device          # 设备选择贯通全部子进程
    return check() if a.check else run_all(a)


if __name__ == "__main__":
    sys.exit(main())
