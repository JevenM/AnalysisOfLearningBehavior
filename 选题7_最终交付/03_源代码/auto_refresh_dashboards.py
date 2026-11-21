# -*- coding: utf-8 -*-
"""看板自动刷新监视器：统一管线（或任一阶段脚本）跑完后自动重建三个看板。

同时维护两套看板（各自独立做陈旧度检测，避免重复构建）：
  - 正式版 dashboard.html 等        ← 数据源 04_运行结果/
  - 沙箱版 dashboard.sandbox.html 等 ← 数据源 04_运行结果_沙箱/（存在才构建，
    控制台沙箱试验跑完后自动产出，供控制台切换对比）

触发机制（陈旧度检测）：比较数据源目录中关键运行产物的最新修改时间与对应看板
HTML 的修改时间——只要产物比看板新，即视为"看板过期"，去抖（产物集合连续
两次检查无变化）后重建。

用法：
  python 03_源代码/auto_refresh_dashboards.py              # 持续监视（默认每 5 秒检查）
  python 03_源代码/auto_refresh_dashboards.py --once      # 检查并刷新一次后退出

model_console.py 启动时会自动以后台线程运行本监视器（控制台自身跑链时暂停监视，
由链尾的看板阶段负责构建），因此日常无需手动启动本脚本；只有绕过控制台单独运行
`python 03_源代码/up_cluster.py` 等阶段脚本、又希望看板跟着刷新时才需要手动开着它。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
RUN = os.path.join(PKG, "04_运行结果")
SANDBOX = os.path.join(PKG, "04_运行结果_沙箱")
DASH = os.path.join(PKG, "05_图表与看板")
PY = sys.executable

# 触发刷新的关键运行产物（存在才参与判断）
WATCH_FILES = [
    "up_metrics.json", "up_mining_summary.json", "up_association_summary.json",
    "up_eval_summary.json", "up_queues.csv", "up_persona_recommendations.csv",
    "up_optimization_suggestions.csv", "up_learners.csv",
    "profile_results.json", "multimodal_results.json", "results.json",
]
# 构建目标：(脚本, 数据源目录, 输出文件名, 图片子目录, 必需文件[None=仅要求 up_metrics.json])
BUILD_TARGETS = [
    ("build_dashboard.py", RUN, "dashboard.html", "", None),
    ("build_persona_dashboard.py", RUN, "persona_dashboard.html", "", None),
    ("build_profile_dashboard.py", RUN, "profile_dashboard.html", "", None),
    ("build_dashboard.py", SANDBOX, "dashboard.sandbox.html", "sandbox", None),
    ("build_persona_dashboard.py", SANDBOX, "persona_dashboard.sandbox.html",
     "", "multimodal_results.json"),
    ("build_profile_dashboard.py", SANDBOX, "profile_dashboard.sandbox.html",
     "", "profile_results.json"),
]


def _mt(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def artifacts_mtime(src_dir: str = RUN) -> float:
    """数据源目录中所有存在的关键产物里最新的修改时间（无任何产物返回 0）。"""
    return max((_mt(os.path.join(src_dir, f)) for f in WATCH_FILES), default=0.0)


def stale_targets() -> list:
    """返回过期的构建目标（数据源产物比对应看板新 = 过期）。"""
    out = []
    for script, src_dir, out_name, fig_sub, required in BUILD_TARGETS:
        # 数据源门槛：沙箱需已产出 up_metrics.json；形态/画像看板还需各自源文件
        if src_dir == SANDBOX and \
                not os.path.isfile(os.path.join(SANDBOX, "up_metrics.json")):
            continue
        if required and not os.path.isfile(os.path.join(src_dir, required)):
            continue
        newest = artifacts_mtime(src_dir)
        if newest <= 0:
            continue
        hp = os.path.join(DASH, out_name)
        if not os.path.exists(hp) or _mt(hp) < newest:
            out.append((script, src_dir, out_name, fig_sub))
    return out


def run_builders(targets: list, log=_log) -> bool:
    """依次运行给定的看板构建目标，全部成功返回 True。"""
    ok = True
    for script, src_dir, out_name, fig_sub in targets:
        t0 = time.time()
        env = dict(os.environ)
        if src_dir != RUN:
            env["DASH_SOURCE"] = src_dir
        env["DASH_OUT"] = out_name
        env["DASH_FIG_SUB"] = fig_sub
        r = subprocess.run([PY, os.path.join(DASH, script)], cwd=HERE,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
        if r.returncode == 0:
            log(f"  已重建 {out_name}（{time.time() - t0:.1f}s，"
                f"源 {'04_运行结果_沙箱' if src_dir == SANDBOX else '04_运行结果'}）")
        else:
            ok = False
            log(f"  失败 {script} → {out_name}（退出码 {r.returncode}）："
                f"{(r.stderr or r.stdout or '').strip().splitlines()[-1:]}")
    return ok


def refresh_if_stale(log=_log, running_check=None) -> bool:
    """有看板过期则重建（去抖 3 秒）；返回是否执行了刷新。

    running_check：可选回调（返回 True 表示有建模链正在运行，此时跳过——
    由链尾自己的看板阶段负责构建，避免并发重复写）。
    """
    if running_check is not None and running_check():
        return False
    stale = stale_targets()
    if not stale:
        return False
    # 去抖：等数据源产物稳定（3 秒内时间戳不再变化）再构建，
    # 避免管线写文件写到一半时用半成品刷新
    time.sleep(3)
    if running_check is not None and running_check():
        return False
    stale = stale_targets()
    if not stale:
        return False
    log("检测到运行产物比看板新（过期：" +
        "、".join(t[2] for t in stale) + "），自动重建看板 …")
    ok = run_builders(stale, log)
    log("看板自动刷新完成" if ok else "看板自动刷新有失败项（详见上方）")
    return ok


def watch_loop(interval: float = 5.0, running_check=None) -> None:
    _log(f"看板自动刷新监视器启动（每 {interval:.0f} 秒检查："
         "04_运行结果 与 04_运行结果_沙箱 vs 两套看板）")
    while True:
        try:
            refresh_if_stale(running_check=running_check)
        except Exception as e:  # noqa: BLE001 监视器不因单次异常退出
            _log(f"监视器异常（继续运行）: {type(e).__name__}: {e}")
        time.sleep(interval)


def main() -> int:
    ap = argparse.ArgumentParser(description="看板自动刷新监视器")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="检查间隔秒数（默认 5）")
    ap.add_argument("--once", action="store_true",
                    help="只检查并刷新一次后退出")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.once:
        refresh_if_stale()
        return 0
    watch_loop(a.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())

