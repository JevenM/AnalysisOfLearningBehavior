# -*- coding: utf-8 -*-
"""model_console.py —— 建模控制台（课程要求：可交互验证程序 + 现场换数据/换参数重跑）

对应《2026 年实践项目指南》4.4：
  (1) 可运行的方法验证程序，具备可交互界面，能读入不同输入数据、运行验证方法、输出结果；
  (2) 答辩现场更换输入数据或参数后重新运行，并解释结果变化。

功能：
  - 本地网页控制台（Python 标准库实现，无第三方依赖）：修改建模参数
    （聚类数 K / 计算设备 / Apriori 支持度与置信度 / SD 仿真周数 / 多形态预算比例）；
  - 更换输入数据：上传自定义学习者总表（up_unified.csv 同 schema）替换默认六源合并表，
    可一键恢复默认数据；
  - 按新参数重跑管线（正式模式写入 04_运行结果/ 并刷新看板；沙箱模式写入
    04_运行结果_沙箱/，不动正式结果），输出预测结果、干预队列与优化建议，
    并与上一次运行自动对比（结果变化高亮，供答辩解释）。

启动：python 03_源代码/model_console.py [--port 8000] [--no-browser]
页面：05_图表与看板/model_console.html（由本服务托管，与三个看板互链）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
RUN = os.path.join(PKG, "04_运行结果")
SANDBOX = os.path.join(PKG, "04_运行结果_沙箱")
DASH = os.path.join(PKG, "05_图表与看板")
STATE = os.path.join(RUN, "model_console")
PY = sys.executable

DEFAULT_PARAMS = {
    "k": 0,                 # 聚类数：0=自动（肘部+轮廓双准则）
    "device": "auto",       # auto / cpu / gpu（WB_DEVICE 贯通子进程）
    "min_support": 0.05,    # Apriori 最小支撑度
    "min_conf": 0.4,        # Apriori 最小置信度
    "sd_weeks": 52,         # SD 仿真周数
    "budget_ratio": 0.45,   # 多形态 0-1 规划预算比例
}
PARAM_LIMITS = {"k": (0, 8), "min_support": (0.01, 0.30), "min_conf": (0.10, 0.95),
                "sd_weeks": (8, 260), "budget_ratio": (0.05, 1.0)}

# 阶段定义：key → (名称, 脚本, 参数模板, 类别)
#   ext=扩展证据线（仅正式模式）  data=数据层  pipeline=分析链  dash=看板构建
STAGES = {
    "multimodal":       ("四形态测评（含 0-1 预算规划）", "multimodal.py",
                         ["--budget-ratio", "{budget_ratio}"], "ext"),
    "profile":          ("内容级画像", "profile_modeling.py", [], "ext"),
    "up_build_unified": ("统一管线[1] 六源融合", "up_build_unified.py", [], "data"),
    "up_cluster":       ("统一管线[2] 聚类画像", "up_cluster.py",
                         ["--k", "{k}"], "pipeline"),
    "up_mine":          ("统一管线[3] 行为挖掘", "up_mine.py",
                         ["--min-support", "{min_support}", "--min-conf", "{min_conf}"], "pipeline"),
    "up_associate":     ("统一管线[4] 相关与分层", "up_associate.py", [], "pipeline"),
    "up_predict":       ("统一管线[5] 预测与预警", "up_predict.py", [], "pipeline"),
    "up_evaluate":      ("统一管线[6] 评价与 SD 仿真", "up_evaluate.py",
                         ["--sd-weeks", "{sd_weeks}"], "pipeline"),
    "up_decide":        ("统一管线[7] 决策与建议", "up_decide.py", [], "pipeline"),
    "up_report":        ("统一管线[8] 汇总报告", "up_report.py", [], "pipeline"),
    "dash_main":        ("主看板", os.path.join(DASH, "build_dashboard.py"), [], "dash"),
    "dash_persona":     ("形态画像看板", os.path.join(DASH, "build_persona_dashboard.py"), [], "dash"),
    "dash_profile":     ("内容画像看板", os.path.join(DASH, "build_profile_dashboard.py"), [], "dash"),
}
SCOPE_STAGES = {
    "full": ["multimodal", "profile", "up_build_unified", "up_cluster", "up_mine",
             "up_associate", "up_predict", "up_evaluate", "up_decide", "up_report",
             "dash_main", "dash_persona", "dash_profile"],
    "pipeline": ["up_build_unified", "up_cluster", "up_mine", "up_associate",
                 "up_predict", "up_evaluate", "up_decide", "up_report",
                 "dash_main", "dash_persona", "dash_profile"],
    "sandbox": ["up_build_unified", "up_cluster", "up_mine", "up_associate",
                "up_predict", "up_evaluate", "up_decide", "up_report"],
    "sandbox_upload": ["up_cluster", "up_mine", "up_associate", "up_predict",
                       "up_evaluate", "up_decide", "up_report"],
}
DASH_FILES = {"dashboard.html", "persona_dashboard.html", "profile_dashboard.html",
              "model_console.html"}
RUN_FILES = {"up_report.md", "up_metrics.json", "up_eval_summary.json",
             "multimodal_results.json", "profile_results.json", "up_queues.csv",
             "up_optimization_suggestions.csv", "up_persona_recommendations.csv"}

_LOCK = threading.Lock()
RUN_STATE = {"running": False, "scope": "", "stage": "", "log": [], "rc": None,
             "t_start": None, "t_end": None, "out_dir": ""}


def _log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    with _LOCK:
        RUN_STATE["log"].append(line)
        if len(RUN_STATE["log"]) > 4000:
            del RUN_STATE["log"][:1000]
    try:
        os.makedirs(STATE, exist_ok=True)
        with open(os.path.join(STATE, "last_run.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ------------------------------------------------------------ 参数与状态
def load_params() -> dict:
    try:
        with open(params_path(), encoding="utf-8") as f:
            p = {**DEFAULT_PARAMS, **json.load(f)}
    except (OSError, ValueError):
        p = dict(DEFAULT_PARAMS)
    for k, (lo, hi) in PARAM_LIMITS.items():
        try:
            v = type(p[k])(p[k])
        except (TypeError, ValueError):
            v = DEFAULT_PARAMS[k]
        p[k] = min(hi, max(lo, v))
    return p


def params_path() -> str:
    return os.path.join(STATE, "model_params.json")


def save_params(p: dict) -> dict:
    out = {}
    for k, (lo, hi) in PARAM_LIMITS.items():
        v = p.get(k, DEFAULT_PARAMS[k])
        try:
            v = type(DEFAULT_PARAMS[k])(v)
        except (TypeError, ValueError):
            raise ValueError(f"参数 {k} 类型错误")
        if not (lo <= v <= hi):
            raise ValueError(f"参数 {k}={v} 超出允许范围 [{lo}, {hi}]")
        out[k] = v
    out["device"] = p.get("device") if p.get("device") in ("auto", "cpu", "gpu") else "auto"
    os.makedirs(STATE, exist_ok=True)
    with open(params_path(), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def data_mode() -> dict:
    """输入数据模式：默认六源合并表 / 自定义上传。"""
    backup = os.path.join(RUN, "up_unified.default.csv")
    cur = os.path.join(RUN, "up_unified.csv")
    custom = os.path.exists(backup)
    info = {"custom": custom, "sandbox_ready":
            os.path.exists(os.path.join(SANDBOX, "up_unified.csv")), "rows": None}
    if os.path.exists(cur):
        with open(cur, encoding="utf-8-sig") as f:
            info["rows"] = sum(1 for _ in f) - 1
    return info


# ------------------------------------------------------------ 结果摘要与对比
def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _counter_csv(path, col):
    try:
        with open(path, encoding="utf-8-sig") as f:
            return dict(Counter(r.get(col, "") for r in csv.DictReader(f)))
    except OSError:
        return {}


def summarize(out_dir: str):
    """从指定产物目录提取结果摘要（供答辩对比解释）。"""
    if not os.path.isdir(out_dir):
        return None
    um = _read_json(os.path.join(out_dir, "up_metrics.json"))
    ev = _read_json(os.path.join(out_dir, "up_eval_summary.json"))
    mi = _read_json(os.path.join(out_dir, "up_mining_summary.json"))
    mm = _read_json(os.path.join(RUN, "multimodal_results.json"))
    s = {
        "k": um.get("k"),
        "silhouette": um.get("silhouette"),
        "personas": _counter_csv(os.path.join(out_dir, "up_learners.csv"), "persona"),
        "cert_acc": um.get("cert_acc"),
        "cert_recall_risk": um.get("cert_recall_risk"),
        "rf_auc_early": um.get("rf_auc_early"),
        "rf_auc_full": um.get("rf_auc_full"),
        "gm_trends": um.get("gm_trends", {}),
        "ahp_cr": ev.get("ahp_cr"),
        "quality_mean": ev.get("quality_mean"),
        "sd_best": ev.get("sd_best"),
        "sd_growth": (ev.get("sd_results", {}) or {}).get(
            ev.get("sd_best", ""), {}).get("growth_pct"),
        "queues": _counter_csv(os.path.join(out_dir, "up_queues.csv"), "queue"),
        "apriori_top": (mi.get("apriori_top") or [{}])[0],
        "suggestions": None,
        "budget": {k: (mm.get("decide", {}) or {}).get(k) for k in
                   ("budget_ratio", "budget", "opt_gain", "heur_gain", "uniform_gain")},
    }
    try:
        with open(os.path.join(out_dir, "up_optimization_suggestions.csv"),
                  encoding="utf-8-sig") as f:
            s["suggestions"] = len(list(csv.DictReader(f)))
    except OSError:
        pass
    return s


def snapshot_before(scope: str, out_dir: str) -> None:
    """运行前把当前结果快照为 previous，供运行后对比。"""
    os.makedirs(STATE, exist_ok=True)
    cur = summarize(out_dir)
    if cur is not None:
        with open(os.path.join(STATE, f"prev_{scope}.json"), "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)


def results_with_prev(scope: str) -> dict:
    out_dir = SANDBOX if scope.startswith("sandbox") else RUN
    prev = None
    try:
        with open(os.path.join(STATE, f"prev_{scope}.json"), encoding="utf-8") as f:
            prev = json.load(f)
    except (OSError, ValueError):
        pass
    return {"current": summarize(out_dir), "previous": prev, "out_dir": out_dir}


# ------------------------------------------------------------ 运行链
def _fmt(args, p):
    return [a.format(**p) if "{" in a else a for a in args]


def build_cmds(scope: str, stages):
    p = load_params()
    if scope.startswith("sandbox"):
        out_dir, allow = SANDBOX, ("data", "pipeline")
    else:
        out_dir, allow = RUN, ("ext", "data", "pipeline", "dash")
    keys = stages if stages else SCOPE_STAGES.get(scope, [])
    cmds = []
    for key in keys:
        if key not in STAGES:
            continue
        name, script, extra, kind = STAGES[key]
        if kind not in allow:
            continue
        cmd = [PY, os.path.join(HERE, script)] + _fmt(extra, p) + \
            (["--out-dir", out_dir] if kind in ("data", "pipeline") else [])
        cmds.append({"key": key, "name": name, "cmd": cmd, "kind": kind})
    if scope.startswith("sandbox"):
        os.makedirs(SANDBOX, exist_ok=True)
        src, dst = os.path.join(RUN, "up_clicks.csv"), os.path.join(SANDBOX, "up_clicks.csv")
        if os.path.exists(src) and not os.path.exists(dst):
            _log("沙箱补齐事件流表（沿用默认数据，供挖掘/预警演示）")
            shutil.copyfile(src, dst)
    return cmds


def _run_chain(scope: str, cmds) -> None:
    try:
        env = {**os.environ, "WB_DEVICE": load_params()["device"],
               "PYTHONIOENCODING": "utf-8"}
        for i, c in enumerate(cmds, 1):
            with _LOCK:
                RUN_STATE["stage"] = c["name"]
            _log(f"[{i}/{len(cmds)}] {c['name']}")
            t = time.time()
            proc = subprocess.Popen(c["cmd"], cwd=HERE, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace",
                                    bufsize=1)
            for line in proc.stdout:
                _log("  " + line.rstrip())
            rc = proc.wait()
            if rc != 0:
                _log(f"[失败] {c['name']} 退出码 {rc}，链路停止")
                with _LOCK:
                    RUN_STATE["rc"] = rc
                return
            _log(f"[完成] {c['name']} 耗时 {time.time() - t:.1f}s")
        with _LOCK:
            RUN_STATE["rc"] = 0
        _log(f"[链路结束] scope={scope}，共 {len(cmds)} 步，结果目录 {RUN_STATE['out_dir']}")
        if scope == "full":
            _log("[提示] 正式结果已更新：在“结果看板”区刷新三个看板链接查看")
    except Exception as e:  # noqa: BLE001
        _log(f"[异常] {type(e).__name__}: {e}")
        with _LOCK:
            RUN_STATE["rc"] = 2
    finally:
        with _LOCK:
            RUN_STATE["running"] = False
            RUN_STATE["t_end"] = time.time()
            RUN_STATE["stage"] = ""


def start_run(scope: str, stages) -> dict:
    if RUN_STATE["running"]:
        return {"ok": False, "error": "已有任务在运行，请等待完成"}
    cmds = build_cmds(scope, stages)
    if not cmds:
        return {"ok": False, "error": "没有可运行的阶段（检查 scope/stages）"}
    RUN_STATE.update({"running": True, "scope": scope, "stage": "", "log": [],
                      "rc": None, "t_start": time.time(), "t_end": None,
                      "out_dir": SANDBOX if scope.startswith("sandbox") else RUN})
    snapshot_before(scope, RUN_STATE["out_dir"])
    _log(f"[开始] scope={scope}，参数 {load_params()}，{len(cmds)} 个阶段")
    threading.Thread(target=_run_chain, args=(scope, cmds), daemon=True).start()
    return {"ok": True, "stages": [c["name"] for c in cmds]}


# ------------------------------------------------------------ 数据上传
REQUIRED_COLS = {"student_id", "source", "total_clicks", "active_days",
                 "final_grade", "success", "completion_rate"}


def upload_data(body: bytes, target: str, filename: str) -> dict:
    header_line = body.split(b"\n", 1)[0].decode("utf-8-sig", "replace").strip().lower()
    header = {h.strip().strip('"') for h in header_line.split(",")}
    missing = REQUIRED_COLS - header
    if missing:
        return {"ok": False, "error":
                f"缺少必需列: {', '.join(sorted(missing))}。上传表需与 up_unified.csv "
                "同 schema（含 learner_id/source/行为特征/结果标签）"}
    n = body.count(b"\n")
    if target == "sandbox":
        os.makedirs(SANDBOX, exist_ok=True)
        with open(os.path.join(SANDBOX, "up_unified.csv"), "wb") as f:
            f.write(body)
        _log(f"[数据] 沙箱输入已替换: {filename}（{n - 1} 行）→ 沙箱链路将从“聚类画像”起跑")
        return {"ok": True, "rows": n - 1, "target": target}
    backup = os.path.join(RUN, "up_unified.default.csv")
    cur = os.path.join(RUN, "up_unified.csv")
    if not os.path.exists(backup) and os.path.exists(cur):
        shutil.copyfile(cur, backup)
    with open(cur, "wb") as f:
        f.write(body)
    return {"ok": True, "rows": n - 1, "target": target,
            "note": "正式总表已替换（默认表备份为 up_unified.default.csv）；"
                    "建议运行“统一管线(含数据层)+看板”"}


def reset_data() -> dict:
    msgs = []
    backup = os.path.join(RUN, "up_unified.default.csv")
    if os.path.exists(backup):
        shutil.copyfile(backup, os.path.join(RUN, "up_unified.csv"))
        os.remove(backup)
        msgs.append("正式总表已恢复为默认六源合并表")
    if os.path.isdir(SANDBOX):
        shutil.rmtree(SANDBOX)
        msgs.append("沙箱目录已清空")
    _log("[数据] " + ("；".join(msgs) if msgs else "无可恢复内容"))
    return {"ok": True, "messages": msgs or ["无可恢复内容"]}


# ------------------------------------------------------------ HTTP 服务
class Handler(BaseHTTPRequestHandler):
    server_version = "ModelConsole/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/model_console.html"):
                with open(os.path.join(DASH, "model_console.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/api/state":
                with _LOCK:
                    run = {k: RUN_STATE[k] for k in
                           ("running", "scope", "stage", "rc", "t_start", "t_end", "out_dir")}
                self._json({"params": load_params(), "run": run,
                            "log_n": len(RUN_STATE["log"]), "data": data_mode(),
                            "results": {"canonical": summarize(RUN),
                                        "sandbox": summarize(SANDBOX)}})
            elif path == "/api/log":
                m = re.search(r"[?&]n=(\d+)", self.path)
                n = int(m.group(1)) if m else 0
                with _LOCK:
                    lines = RUN_STATE["log"][n:]
                    run = {k: RUN_STATE[k] for k in
                           ("running", "scope", "stage", "rc", "t_start", "t_end")}
                    total = len(RUN_STATE["log"])
                self._json({"lines": lines, "next": total, "run": run})
            elif path == "/api/results":
                m = re.search(r"[?&]scope=([a-z_]+)", self.path)
                self._json(results_with_prev(m.group(1) if m else "sandbox"))
            elif path == "/api/figures":
                m = re.search(r"[?&]scope=(canonical|sandbox)", self.path)
                sandbox = bool(m and m.group(1) == "sandbox")
                d = SANDBOX if sandbox else RUN
                figs = sorted(
                    ({"name": f,
                      "mtime": os.path.getmtime(os.path.join(d, f))}
                     for f in os.listdir(d)
                     if f.lower().endswith(".png") and
                     os.path.isfile(os.path.join(d, f))),
                    key=lambda x: x["name"])
                self._json({"ok": True,
                            "scope": "sandbox" if sandbox else "canonical",
                            "figures": figs})
            elif path.startswith("/fig/"):
                m = re.match(r"^/fig/(canonical|sandbox)/([\w\-]+\.png)$", path)
                d = (SANDBOX if m.group(1) == "sandbox" else RUN) if m else None
                fp = os.path.join(d, m.group(2)) if m else ""
                if not m or not os.path.isfile(fp):
                    self._json({"error": "图表不存在（先运行管线）"}, 404)
                else:
                    with open(fp, "rb") as f:
                        self._send(200, f.read(), "image/png")
            elif path.startswith("/dash/") and os.path.basename(path) in DASH_FILES:
                with open(os.path.join(DASH, os.path.basename(path)), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path.startswith("/files/") and os.path.basename(path) in RUN_FILES:
                fp = os.path.join(RUN, os.path.basename(path))
                if os.path.exists(fp):
                    with open(fp, "rb") as f:
                        self._send(200, f.read(), "text/plain; charset=utf-8")
                else:
                    self._json({"error": "文件不存在（先运行管线）"}, 404)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        try:
            path = self.path.split("?")[0]
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            if path == "/api/params":
                self._json({"ok": True, "params": save_params(json.loads(body or b"{}"))})
            elif path == "/api/params/reset":
                self._json({"ok": True, "params": save_params(dict(DEFAULT_PARAMS))})
            elif path == "/api/run":
                req = json.loads(body or b"{}")
                scope, stages = req.get("scope", ""), req.get("stages")
                stages = stages if isinstance(stages, list) else None
                if scope not in SCOPE_STAGES and scope != "single":
                    self._json({"ok": False, "error": "scope 非法"}, 400)
                elif scope == "single" and not stages:
                    self._json({"ok": False, "error": "单阶段运行需提供 stages"}, 400)
                else:
                    self._json(start_run(scope, stages))
            elif path == "/api/data":
                m = re.search(r"[?&]target=(canonical|sandbox)", self.path)
                fn = re.search(r"[?&]filename=([^&]+)", self.path)
                self._json(upload_data(body, m.group(1) if m else "sandbox",
                                       fn.group(1) if fn else "upload.csv"))
            elif path == "/api/data/reset":
                self._json(reset_data())
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main() -> int:
    ap = argparse.ArgumentParser(description="选题7 建模控制台（可交互验证程序）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    url = f"http://127.0.0.1:{a.port}/"
    print("=" * 60)
    print("选题7 建模控制台（可交互验证程序）")
    print(f"  控制台: {url}")
    print("  看板:   {0}dash/dashboard.html".format(url))
    print("  看板自动刷新监视器已随控制台启动（跑完管线自动重建三个看板）")
    print("  停止:   Ctrl+C")
    print("=" * 60)
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    # 看板自动刷新监视器：绕过控制台单独跑 up_*.py 后，看板也会自动重建
    import auto_refresh_dashboards as autod
    threading.Thread(target=autod.watch_loop, daemon=True,
                     kwargs={"interval": 5.0,
                             "running_check": lambda: RUN_STATE["running"]}
                     ).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n控制台已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
