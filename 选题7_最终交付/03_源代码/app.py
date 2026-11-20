# -*- coding: utf-8 -*-
"""
app.py —— 《系统工程原理》选题7「可验证成果」交互验证平台

对应指南 4.4：具备可交互界面，能读入不同的输入数据、修改参数并重新运行
全部方法、输出可核验结果；答辩现场可"换数据 / 改参数 → 重跑 → 解释变化"。

三种运行方式：
  1) Web 交互界面（推荐）：
       python app.py                      # 自动打开 http://127.0.0.1:8765
       python app.py --port 9000 --no-browser
  2) 命令行交互（无浏览器环境）：
       python app.py --cli
  3) 一次性批跑 / 复现报告基准结果：
       python app.py --batch --k 4 --budget 0.45 --out outputs/results.json

依赖：numpy / pandas / scipy / scikit-learn（见 requirements.txt）
界面与看板均为本地文件，无外网依赖；上传的 CSV 只写入本目录 data/ 下。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

import analysis

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "02_数据与数据集"))
OUT_DIR = os.path.abspath(os.path.join(ROOT, "..", "04_运行结果"))
PKG_DIR = os.path.abspath(os.path.join(ROOT, ".."))
DASHBOARD = os.path.join(PKG_DIR, "05_图表与看板", "dashboard.html")
PERSONA_DASHBOARD = os.path.join(PKG_DIR, "05_图表与看板", "persona_dashboard.html")
PROFILE_DASHBOARD = os.path.join(PKG_DIR, "05_图表与看板", "profile_dashboard.html")
FIGURE_DIR = os.path.join(PKG_DIR, "05_图表与看板", "figures")
BUILTIN_CSV = os.path.join(DATA_DIR, "real_learners.csv")
UPLOAD_CSV = os.path.join(DATA_DIR, "uploaded.csv")
BENCHMARK_JSON = os.path.join(OUT_DIR, "results.json")
LATEST_JSON = os.path.join(OUT_DIR, "results_latest.json")

DEFAULTS = {
    "data_source": "builtin",   # builtin（OULAD 真实）| uploaded（现场上传真实数据）
    "k": 4,
    "seed": 20260101,           # 可复现随机种子（KMeans/CV 用），非仿真参数
    "budget_ratio": 0.45,
    "test_size": 0.25,
    "weight_mode": "combined",  # combined | custom
    "ahp_alpha": 0.5,
    "custom_weights": [0.2361, 0.1959, 0.1425, 0.1569, 0.1324, 0.1360],
    "speed": "full",            # full（与报告一致）| fast（现场快速预览）
}

# 线程锁：同一时刻只允许一次重跑，避免答辩现场并发点按导致结果串台
RUN_LOCK = threading.Lock()
STATE = {"latest": None, "last_params": None, "last_elapsed": None, "last_error": None}


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
def prepare_data(p: dict) -> tuple[str, str]:
    """按参数选择数据源，返回 (csv路径, 数据源说明)。"""
    src = p.get("data_source", "builtin")
    if src == "uploaded":
        if not os.path.exists(UPLOAD_CSV):
            raise ValueError("尚未上传 CSV，请先在界面『数据源』处上传，或改用内置真实数据")
        return UPLOAD_CSV, "现场上传的 CSV 数据"
    return BUILTIN_CSV, "内置 OULAD 真实学习者主表 real_learners.csv（20105 条 / 18634 人）"


def validate_csv(path: str) -> dict:
    """读入任意 CSV 时做最小口径检查，给出可解释的错误信息。"""
    df = pd.read_csv(path, encoding="utf-8-sig", nrows=5)
    # course_num 由 load_clean() 依据 learner_id×course_id 派生，不要求输入提供
    required = {
        "record_id", "learner_id", "course_id", "course_name", "discipline",
        "region", "education", "gender", "age", "visit_count", "active_days",
        "study_hours", "avg_session_min", "forum_posts", "first_active_lag",
        "video_rate", "assign_rate", "quiz_avg", "final_grade", "certificate",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("上传数据缺少必需字段：" + "、".join(missing) +
                         "。字段口径见《数据说明.md》，可直接下载模板对照。")
    full = pd.read_csv(path, encoding="utf-8-sig")
    return {"rows": int(len(full)), "cols": int(len(full.columns)),
            "columns": list(full.columns), "missing_fields": missing}


# ---------------------------------------------------------------------------
# 预测（fast 版：仅保留与报告口径一致的 Logistic 回归主模型）
# ---------------------------------------------------------------------------
def predict_fast(df: pd.DataFrame, test_size: float, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score, roc_curve)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    feats = analysis.PREDICT_FEATURES   # 无泄漏：前期行为 + 人口属性，剔除同期测评结果
    X = df[feats].values
    y = df["certificate"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    m.fit(sc.transform(Xtr), ytr)
    prob = m.predict_proba(sc.transform(Xte))[:, 1]
    pred = (prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(yte, prob)
    met = {"auc": round(float(roc_auc_score(yte, prob)), 4),
           "accuracy": round(float(accuracy_score(yte, pred)), 4),
           "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
           "recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
           "f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
           "cv_auc": None,
           "confusion": confusion_matrix(yte, pred).tolist()}
    odds = [{"feature": f, "feature_cn": analysis.FEATURE_CN[f],
             "coef": round(float(c), 4), "odds_ratio": round(float(np.exp(c)), 4)}
            for f, c in zip(feats, m.coef_[0])]
    odds.sort(key=lambda d: -d["odds_ratio"])
    imp = [{"feature": f, "feature_cn": analysis.FEATURE_CN[f],
            "importance": round(float(abs(c)), 4)}
           for f, c in zip(feats, m.coef_[0])]
    imp.sort(key=lambda d: -d["importance"])
    step = max(1, len(fpr) // 60)
    return {"metrics": {"Logistic回归": met}, "odds": odds, "importance": imp,
            "importance_source": "快速模式：Logistic 回归标准化系数 |β|（完整模式为随机森林特征重要性）",
            "roc": {"fpr": [round(float(v), 4) for v in fpr[::step]],
                    "tpr": [round(float(v), 4) for v in tpr[::step]]},
            "test_n": int(len(yte)), "pos_rate": round(float(yte.mean()), 4)}


# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------
def load_multimodal():
    """读取国内多形态层测评结果；不存在时返回 None，不影响主流程。"""
    p = os.path.join(OUT_DIR, "multimodal_results.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run_pipeline(p: dict) -> dict:
    """按界面参数重新运行：清洗 → ISM → 聚类 → 预测 → 评价 → 决策 → 灵敏度。"""
    t0 = time.time()
    p = {**DEFAULTS, **p}
    seed = int(p["seed"])
    analysis.SEED = seed                      # 影响 KMeans / 划分 / 模型随机性

    csv_path, src_desc = prepare_data(p)
    info = validate_csv(csv_path)
    df, log = analysis.load_clean(csv_path)
    q = analysis.quality_profile(df)
    desc = analysis.describe(df)
    cr = analysis.corr_matrix(df)

    k = int(p["k"])
    fast = p["speed"] == "fast"
    cl = analysis.cluster(df, k=k, k_range=[k] if fast else range(2, 9))

    if fast:
        pr = predict_fast(cl["df"], float(p["test_size"]), seed)
    else:
        pr = analysis.predict(cl["df"], test_size=float(p["test_size"]))
        pr = {kk: vv for kk, vv in pr.items() if kk != "fitted"}

    # 系统评价：AHP-熵权组合权重 → TOPSIS
    a = analysis.ahp()
    ent = analysis.entropy_weight(df)
    alpha = float(p["ahp_alpha"])
    if p["weight_mode"] == "custom":
        w = np.array(p["custom_weights"], dtype=float)
        if len(w) != 6 or w.sum() <= 0:
            raise ValueError("自定义权重需为 6 个非负数且和大于 0")
        w = w / w.sum()
        a = {"weights": None, "cr": None, "note": "本次采用界面自定义权重，未做 AHP 一致性检验"}
    else:
        w = alpha * np.array(a["weights"]) + (1 - alpha) * np.array(ent["weights"])
        w = w / w.sum()
    ev = {"ahp": a, "entropy": ent, "combined": w.round(4).tolist(),
          "rank": analysis.topsis(df, w),
          "criteria": [{"code": c, "name": n, "field": f} for c, n, f in analysis.CRITERIA]}

    budget = float(p["budget_ratio"])
    dc = analysis.decide(cl, budget_ratio=budget)
    sw = ([] if fast else analysis.budget_sweep(
        cl, [0.2, 0.3, 0.4, 0.45, 0.55, 0.7, 0.85, 1.0]))
    im = analysis.ism()

    elapsed = round(time.time() - t0, 2)
    results = {
        "meta": {"params": p, "data_source": src_desc, "csv_info": info,
                 "elapsed_sec": elapsed, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "speed": p["speed"], "ahp_alpha": alpha,
                 "weight_mode": p["weight_mode"]},
        "clean_log": log, "quality": q, "describe": desc,
        "corr": {"cols": list(cr.columns), "matrix": cr.values.round(3).tolist()},
        "cluster": {kk: vv for kk, vv in cl.items() if kk not in ("df", "labels")},
        "predict": pr, "evaluate": ev, "decide": dc, "sweep": sw, "ism": im,
        "multimodal": load_multimodal(),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    STATE.update(latest=results, last_params=p, last_elapsed=elapsed, last_error=None)
    return results


def to_view(r: dict) -> dict:
    """把完整结果转换成看板所用结构，保证『在线重跑』与『离线基准』结构完全一致。"""
    roc = r["predict"]["roc"]
    step = max(1, len(roc["fpr"]) // 120)
    return {
        "meta": r["meta"],
        "clean_log": r["clean_log"], "quality": r["quality"], "corr": r["corr"],
        "cluster": {"k": r["cluster"]["k"], "silhouette": r["cluster"]["silhouette"],
                    "ari": r["cluster"]["ari"], "scan": r["cluster"]["scan"],
                    "profile": r["cluster"]["profile_records"]},
        "predict": {"metrics": r["predict"]["metrics"],
                    "importance": r["predict"]["importance"],
                    "importance_source": r["predict"].get("importance_source"),
                    "odds": r["predict"]["odds"],
                    "roc": {"fpr": roc["fpr"][::step], "tpr": roc["tpr"][::step]}},
        "evaluate": r["evaluate"],
        "decide": {k: v for k, v in r["decide"].items() if k in
                   ("budget", "opt_gain", "heur_gain", "uniform_gain", "opt_cost",
                    "chosen", "status", "segments", "uniform_note")},
        "sweep": r["sweep"],
        "ism": {k: r["ism"][k] for k in ("levels", "names", "micmac")},
        "multimodal": r.get("multimodal"),
    }


def summarize(r: dict) -> dict:
    """给前端的精简结果（响应体小、刷新快）。"""
    cl, pr, ev, dc = r["cluster"], r["predict"], r["evaluate"], r["decide"]
    best = max(pr["metrics"].items(), key=lambda kv: kv[1]["auc"])
    return {
        "meta": r["meta"],
        "data": {"rows": r["clean_log"]["clean_rows"], "learners": r["clean_log"]["learners"],
                 "courses": r["clean_log"]["courses"], "cert_rate": r["clean_log"]["cert_rate"],
                 "quality": r["quality"]},
        "ism": {"levels": r["ism"]["levels"], "names": r["ism"]["names"]},
        "cluster": {"k": cl["k"], "silhouette": cl["silhouette"], "ari": cl["ari"],
                    "scan": cl["scan"], "profile": cl["profile_records"]},
        "predict": {"metrics": pr["metrics"], "best_model": best[0], "best_auc": best[1]["auc"],
                    "importance": pr["importance"], "odds": pr["odds"], "roc": pr["roc"]},
        "evaluate": {"ahp": ev["ahp"], "entropy_weights": ev["entropy"]["weights"],
                     "combined": ev["combined"], "rank": ev["rank"], "criteria": ev["criteria"]},
        "decide": {"budget": dc["budget"], "opt_gain": dc["opt_gain"],
                   "heur_gain": dc["heur_gain"], "uniform_gain": dc["uniform_gain"],
                   "opt_cost": dc["opt_cost"], "chosen": dc["chosen"], "status": dc["status"]},
        "sweep": r["sweep"],
        "multimodal": (None if not r.get("multimodal") else {
            "headline": r["multimodal"]["headline"],
            "form_auc": {f: r["multimodal"]["models"][f]["metrics"].get("best_cv_auc")
                         for f in r["multimodal"]["meta"]["forms"]},
            "rank": r["multimodal"]["evaluate"]["rank"],
            "decide": {k: r["multimodal"]["decide"][k]
                       for k in ("budget", "opt_gain", "heur_gain", "uniform_gain")},
            "transfer": r["multimodal"]["transfer"]["pairs"],
        }),
    }


# ---------------------------------------------------------------------------
# HTTP 服务
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "SysEng7/1.0"

    def log_message(self, fmt, *args):
        if os.environ.get("APP_VERBOSE"):
            sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, body: bytes, ctype="application/json; charset=utf-8", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), code=code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        path = unquote(u.path)
        try:
            if path in ("/", "/index.html", "/dashboard.html"):
                with open(DASHBOARD, "rb") as f:
                    return self._send(f.read(), "text/html; charset=utf-8")
            if path in ("/persona", "/persona_dashboard.html"):
                with open(PERSONA_DASHBOARD, "rb") as f:
                    return self._send(f.read(), "text/html; charset=utf-8")
            if path in ("/profile", "/profile_dashboard.html"):
                with open(PROFILE_DASHBOARD, "rb") as f:
                    return self._send(f.read(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json({
                    "mode": "server", "defaults": DEFAULTS,
                    "has_uploaded": os.path.exists(UPLOAD_CSV),
                    "benchmark_exists": os.path.exists(BENCHMARK_JSON),
                    "builtin_rows": 20105,
                    "figures": sorted(os.listdir(FIGURE_DIR)) if os.path.isdir(FIGURE_DIR) else [],
                    "latest_params": STATE["last_params"],
                })
            if path == "/api/results":
                with open(BENCHMARK_JSON, "rb") as f:
                    return self._send(f.read())
            if path == "/api/latest":
                if STATE["latest"] is None:
                    return self._json({"empty": True})
                return self._json(to_view(STATE["latest"]))
            if path == "/api/export":
                target = LATEST_JSON if os.path.exists(LATEST_JSON) else BENCHMARK_JSON
                with open(target, "rb") as f:
                    b = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="results.json"')
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                return self.wfile.write(b)
            if path == "/api/template":
                with open(BUILTIN_CSV, "rb") as f:
                    head = b"\n".join(f.read().split(b"\n")[:6])
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="template.csv"')
                self.send_header("Content-Length", str(len(head)))
                self.end_headers()
                return self.wfile.write(head)
            if path.startswith("/figures/"):
                name = os.path.basename(path)
                fp = os.path.join(FIGURE_DIR, name)
                if not os.path.isfile(fp):
                    return self._json({"error": "figure not found"}, 404)
                with open(fp, "rb") as f:
                    return self._send(f.read(), "image/png")
            if path.endswith((".css", ".js", ".png", ".ico")):
                fp = os.path.join(PKG_DIR, "05_图表与看板", os.path.basename(path))
                if os.path.isfile(fp):
                    with open(fp, "rb") as f:
                        ctype = "text/css" if path.endswith(".css") else (
                            "application/javascript" if path.endswith(".js") else "image/png")
                        return self._send(f.read(), ctype + "; charset=utf-8")
            return self._json({"error": "not found", "path": path}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": str(e)}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/upload":
                content = body.get("content", "")
                if not content.strip():
                    raise ValueError("上传内容为空")
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(UPLOAD_CSV, "w", encoding="utf-8-sig", newline="") as f:
                    f.write(content)
                info = validate_csv(UPLOAD_CSV)
                return self._json({"ok": True, "filename": body.get("filename", "uploaded.csv"),
                                   "info": info})
            if path == "/api/run":
                if not RUN_LOCK.acquire(blocking=False):
                    return self._json({"error": "上一次运行尚未结束，请稍候"}, 429)
                try:
                    r = run_pipeline(body)
                    return self._json({"ok": True, "result": to_view(r)})
                finally:
                    RUN_LOCK.release()
            return self._json({"error": "not found"}, 404)
        except ValueError as e:
            return self._json({"ok": False, "error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            STATE["last_error"] = str(e)
            return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


def serve(port: int, open_browser: bool):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print("=" * 66)
    print("  选题7 · 学习者行为分析 —— 可验证成果交互验证平台")
    print("=" * 66)
    print(f"  访问地址 : {url}")
    print("  功能     : 上传任意 CSV / 改参数重跑 / 导出结果 / 查看图表")
    print("  停止服务 : 在本窗口按 Ctrl+C")
    print("=" * 66)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[已停止]")
        srv.shutdown()


# ---------------------------------------------------------------------------
# 命令行交互 / 批跑
# ---------------------------------------------------------------------------
def _ask(prompt: str, default, cast=str):
    raw = input(f"  {prompt} [当前={default}] > ").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except Exception:
        print("    输入无法解析，沿用当前值")
        return default


def cli_loop():
    p = dict(DEFAULTS)
    print("=" * 66)
    print("  命令行交互模式：直接回车沿用当前值；输入 q 退出")
    print("=" * 66)
    while True:
        print("\n当前参数：")
        for key in ("data_source", "k", "budget_ratio", "test_size",
                    "weight_mode", "ahp_alpha", "speed", "seed"):
            print(f"  {key:<15} = {p[key]}")
        print("  可改项：data_source / k / budget_ratio / test_size / weight_mode /")
        print("          ahp_alpha / speed / seed / custom_weights(6个逗号分隔)")
        print("  输入 mm 可重跑国内多形态层测评（multimodal.py）")
        key = input("\n请输入要修改的参数名（直接回车=运行，q=退出）> ").strip().lower()
        if key == "q":
            return
        if key == "mm":
            import subprocess
            print("\n[多形态重跑] 正在执行 multimodal.py …")
            try:
                subprocess.run([sys.executable, os.path.join(ROOT, "multimodal.py")],
                               cwd=ROOT, check=True)
                print("[完成] outputs/multimodal_results.json 已刷新；再次运行主流程即可同步显示。")
            except Exception as e:
                print(f"[失败] {type(e).__name__}: {e}")
            input("\n按回车继续……")
            continue
        if key == "":
            print("\n[运行中] 请稍候……")
            t0 = time.time()
            try:
                r = run_pipeline(p)
                s = summarize(r)
                print(f"\n[完成] 耗时 {time.time()-t0:.1f}s ｜ 数据源：{r['meta']['data_source']}")
                print(f"  样本 {s['data']['rows']} 条 / 认证率 {s['data']['cert_rate']:.2%}")
                print(f"  聚类 k={s['cluster']['k']} 轮廓系数={s['cluster']['silhouette']}")
                for row in s["cluster"]["profile"]:
                    print(f"     {row['name']:<8} n={row['n']:<5} 占比={row['share']:.1%} "
                          f"时长={row['hours']:.1f}h 达标率={row['cert']:.1%}")
                print(f"  预测 最佳模型={s['predict']['best_model']} AUC={s['predict']['best_auc']}")
                print(f"  评价 组合权重={s['evaluate']['combined']}")
                print(f"        TOPSIS 第1名={s['evaluate']['rank'][0]['course_name']} "
                      f"({s['evaluate']['rank'][0]['closeness']})")
                print(f"  决策 预算={s['decide']['budget']:.0f}元 最优={s['decide']['opt_gain']}人 "
                      f"经验={s['decide']['heur_gain']}人 普惠={s['decide']['uniform_gain']}人")
                mm = s.get("multimodal")
                if mm:
                    h = mm["headline"]
                    print(f"  多形态 样本 {h['n_total']:,} 条（MOOC / 微课 / 短视频 / 课堂对照）")
                    print(f"         形态效应 ΔAUC={h['form_effect_gain']:.4f} "
                          f"(LR 检验 p={h['form_effect_p']:.2g})，"
                          f"跨域 {h['transfer_mean']:.3f} < 域内 {h['indomain_mean']:.3f}")
                    print(f"         评价首位={h['topsis_top']} ｜ 最优干预 +{h['opt_gain']:.0f} 人 "
                          f"vs 经验 +{h['heur_gain']:.0f} 人")
                else:
                    print("  多形态 未生成（运行 python multimodal.py 后刷新）")
                print(f"  完整结果已写入 {os.path.relpath(LATEST_JSON, ROOT)}")
            except Exception as e:
                print(f"[失败] {type(e).__name__}: {e}")
            input("\n按回车继续……")
            continue
        if key not in p:
            print("  未知参数名")
            continue
        if key == "custom_weights":
            raw = input("  请输入 6 个权重（逗号分隔）> ").strip()
            try:
                vals = [float(x) for x in raw.split(",")]
                if len(vals) != 6:
                    raise ValueError
                p[key] = vals
                p["weight_mode"] = "custom"
                print("  已切换为自定义权重模式")
            except Exception:
                print("  格式错误，需 6 个数字")
            continue
        cast = {"k": int, "seed": int, "budget_ratio": float,
                "test_size": float, "ahp_alpha": float}.get(key, str)
        p[key] = _ask(key, p[key], cast)


def main(argv=None):
    ap = argparse.ArgumentParser(description="选题7 可验证成果交互验证平台")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--cli", action="store_true", help="命令行交互模式")
    ap.add_argument("--batch", action="store_true", help="一次性批跑并保存结果")
    ap.add_argument("--data", default="builtin")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--budget", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "results.json"))
    ap.add_argument("--multimodal", action="store_true",
                    help="重跑国内多形态层建模与跨平台测评（multimodal.py）")
    args = ap.parse_args(argv)

    if args.multimodal:
        import subprocess
        return subprocess.run([sys.executable, os.path.join(ROOT, "multimodal.py")],
                             cwd=ROOT, check=True).returncode
    if args.cli:
        return cli_loop()
    if args.batch:
        p = {**DEFAULTS, "data_source": args.data, "k": args.k,
             "budget_ratio": args.budget, "seed": args.seed}
        r = run_pipeline(p)
        out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in r.items() if k != "meta"},
                      f, ensure_ascii=False, indent=1, default=str)
        print(f"[OK] 基准结果已写入 {out}（耗时 {r['meta']['elapsed_sec']}s）")
        return
    serve(args.port, not args.no_browser)


if __name__ == "__main__":
    main()
