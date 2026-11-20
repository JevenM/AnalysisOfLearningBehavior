# -*- coding: utf-8 -*-
"""
demo_runs.py —— 可验证成果的「三组改参重跑」实跑与数字留痕。

依次以基准 / 改权重 / 改 K+预算运行 app.run_pipeline，
把每次关键结果写入 04_运行结果/demo_runs/，供报告 8.4 节与答辩演示引用；
结束后把 results_latest.json 恢复为基准结果。
"""
from __future__ import annotations

import json
import os
import shutil

import app

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(PKG, "04_运行结果")
DEMO = os.path.join(OUT, "demo_runs")
LATEST = os.path.join(OUT, "results_latest.json")

CASES = [
    ("A_基准", {"k": 4, "budget_ratio": 0.45, "ahp_alpha": 0.5}),
    ("B_改权重", {"k": 4, "budget_ratio": 0.45, "ahp_alpha": 0.8}),
    ("C_改K与预算", {"k": 3, "budget_ratio": 0.60, "ahp_alpha": 0.5}),
]


def digest(r):
    cl, pr, ev, dc = r["cluster"], r["predict"], r["evaluate"], r["decide"]
    best = max(pr["metrics"].items(), key=lambda kv: kv[1]["auc"])
    return {
        "params": {k: r["meta"]["params"][k] for k in
                   ("seed", "k", "budget_ratio", "ahp_alpha", "test_size", "speed")},
        "data_source": r["meta"]["data_source"],
        "clean_rows": r["clean_log"]["clean_rows"],
        "cluster": {"k": cl["k"], "silhouette": cl["silhouette"], "ari": cl["ari"],
                    "archetypes": [x.get("name") for x in cl["profile_records"]]},
        "predict": {"best_model": best[0], "best_auc": best[1]["auc"],
                    "f1": best[1].get("f1")},
        "evaluate": {"top": ev["rank"][0]["course_name"],
                     "closeness": ev["rank"][0]["closeness"],
                     "ahp_cr": ev["ahp"]["cr"], "combined": ev["combined"]},
        "decide": {"budget": dc["budget"], "opt_gain": dc["opt_gain"],
                   "heur_gain": dc["heur_gain"], "uniform_gain": dc["uniform_gain"],
                   "opt_cost": dc["opt_cost"], "chosen": len(dc["chosen"]),
                   "status": dc["status"]},
        "elapsed_sec": r["meta"]["elapsed_sec"],
    }


def main():
    os.makedirs(DEMO, exist_ok=True)
    backup = None
    if os.path.exists(LATEST):
        with open(LATEST, "rb") as f:
            backup = f.read()
    summaries = []
    for name, patch in CASES:
        r = app.run_pipeline(patch)
        d = digest(r)
        d["case"] = name
        with open(os.path.join(DEMO, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        summaries.append(d)
        print(name, "k=", d["cluster"]["k"], "sil=", d["cluster"]["silhouette"],
              "auc=", d["predict"]["best_auc"], "top=", d["evaluate"]["top"],
              "gain=", d["decide"]["opt_gain"])
    with open(os.path.join(DEMO, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=1)
    if backup is not None:
        with open(LATEST, "wb") as f:
            f.write(backup)
    print("[OK] 三组改参结果已留痕，results_latest.json 已恢复为基准。")


if __name__ == "__main__":
    main()
