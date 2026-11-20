# -*- coding: utf-8 -*-
"""
forecast.py —— 系统预测模块（课件第 10—11 讲方法链）

严格按「先识别序列特征，再选模型」的顺序实跑：
  1) 序列特征识别：滞后 1 阶自相关（±1.96/√n 阈值）+ 逆序数 u 检验（趋势）；
  2) 一次指数平滑 SES（α 由训练段 one-step MAE 网格寻优，适用于平稳序列）；
  3) 二次指数平滑 Holt（α/β 网格寻优，适用于含趋势序列）；
  4) GM(1,1) 灰色模型（累加生成 + 后验差检验 C/P）；
  5) 朴素法与训练均值作为对照基准；
  6) 早期预警时点识别：2/4/6 周累计行为 + 背景属性 → 5 折 CV AUC/F1。

数据源：02_数据与数据集/weekly_activity.csv、early_warning.csv
        （由 build_forecast_data.py 从 OULAD 公开数据派生）
输出：04_运行结果/forecast_results.json 与 05_图表与看板/figures/fig15、fig16
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 20260101
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(PKG, "02_数据与数据集")
OUT = os.path.join(PKG, "04_运行结果")
FIG = os.path.join(PKG, "05_图表与看板", "figures")
TRAIN_WEEKS = 8
TEST_WEEKS = 4


def _metrics(y_true, y_pred):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = yt - yp
    return {
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "mape": round(float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-9)) * 100), 2),
    }


def _series_features(y):
    """滞后 1 阶自相关 + 逆序数 u 检验，给出模型选择的经验依据。"""
    y = np.asarray(y, float)
    n = len(y)
    ym = y.mean()
    denom = float(np.sum((y - ym) ** 2))
    r1 = float(np.sum((y[:-1] - ym) * (y[1:] - ym)) / denom) if denom > 0 else 0.0
    acf_limit = 1.96 / np.sqrt(n)
    inv = int(sum(1 for i in range(n) for j in range(i + 1, n) if y[j] > y[i]))
    u_stat = (inv - n * (n - 1) / 4) / np.sqrt(n * (n - 1) * (2 * n + 5) / 72)
    cv = float(y.std(ddof=1) / y.mean() * 100)
    trend = abs(u_stat) > 1.96
    return {
        "n": n, "r1": round(r1, 4), "acf_limit": round(float(acf_limit), 4),
        "acf_significant": bool(abs(r1) > acf_limit),
        "inversions": inv, "u_stat": round(float(u_stat), 4),
        "trend_significant": bool(trend), "cv_percent": round(cv, 2),
        "judgement": ("序列存在显著趋势，优先二次指数平滑 / GM(1,1)"
                      if trend else "序列无显著趋势，优先一次指数平滑"),
    }


def _ses(y, h, alpha):
    y = np.asarray(y, float)
    l = y[0]
    one = []
    for t in range(1, len(y)):
        one.append(l)
        l = alpha * y[t] + (1 - alpha) * l
    return np.array(one), np.full(h, l), {"alpha": round(float(alpha), 3)}


def _holt(y, h, alpha, beta):
    y = np.asarray(y, float)
    l, b = y[0], y[1] - y[0]
    one = []
    for t in range(1, len(y)):
        one.append(l + b)
        prev = l
        l = alpha * y[t] + (1 - alpha) * (l + b)
        b = beta * (l - prev) + (1 - beta) * b
    return np.array(one), np.array([l + (i + 1) * b for i in range(h)]), \
        {"alpha": round(float(alpha), 3), "beta": round(float(beta), 3)}


def _gm11(y, h):
    """GM(1,1)：x0 累加生成，最小二乘估计 a,b，后验差检验 C/P。"""
    x0 = np.asarray(y, float)
    n = len(x0)
    x1 = np.cumsum(x0)
    z = (x1[:-1] + x1[1:]) / 2
    B = np.column_stack([-z, np.ones(n - 1)])
    a, b = np.linalg.lstsq(B, x0[1:], rcond=None)[0]
    total = n + h
    fitted = np.empty(total)
    fitted[0] = x0[0]
    for k in range(1, total):
        fitted[k] = (x0[0] - b / a) * np.exp(-a * k) + b / a
    pred = np.diff(fitted)                      # 长度 n+h-1，对应 k=1..n+h-1
    fitted_train = np.concatenate([x0[:1], pred[:n - 1]])   # k=0 处按首值还原
    resid = x0 - fitted_train
    c = float(resid.std(ddof=1) / x0.std(ddof=1))
    p = float(np.mean(np.abs(resid - resid.mean()) < 0.6745 * x0.std(ddof=1)))
    grade = "好" if c < 0.35 and p >= 0.95 else ("合格" if c < 0.5 and p >= 0.8 else "勉强/不合格")
    return fitted_train, pred[n - 1:n + h - 1], {
        "a": round(float(a), 5), "b": round(float(b), 4),
        "C": round(c, 4), "P": round(p, 4), "grade": grade,
        "develop_coef": round(float(-a), 4),
    }


def fit_series(y, h=TEST_WEEKS):
    y = np.asarray(y, float)
    train, test = y[:len(y) - h], y[len(y) - h:]
    feat = _series_features(train)

    # SES：训练段 one-step MAE 最小
    best_ses = None
    for a in np.arange(0.05, 1.0, 0.05):
        one, fut, par = _ses(train, h, a)
        m = float(np.mean(np.abs(train[1:] - one)))
        if best_ses is None or m < best_ses[0]:
            best_ses = (m, a, fut, par)
    _, a_ses, fut_ses, par_ses = best_ses

    # Holt：训练段 one-step MAE 最小
    best_holt = None
    for a in np.arange(0.1, 1.0, 0.1):
        for bb in np.arange(0.1, 1.0, 0.1):
            one, fut, par = _holt(train, h, a, bb)
            m = float(np.mean(np.abs(train[1:] - one)))
            if best_holt is None or m < best_holt[0]:
                best_holt = (m, a, bb, fut, par)
    _, a_h, b_h, fut_holt, par_holt = best_holt

    _, fut_gm, par_gm = _gm11(train, h)
    fut_naive = np.full(h, train[-1])
    fut_mean = np.full(h, train.mean())

    preds = {
        "SES一次指数平滑": fut_ses,
        "Holt二次指数平滑": fut_holt,
        "GM(1,1)灰色模型": fut_gm,
        "朴素法（上期值）": fut_naive,
        "训练均值": fut_mean,
    }
    pars = {
        "SES一次指数平滑": par_ses,
        "Holt二次指数平滑": par_holt,
        "GM(1,1)灰色模型": par_gm,
        "朴素法（上期值）": {},
        "训练均值": {},
    }
    models = {}
    naive_mae = _metrics(test, fut_naive)["mae"]
    for name, pred in preds.items():
        m = _metrics(test, pred)
        m["vs_naive_improve_percent"] = round((naive_mae - m["mae"]) / naive_mae * 100, 2)
        models[name] = {"params": pars[name], "metrics": m,
                        "forecast": [round(float(v), 2) for v in pred]}
    recommended = "Holt二次指数平滑" if feat["trend_significant"] else "SES一次指数平滑"
    # 实跑精度若显著劣于另一平滑法，按 MAE 选择并说明
    better = min(["SES一次指数平滑", "Holt二次指数平滑"],
                 key=lambda k: models[k]["metrics"]["mae"])
    return {
        "features": feat, "train": [round(float(v), 2) for v in train],
        "test": [round(float(v), 2) for v in test], "models": models,
        "recommended_by_feature": recommended,
        "best_by_mae": better, "naive_mae": naive_mae,
    }


def fit_early_warning():
    p = os.path.join(DATA, "early_warning.csv")
    df = pd.read_csv(p)
    cat = ["gender", "region", "education", "age_band"]
    Xbase = df[["w2_clicks", "w2_days", "w4_clicks", "w4_days",
                "w6_clicks", "w6_days", "prev_attempts", "credits"]].copy()
    Xcat = pd.get_dummies(df[cat], drop_first=True)
    y = df["label"].astype(int).values
    windows = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for w in ("w2", "w4", "w6"):
        cols = [f"{w}_clicks", f"{w}_days"]
        X = pd.concat([Xbase[cols + ["prev_attempts", "credits"]], Xcat], axis=1)
        pipe_inputs = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        auc = cross_val_score(clf, pipe_inputs, y, cv=cv, scoring="roc_auc")
        f1 = cross_val_score(clf, pipe_inputs, y, cv=cv, scoring="f1")
        Xtr, Xte, ytr, yte = train_test_split(pipe_inputs, y, test_size=0.25,
                                              stratify=y, random_state=SEED)
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        pr = clf.predict_proba(Xte)[:, 1]
        windows[w] = {
            "weeks": int(w[1:]), "features": cols + ["prev_attempts", "credits"] + cat,
            "auc_mean": round(float(auc.mean()), 4),
            "auc_std": round(float(auc.std()), 4),
            "f1_mean": round(float(f1.mean()), 4),
            "holdout_auc": round(float(roc_auc_score(yte, pr)), 4),
            "holdout_f1": round(float(f1_score(yte, yp)), 4),
        }
    best = max(windows, key=lambda k: windows[k]["auc_mean"])
    return {"n": int(len(df)), "positive_rate": round(float(y.mean()), 4),
            "windows": windows, "best_window": best,
            "earliest_usable": "w2" if windows["w2"]["auc_mean"] >= windows["w6"]["auc_mean"] - 0.03 else "w4",
            "note": "标签为最终 Pass/Distinction（OULAD 真实结果）；累计行为仅使用观察窗口内的日志，不存在未来信息泄漏。"}


def make_figures(res):
    os.makedirs(FIG, exist_ok=True)
    s = res["series"]
    train, test = s["train"], s["test"]
    weeks = list(range(len(train))) + list(range(len(train), len(train) + len(test)))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(range(len(train)), train, "o-", color="#1f5a99", label="训练期实际")
    ax[0].plot(range(len(train), len(train) + len(test)), test, "o-",
               color="#d1495b", label="测试期实际")
    ax[0].axvline(len(train) - 0.5, ls="--", color="#888")
    ax[0].set_title("OULAD 周人均点击序列（训练 8 周 / 测试 4 周）")
    ax[0].set_xlabel("课程周次"); ax[0].set_ylabel("人均周点击数")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    for name, color in (("SES一次指数平滑", "#2a9d8f"),
                        ("Holt二次指数平滑", "#e9a23b"),
                        ("GM(1,1)灰色模型", "#7a5195"),
                        ("朴素法（上期值）", "#777777")):
        fut = s["models"][name]["forecast"]
        ax[1].plot(range(len(train), len(train) + len(test)), fut, "--o",
                   color=color, label=name, ms=4)
    ax[1].plot(range(len(train), len(train) + len(test)), test, "o-",
               color="#222", lw=2, label="测试期实际")
    ax[1].set_title("多模型 4 周外推预测对比")
    ax[1].set_xlabel("课程周次"); ax[1].set_ylabel("人均周点击数")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig15_forecast.png"), dpi=160)
    plt.close(fig)

    ew = res["early_warning"]["windows"]
    ks = ["w2", "w4", "w6"]
    xs = [ew[k]["weeks"] for k in ks]
    aucs = [ew[k]["auc_mean"] for k in ks]
    stds = [ew[k]["auc_std"] for k in ks]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(xs, aucs, yerr=stds, fmt="o-", color="#1f5a99", capsize=4, lw=2)
    for x, a in zip(xs, aucs):
        ax.annotate(f"{a:.3f}", (x, a), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xticks(xs); ax.set_xlabel("观察窗口（周）")
    ax.set_ylabel("5 折 CV AUC（均值±标准差）")
    ax.set_title("早期预警可用时点识别")
    ax.set_ylim(min(aucs) - 0.06, min(1.0, max(aucs) + 0.08))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig16_earlywarning.png"), dpi=160)
    plt.close(fig)


def main():
    weekly = pd.read_csv(os.path.join(DATA, "weekly_activity.csv"))
    ov = weekly[weekly["code_presentation"] == "ALL"].sort_values("week")
    y = ov["clicks_per_active"].values[:TRAIN_WEEKS + TEST_WEEKS]
    res = {
        "meta": {
            "seed": SEED,
            "source": "OULAD 公开数据集 studentVle/studentInfo（2013J+2014J）",
            "series": "全体学习者人均周点击数 clicks_per_active",
            "train_weeks": TRAIN_WEEKS, "test_weeks": TEST_WEEKS,
            "derived_by": "build_forecast_data.py",
        },
        "series": fit_series(y),
        "early_warning": fit_early_warning(),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "forecast_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    make_figures(res)
    s = res["series"]
    print("features:", s["features"])
    for k, v in s["models"].items():
        print(k, v["params"], v["metrics"])
    print("early warning:", {k: (v["auc_mean"], v["f1_mean"])
                             for k, v in res["early_warning"]["windows"].items()})


if __name__ == "__main__":
    main()
