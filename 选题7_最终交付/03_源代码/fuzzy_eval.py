# -*- coding: utf-8 -*-
"""
fuzzy_eval.py —— 模糊综合评价补充实算（报告 7.2 节「系统评价」）

评语集 V={优,良,合格,待改进}，分值向量 (95,80,65,50)；
对跨形态评价的群体级五指标先做 min-max 归一化，再按梯形隶属函数生成 R；
采用加权平均型算子 M(·,+) 得 B=W∘R，综合得分 S=B·Vᵀ；
并与 TOPSIS 贴近度排序作 Spearman 等级相关，检验两种方法的一致性。

输入：04_运行结果/multimodal_results.json（evaluate.raw / evaluate.combined / evaluate.rank）
输出：04_运行结果/fuzzy_results.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(PKG, "04_运行结果")

V = np.array([95.0, 80.0, 65.0, 50.0])
LABELS = ["优", "良", "合格", "待改进"]


def trapezoid(x, a, b, c, d):
    """梯形隶属度：a→b 上升，b→c 保持 1，c→d 下降，其余为 0。"""
    x = np.asarray(x, float)
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return float((x - a) / (b - a))
    if b <= x <= c:
        return 1.0
    return float((d - x) / (d - c))


def membership(x):
    """四档梯形隶属函数，按行归一化使 Σr=1。"""
    r = np.array([
        [trapezoid(x, 78, 90, 100, 101),
         trapezoid(x, 58, 72, 82, 92),
         trapezoid(x, 38, 52, 65, 78),
         trapezoid(x, -1, 20, 38, 55)],
    ], dtype=float)
    s = r.sum()
    return r / s if s > 0 else np.full((1, 4), 0.25)


def main():
    M = json.load(open(os.path.join(OUT, "multimodal_results.json"), encoding="utf-8"))
    ev = M["evaluate"]
    raw = np.asarray(ev["raw"], float)
    w = np.asarray(ev["combined"], float)
    w = w / w.sum()
    units = ev["units"]
    criteria = ev["criteria"]

    lo, hi = raw.min(axis=0), raw.max(axis=0)
    scale = np.where(hi - lo > 0, hi - lo, 1.0)
    norm = (raw - lo) / scale * 100

    results = []
    scores = []
    for i, unit in enumerate(units):
        R = np.vstack([membership(v) for v in norm[i]])
        B = (w @ R).reshape(1, -1)
        B = B / B.sum()
        score = float((B @ V)[0])
        # 初稿口径：按综合得分 S 落入分值区间定级；主导隶属等级另列，避免双峰分布误判
        grade = "优" if score >= 85 else ("良" if score >= 75 else
                                        ("合格" if score >= 60 else "待改进"))
        results.append({
            "unit": unit, "B": [round(float(v), 4) for v in B[0]],
            "score": round(score, 2), "grade": grade,
            "dominant_grade": LABELS[int(np.argmax(B[0]))],
        })
        scores.append(score)

    topsis = {r["unit"]: r["closeness"] for r in ev["rank"]}
    tc = [topsis[r["unit"]] for r in results]
    rho, pval = spearmanr(scores, tc)

    out = {
        "method": "梯形隶属函数 + M(·,+) 加权平均算子",
        "comment_set": LABELS, "score_vector": V.tolist(),
        "criteria": criteria, "weights": [round(float(x), 4) for x in w],
        "normalization": "群体级五指标 min-max 归一到 0—100 后计算隶属度",
        "units": results,
        "spearman_with_topsis": {"rho": round(float(rho), 4),
                                 "p": round(float(pval), 4)},
        "note": "等级判定只作相对运营诊断；各形态标签语义不同，不作跨形态绝对值比较。",
    }
    with open(os.path.join(OUT, "fuzzy_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("weights:", out["weights"])
    for r in results:
        print(r["unit"], r["score"], r["grade"], r["B"])
    print("Spearman with TOPSIS:", out["spearman_with_topsis"])


if __name__ == "__main__":
    main()
