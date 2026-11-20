# -*- coding: utf-8 -*-
"""
evaluators.py —— AHP / 熵权法 / TOPSIS 唯一实现

原 analysis.py 与 multimodal.py 各有一份独立实现（数学完全相同、
接口与输出精度略有差异）。按合并决策「评价/决策以同事实现为准」，
抽到本模块作为唯一权威实现：

  - ahp(matrix, nd)            层次分析法：主特征向量定权 + CR 一致性检验
  - entropy_weight_matrix(X)   熵权法（矩阵口径，multimodal 使用）
  - entropy_weight_courses(df, criteria_fields)  熵权法（课程表口径，analysis 使用）

两处调用方均改为委托本模块，数值口径保持原样（nd 参数控制输出精度）。
"""
from __future__ import annotations

import numpy as np

# 一致性指标 RI 表（n=1..9）
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def ahp(matrix: np.ndarray) -> dict:
    """AHP 求权：主特征向量法 + 一致性比率 CR 检验。

    返回未取整的结果；各调用方按原输出口径自行取整
    （analysis.py 全部 4 位，multimodal.py 权重/λmax 4 位、ci/cr 5 位），
    保证合并前后 JSON 数值逐位一致。
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    vals, vecs = np.linalg.eig(matrix)
    k = int(np.argmax(vals.real))
    lam = float(vals[k].real)
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    ci = (lam - n) / (n - 1)
    cr = ci / RI[n] if RI[n] else 0.0
    return {"weights": [float(x) for x in w], "lambda_max": lam,
            "ci": float(ci), "cr": float(cr), "n": n}


def entropy_weight_matrix(X: np.ndarray) -> list:
    """熵权法（矩阵口径）：输入行为 × 指标矩阵（已极差标准化），输出权重列表。"""
    X = np.asarray(X, dtype=float)
    p = X / X.sum(axis=0, keepdims=True)
    p = np.clip(p, 1e-12, None)
    e = -(p * np.log(p)).sum(axis=0) / np.log(len(X))
    d = 1 - e
    return [round(float(v), 4) for v in d / d.sum()]


def entropy_weight_courses(df, criteria_fields: list, nd: int = 4) -> dict:
    """熵权法（课程表口径）：按 course_id 聚组取均值 → 极差标准化 → 熵权。

    criteria_fields 为指标列名列表（顺序与 CRITERIA 对齐）。
    """
    courses = sorted(df["course_id"].unique())
    X = []
    for c in courses:
        s = df[df["course_id"] == c]
        X.append([s[col].mean() for col in criteria_fields])
    X = np.array(X, dtype=float)
    # 极差标准化（全部为效益型指标）
    Z = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)
    Z = Z + 1e-6
    P = Z / Z.sum(0)
    e = -(P * np.log(P)).sum(0) / np.log(len(courses))
    d = 1 - e
    w = d / d.sum()
    return {"weights": w.round(nd).tolist(), "entropy": e.round(nd).tolist(),
            "courses": courses, "matrix": X.round(3).tolist()}


def topsis_closeness(X: np.ndarray, w) -> np.ndarray:
    """TOPSIS 核心矩阵计算：输入行为 × 指标矩阵与权重，输出贴近度向量。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    R = X / np.sqrt((X ** 2).sum(0))
    V = R * w
    best, worst = V.max(0), V.min(0)
    dp = np.sqrt(((V - best) ** 2).sum(1))
    dn = np.sqrt(((V - worst) ** 2).sum(1))
    return dn / (dp + dn)
