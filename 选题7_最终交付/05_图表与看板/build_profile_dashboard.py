# -*- coding: utf-8 -*-
"""构建「四维学习者画像交互看板」profile_dashboard.html。

数据源（全部真实、来自 04_运行结果/profile_results.json，由 profile_modeling.py 实跑）：
  - 地区差异、专业背景、学习习惯、兴趣爱好 四个维度
本脚本只做结构裁剪与传输，不改动任何实跑数字。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "04_运行结果")
TPL = os.path.join(HERE, "profile_dashboard.template.html")
OUT = os.path.join(HERE, "profile_dashboard.html")

P = json.load(open(os.path.join(RES, "profile_results.json"), encoding="utf-8"))
html = open(TPL, encoding="utf-8").read()
assert "__PROFILE_DATA__" in html, "模板缺少数据占位符"
html = html.replace("__PROFILE_DATA__", json.dumps(P, ensure_ascii=False))
open(OUT, "w", encoding="utf-8").write(html)
print("PROFILE_DASHBOARD_OK", OUT, len(html), "chars")
