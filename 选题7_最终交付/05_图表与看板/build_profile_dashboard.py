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
RES = os.environ.get("DASH_SOURCE") or os.path.join(ROOT, "04_运行结果")
TPL = os.path.join(HERE, "profile_dashboard.template.html")
OUT = os.path.join(HERE, os.environ.get("DASH_OUT", "profile_dashboard.html"))

_req = os.path.join(RES, "profile_results.json")
if not os.path.isfile(_req):
    # 沙箱构建（DASH_SOURCE 指向沙箱目录）且沙箱范围不跑内容画像 → 无数据源，跳过
    print(f"SKIP {os.path.basename(RES)} 无 profile_results.json，"
          "跳过沙箱版内容画像看板（该看板仅反映内容级画像）")
    raise SystemExit(0)
P = json.load(open(_req, encoding="utf-8"))
html = open(TPL, encoding="utf-8").read()
assert "__PROFILE_DATA__" in html, "模板缺少数据占位符"
html = html.replace("__PROFILE_DATA__", json.dumps(P, ensure_ascii=False))
open(OUT, "w", encoding="utf-8").write(html)
print("PROFILE_DASHBOARD_OK", OUT, len(html), "chars")
