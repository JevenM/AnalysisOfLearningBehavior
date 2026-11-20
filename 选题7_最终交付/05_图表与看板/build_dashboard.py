# -*- coding: utf-8 -*-
"""把统一管线（六类真实数据集合并分析）的实跑结果内联进主看板模板，产出单文件离线 HTML。

数据来源（均真实实跑，落盘于 04_运行结果/）：
  - up_metrics.json              聚类/预测/早期预警/GM 趋势
  - up_mining_summary.json       SNA / LSA / 时间模式 / 关联规则 / 场景
  - up_association_summary.json  相关 Top / 投入分层提升
  - up_eval_summary.json         AHP+模糊质量分层 / SD 仿真
  - up_persona_recommendations.csv  四类画像（人数/占比/描述/内容建议）
  - up_optimization_suggestions.csv 优化建议
  - up_queues.csv                干预队列
  - profile_results.json         内容级画像（时段/兴趣/短视频人口/天池标签，国内数据源）
本脚本只做结构裁剪与传输，不改动任何实跑数字。
"""
import csv
import json
import math
import os
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "04_运行结果")


def _jload(name):
    return json.load(open(os.path.join(RES, name), encoding="utf-8"))


def _drop_nan(x):
    """NaN 不是合法 JSON（模板用 JSON.parse 解析会整体失败），递归替换为 None。"""
    if isinstance(x, dict):
        return {k: _drop_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_drop_nan(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _csv_rows(name):
    with open(os.path.join(RES, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


UM = _drop_nan(_jload("up_metrics.json"))
MI = _drop_nan(_jload("up_mining_summary.json"))
AS = _drop_nan(_jload("up_association_summary.json"))
EV = _drop_nan(_jload("up_eval_summary.json"))
P = _drop_nan(_jload("profile_results.json"))

# 学习者总数（up_learners.csv：1 行 = 1 名学习者，带表头）
with open(os.path.join(RES, "up_learners.csv"), encoding="utf-8") as f:
    N_LEARNERS = max(sum(1 for _ in f) - 1, 0)

# 六源构成：从 up_unified.csv 按来源列计数（列名兼容 source/data_source/source_repo）
sources = OrderedDict()
unified_path = os.path.join(RES, "up_unified.csv")
with open(unified_path, encoding="utf-8-sig") as f:
    rd = csv.reader(f)
    header = next(rd)
    col = next((i for i, c in enumerate(header) if c.lower() in ("source", "data_source", "source_repo", "src")), None)
    if col is not None:
        cnt = Counter(row[col] for row in rd if len(row) > col)
        sources = OrderedDict(sorted(cnt.items(), key=lambda kv: -kv[1]))

# 四类画像（带中文命名与一句话建议）
PERSONA_CN = {
    "At-Risk / Disengaged": "风险流失型",
    "Guided Learner": "引导依赖型",
    "Independent Deep Learner": "自主深度型",
    "Low-Engagement Strong-Performance Learner": "低投入高绩效型",
}
personas = [{
    "name": r["persona"], "cn": PERSONA_CN.get(r["persona"], r["persona"]),
    "n": int(r["n_students"]), "share": float(r["share_pct"]),
    "desc": r["description"], "reco": r["content_recommendation"],
} for r in _csv_rows("up_persona_recommendations.csv")]
personas.sort(key=lambda x: -x["n"])

# 干预队列统计
queue_cnt = Counter(r["queue"] for r in _csv_rows("up_queues.csv"))
QUEUE_CN = {"none": "持续观察", "enrichment": "拓展培养", "review": "教师复核", "support": "重点支持"}
queues = [{"name": QUEUE_CN.get(k, k), "n": n} for k, n in queue_cnt.most_common()]

# 优化建议（按策略类型分组）
suggestions = OrderedDict()
for r in _csv_rows("up_optimization_suggestions.csv"):
    suggestions.setdefault(r["策略类型"], []).append({"target": r["对象"], "advice": r["建议"]})

# 内容级画像（国内数据源：清华短视频 / 天池微课）
tod = P.get("time_of_day", {}).get("rows", [])
interest = P.get("interest", {})
sv = P.get("sv_profile", {})

insight = {
    "overview": {
        "n": N_LEARNERS, "sources": sources, "k": UM.get("k"),
        "sil": UM.get("silhouette"), "backend": UM.get("backend"),
    },
    "personas": personas,
    # 投入与结果
    "tier_lift": AS.get("tier_lift"),
    "corr_top": [{"label": d.get("indicator"), "r": d.get("pearson_r")}
                 for d in (AS.get("corr_top") or [])[:6]],
    # 节律与兴趣
    "time_patterns": MI.get("time_patterns"),
    "tod": tod,
    "cats": [{"name": c.get("cat"), "share": round(c.get("watch_share", 0) * 100, 1)}
             for c in (interest.get("cat_watch") or [])[:8]],
    "itypes": [{"name": t.get("group"), "share": round(t.get("share", 0) * 100, 1)}
               for t in (interest.get("interest_types") or [])],
    "city": [{"name": c.get("group"), "share": round(c.get("share", 0) * 100, 1),
              "finish": round(c.get("finish_ratio", 0) * 100, 1)}
             for c in (sv.get("city_level") or [])],
    "sv_users": sv.get("n_users"), "sv_cities": sv.get("n_cities"),
    # 行为序列与资源
    "lsa_patterns": MI.get("lsa_patterns"),
    "lsa_top": MI.get("lsa_top_transitions"),
    "apriori_top": (MI.get("apriori_top") or [])[:3],
    "short_only_pct": MI.get("short_only_pct"),
    "sna_top": MI.get("sna_top_resources"),
    # 预测与预警
    "cert_acc": UM.get("cert_acc"),
    "best_model": UM.get("best_model"),
    "auc_early": UM.get("rf_auc_early"), "auc_full": UM.get("rf_auc_full"),
    "top5_features": UM.get("top5_features"),
    "gm_trends": UM.get("gm_trends"),
    # 评价与行动
    "quality_levels": EV.get("quality_levels"),
    "quality_mean": EV.get("quality_mean"),
    "ahp_weights": EV.get("ahp_weights"), "ahp_cr": EV.get("ahp_cr"),
    "sd_results": EV.get("sd_results"), "sd_best": EV.get("sd_best"),
    "queues": queues,
    "suggestions": suggestions,
}

tpl = open(os.path.join(HERE, "dashboard.template.html"), encoding="utf-8").read()
data = json.dumps(insight, ensure_ascii=False, separators=(",", ":"))
assert "</script" not in data.lower(), "数据中含有会截断 script 的字符串"
out = tpl.replace("__DATA__", data)
assert "__DATA__" not in out
with open(os.path.join(HERE, "dashboard.html"), "w", encoding="utf-8") as f:
    f.write(out)
print(f"[OK] dashboard.html 已生成：{len(out):,} 字符，内联数据 {len(data):,} 字符")
print(f"     六源 {N_LEARNERS:,} 名学习者 / K={UM.get('k')} / 轮廓系数 {UM.get('silhouette'):.3f}（{UM.get('backend')} 后端）")
print(f"     画像 {len(personas)} 类 / 干预队列 {len(queues)} 类 / 优化建议 {sum(len(v) for v in suggestions.values())} 条 / 数据源 {len(sources)} 个")
