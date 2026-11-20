# -*- coding: utf-8 -*-
"""把 analysis.py 实跑的 results.json 精简后内联进看板模板，产出单文件离线 HTML。

数据来源（均真实实跑，落盘于 04_运行结果/）：
  - results.json              同事 OULAD 主链（清洗/聚类/预测/评价/决策）
  - profile_results.json      四维画像（地区/背景/习惯/兴趣）
  - forecast_results.json     GM(1,1) 与早期预警
  - fuzzy_results.json        模糊综合评价
  - up_*.json（合并新增）     统一管线：六源真实数据集融合分析（聚类/挖掘/关联/预测/评价）
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
R = json.load(open(os.path.join(ROOT, "04_运行结果", "results.json"), encoding="utf-8"))

roc = R["predict"]["roc"]
step = max(1, len(roc["fpr"]) // 120)

bench = {
    "clean_log": R["clean_log"],
    "quality": R["quality"],
    "corr": R["corr"],
    "cluster": {
        "k": R["cluster"]["k"],
        "silhouette": R["cluster"]["silhouette"],
        "ari": R["cluster"]["ari"],
        "scan": R["cluster"]["scan"],
        "profile": R["cluster"]["profile_records"],
    },
    "predict": {
        "metrics": R["predict"]["metrics"],
        "importance": R["predict"]["importance"],
        "odds": R["predict"]["odds"],
        "roc": {"fpr": roc["fpr"][::step], "tpr": roc["tpr"][::step]},
    },
    "evaluate": R["evaluate"],
    "decide": {k: v for k, v in R["decide"].items() if k in
               ("budget", "opt_gain", "heur_gain", "uniform_gain", "opt_cost",
                "chosen", "status", "segments", "uniform_note")},
    "sweep": R["sweep"],
    "ism": {k: R["ism"][k] for k in ("levels", "names", "micmac")},
}

# 国内内容级证据链：多形态融合建模与跨平台测评（由 multimodal.py 实跑，支撑第 3、5—8 章）
mm = R.get("multimodal")
if mm:
    bench["multimodal"] = {
        "meta": mm["meta"],
        "headline": mm["headline"],
        "sources": mm["sources"],
        "models": {f: {"n": v["n"], "positive_rate": v["positive_rate"],
                       "best_cv_auc": v["metrics"].get("best_cv_auc"),
                       "best_model": v["metrics"].get("best_model"),
                       "k": v["cluster"]["best_k"], "silhouette": v["cluster"]["silhouette"]}
                   for f, v in mm["models"].items()},
        "transfer": {"pairs": mm["transfer"]["pairs"],
                     "mean_transfer_auc": mm["transfer"]["mean_transfer_auc"],
                     "mean_indomain_auc": mm["transfer"]["mean_indomain_auc"]},
        "distribution": {"mean_w1": mm["distribution"]["mean_w1"],
                         "mean_psi": mm["distribution"]["mean_psi"]},
        "form_effect": {k: mm["form_effect"][k] for k in
                        ("global_auc", "with_form_auc", "auc_gain", "lr_p", "separated_mean_auc")},
        "platform_effect": mm["platform_effect"],
        "evaluate": {"rank": mm["evaluate"]["rank"], "combined": mm["evaluate"]["combined"],
                     "ahp": mm["evaluate"]["ahp"]},
        "decide": {k: mm["decide"][k] for k in
                   ("budget", "opt_gain", "heur_gain", "uniform_gain", "chosen")},
    }

# 第 5、6 章：系统预测与模糊综合评价（由 forecast.py / fuzzy_eval.py 实跑）
for fname, key in (("forecast_results.json", "forecast"), ("fuzzy_results.json", "fuzzy")):
    p = os.path.join(ROOT, "04_运行结果", fname)
    if os.path.isfile(p):
        bench[key] = json.load(open(p, encoding="utf-8"))


# ---------------------------------------------------------------- 统一管线数据块（合并新增）
# 六源真实数据集融合分析结果（up_metrics / up_mining / up_association / up_eval），
# 以"人话叙事"口径精简后进 INSIGHT；统一管线未跑（--skip-unified）时整块缺失、看板自动降级。
def _drop_nan(x):
    """NaN 不是合法 JSON（模板用 JSON.parse 解析会整体失败），递归替换为 None。"""
    if isinstance(x, dict):
        return {k: _drop_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_drop_nan(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _load_up():
    uni = {}
    for fname, key in (("up_metrics.json", "metrics"),
                       ("up_mining_summary.json", "mining"),
                       ("up_association_summary.json", "association"),
                       ("up_eval_summary.json", "eval")):
        p = os.path.join(ROOT, "04_运行结果", fname)
        if os.path.isfile(p):
            uni[key] = _drop_nan(json.load(open(p, encoding="utf-8")))
    return uni


def _count_learners():
    """统计统一管线逐学习者画像表行数（无表头，1 行 = 1 名学习者）。"""
    p = os.path.join(ROOT, "04_运行结果", "up_learners.csv")
    if not os.path.isfile(p):
        return 0
    with open(p, encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


# ---------------------------------------------------------------- 人话叙事数据块
# 面向非专业读者：只取"结论级"字段，每张卡片附一句话解读。
def _load(name):
    return json.load(open(os.path.join(ROOT, "04_运行结果", name), encoding="utf-8"))


P = _load("profile_results.json")
FC = _load("forecast_results.json")
R = _load("results.json")

# 四类主画像：人数 / 占比 / 通过率 / 活跃天 / 首次参与滞后 / 一句话
_persona_desc = {
    "社交深耕型": "几乎每天都学，常在讨论区交流，期末基本都能通过。",
    "自主达标型": "自己安排节奏、稳定投入，多数能通过。",
    "应试突击型": "平时来得少，临近考核才集中用力，大约一半能过。",
    "低投入游离型": "很少参与、很晚才出现，几乎都没通过，需要尽早发现。",
}
_personas = []
for pr in sorted(R["cluster"]["profile_records"], key=lambda x: -x["cert"]):
    nm = pr["name"]
    _personas.append({
        "name": nm, "n": pr["n"], "share": round(pr["share"] * 100, 1),
        "pass": round(pr["cert"] * 100, 1), "days": round(pr["days"], 1),
        "visits": round(pr["visits"]), "lag": round(pr["lag"], 1),
        "desc": _persona_desc.get(nm, ""),
    })

# 五种学习习惯：节奏 / 占比 / 通过率 / 首次滞后 / 活跃天
_habit_desc = {
    "规律深耕型": "开课就来、几乎天天学", "稳定参与型": "节奏稳定、持续跟进",
    "晚到追赶型": "来得晚，但一直坚持", "碎片专注型": "来得不算勤，但注意力集中",
    "浅触即离型": "很晚才出现，很快又离开",
}
_habits = [{
    "name": h["group"], "share": round(h["share"] * 100, 1),
    "pass": round(h["pass_rate"] * 100, 1), "lag": round(h["lag"], 1),
    "days": round(h["active_days"], 1), "desc": _habit_desc.get(h["group"], ""),
} for h in sorted(P["habit"]["types"], key=lambda x: -x["pass_rate"])]

# 兴趣：短视频类目观看占比（Top10）+ 三类兴趣人群
_cats = [{"name": c["cat"], "share": round(c["watch_share"] * 100, 1)}
         for c in P["interest"]["cat_watch"][:10]]
_itypes = [{"name": t["group"], "share": round(t["share"] * 100, 1),
            "know": round(t["know_share"] * 100, 1), "ent": round(t["ent_share"] * 100, 1)}
           for t in P["interest"]["interest_types"]]

# 观看时段
_tod = [{"name": r["group"], "share": round(r["share"] * 100, 1)}
        for r in P["habit"]["time_of_day"]["rows"]]

# 短视频用户属性：城市等级 / 性别 / 年龄
_city = [{"name": c["group"], "share": round(c["share"] * 100, 1),
          "finish": round(c["finish_ratio"] * 100, 1), "interact": round(c["interact_rate"] * 100, 1)}
         for c in P["sv_profile"]["city_level"]]
_gender = [{"name": g["group"], "share": round(g["share"] * 100, 1),
            "finish": round(g["finish_ratio"] * 100, 1)} for g in P["sv_profile"]["gender"]]
_age = [{"name": a["group"], "share": round(a["share"] * 100, 1),
         "watch": round(a["avg_watch_sec"], 1)} for a in P["sv_profile"]["age_band"]]

# 学历 / 地区：作为"背景与地区差异"人话卡片
_edu = [{"name": e.get("education") or e.get("group") or e.get("name"),
         "n": e.get("n"), "share": round((e.get("share") or 0) * 100, 1),
         "pass": round((e.get("cert_rate") or e.get("pass_rate") or 0) * 100, 1)}
        for e in P["background"]["education"]]
_region = [{"name": r.get("region") or r.get("group") or r.get("name"),
            "n": r.get("n"), "pass": round((r.get("cert_rate") or r.get("pass_rate") or 0) * 100, 1)}
           for r in P["region"]["rows"]]

# 早期预警：2/4/6 周，用"提前几周 + 判断准确率"表达（不用 AUC 术语）
_warn = []
for wk, v in FC["early_warning"]["windows"].items():
    _warn.append({"weeks": v["weeks"], "acc": round(v["auc_mean"] * 100, 1),
                  "f1": round(v["f1_mean"] * 100, 1)})
_warn.sort(key=lambda x: x["weeks"])

# 决策：三种做法各能多帮多少人
_dec = R["decide"]
_decision = {
    "budget_wan": round(_dec["budget"] / 10000, 1),
    "smart": round(_dec["opt_gain"], 0),
    "weak": round(_dec["heur_gain"], 0),
    "even": round(_dec["uniform_gain"], 0),
    "chosen": [{"seg": c["seg"], "level": c["level"], "gain": round(c["gain"], 0)}
               for c in _dec.get("chosen", [])],
}

# 来源与差异强度（给非专业读者的"哪些差异真实存在"结论）
_sig = {
    "region_v": P["region"]["chi2"]["cramers_v"],
    "habit_v": P["habit"]["chi2"]["cramers_v"],
    "edu_v": P["background"]["chi2_edu"]["cramers_v"],
}

insight = {
    "overview": {
        "records": R["clean_log"]["clean_rows"], "learners": R["clean_log"]["learners"],
        "courses": R["clean_log"]["courses"],
        "sv_users": P["sv_profile"]["n_users"], "sv_cities": P["sv_profile"]["n_cities"],
        # 多形态样本总数：取自 multimodal_results.json 实跑结果（不硬编码）
        "mm_samples": R.get("multimodal", {}).get("headline", {}).get("n_total", 0),
        "persona_k": R["cluster"]["k"],
    },
    "personas": _personas, "habits": _habits, "cats": _cats, "itypes": _itypes,
    "tod": _tod, "city": _city, "gender": _gender, "age": _age,
    "edu": _edu, "region": _region, "warn": _warn, "decision": _decision, "sig": _sig,
}

# 统一管线叙事块（合并新增）：只取"结论级"字段；未跑统一管线时整块缺失，JS 端自动隐藏第 8 节
UNI = _load_up()
if UNI:
    _m = UNI.get("metrics", {})
    _mi = UNI.get("mining", {})
    _as = UNI.get("association", {})
    _ev = UNI.get("eval", {})
    insight["unified"] = {
        # 只保留看板第 8 屏实际渲染的字段，不堆未使用数据
        "n": _count_learners(),
        "k": _m.get("k"), "sil": _m.get("silhouette"), "backend": _m.get("backend"),
        "cert_acc": _m.get("cert_acc"),
        "auc_early": _m.get("rf_auc_early"), "auc_full": _m.get("rf_auc_full"),
        "top5_features": _m.get("top5_features"),
        "gm_trends": _m.get("gm_trends"),
        "lsa_patterns": _mi.get("lsa_patterns"), "time_patterns": _mi.get("time_patterns"),
        "apriori_top": (_mi.get("apriori_top") or [])[:3],
        "tier_lift": _as.get("tier_lift"),
        "ahp_weights": _ev.get("ahp_weights"), "ahp_cr": _ev.get("ahp_cr"),
        "quality_levels": _ev.get("quality_levels"),
        "sd_results": _ev.get("sd_results"), "sd_best": _ev.get("sd_best"),
    }

tpl = open(os.path.join(HERE, "dashboard.template.html"), encoding="utf-8").read()
data = json.dumps(bench, ensure_ascii=False, separators=(",", ":"))
idata = json.dumps(insight, ensure_ascii=False, separators=(",", ":"))
assert "</script" not in data.lower() and "</script" not in idata.lower(), "数据中含有会截断 script 的字符串"
out = tpl.replace("__BENCH_DATA__", data).replace("__INSIGHT_DATA__", idata)
assert "__BENCH_DATA__" not in out and "__INSIGHT_DATA__" not in out
with open(os.path.join(HERE, "dashboard.html"), "w", encoding="utf-8") as f:
    f.write(out)
print(f"[OK] dashboard.html 已生成：{len(out):,} 字符，内联基准数据 {len(data):,} / 叙事数据 {len(idata):,} 字符")
print(f"     画像 {len(_personas)} 类、习惯 {len(_habits)} 型、地区 {len(_region)} 个、预警窗口 {len(_warn)} 个")
if insight.get("unified"):
    u = insight["unified"]
    print(f"     统一管线叙事块已内联：六源 {u['n']:,} 名学习者 / K={u['k']} / 轮廓系数 {u['sil']:.3f}（{u['backend']} 后端）")
else:
    print("     统一管线产物缺失（未跑 up_ 阶段），看板自动降级为仅主链数据")
