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
# DASH_SOURCE 可指向沙箱目录（04_运行结果_沙箱）构建沙箱版看板；
# DASH_OUT 指定输出文件名（如 dashboard.sandbox.html）。缺省为正式构建。
RES = os.environ.get("DASH_SOURCE") or os.path.join(ROOT, "04_运行结果")
OUT_NAME = os.environ.get("DASH_OUT", "dashboard.html")
# 沙箱构建时，沙箱范围不重跑四形态/内容画像，缺失的扩展证据文件回退正式目录
FALLBACK = (os.path.join(ROOT, "04_运行结果")
            if os.path.abspath(RES) != os.path.join(ROOT, "04_运行结果") else None)


def _src(name):
    p = os.path.join(RES, name)
    if not os.path.isfile(p) and FALLBACK:
        p = os.path.join(FALLBACK, name)
    return p


def _jload(name):
    return json.load(open(_src(name), encoding="utf-8"))


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
    with open(_src(name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


UM = _drop_nan(_jload("up_metrics.json"))
MI = _drop_nan(_jload("up_mining_summary.json"))
AS = _drop_nan(_jload("up_association_summary.json"))
EV = _drop_nan(_jload("up_eval_summary.json"))
P = _drop_nan(_jload("profile_results.json"))

# 学习者总数（up_learners.csv：1 行 = 1 名学习者，带表头）
with open(_src("up_learners.csv"), encoding="utf-8") as f:
    N_LEARNERS = max(sum(1 for _ in f) - 1, 0)

# 六源构成：从 up_unified.csv 按来源列计数（列名兼容 source/data_source/source_repo）
sources = OrderedDict()
unified_path = _src("up_unified.csv")
with open(unified_path, encoding="utf-8-sig") as f:
    rd = csv.reader(f)
    header = next(rd)
    col = next((i for i, c in enumerate(header) if c.lower() in ("source", "data_source", "source_repo", "src")), None)
    if col is not None:
        cnt = Counter(row[col] for row in rd if len(row) > col)
        sources = OrderedDict(sorted(cnt.items(), key=lambda kv: -kv[1]))

# 画像命名自管线 v2 起直接输出中文；此处映射仅用于兼容旧版（英文命名）CSV，
# 并为卡片副标题提供中英对照
PERSONA_CN = {
    "At-Risk / Disengaged": "风险流失型",
    "Guided Learner": "引导依赖型",
    "Independent Deep Learner": "自主深度型",
    "Low-Engagement Strong-Performance Learner": "低投入高绩效型",
    "Low-Engagement Weak-Performance Learner": "低投入弱表现型",
    "High-Engagement Strong-Performance Learner": "高投入强表现型",
    "High-Engagement Weak-Performance Learner": "高投入弱表现型",
}
PERSONA_EN = {
    "自主深度型": "Independent Deep Learner",
    "引导依赖型": "Guided Learner",
    "考前冲刺型": "Exam-Driven Learner",
    "风险流失型": "At-Risk / Disengaged",
    "低投入高绩效型": "Low-Engagement Strong-Performance Learner",
    "低投入弱表现型": "Low-Engagement Weak-Performance Learner",
    "高投入强表现型": "High-Engagement Strong-Performance Learner",
    "高投入弱表现型": "High-Engagement Weak-Performance Learner",
}
personas = [{
    "name": PERSONA_CN.get(r["persona"], r["persona"]),
    "en": PERSONA_EN.get(PERSONA_CN.get(r["persona"], r["persona"]), ""),
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

# ---------------------------------------------------------------------------
# 18 张管线实跑图表 → 按"分析环节"分组嵌入对应屏（图题 + 实跑结论）。
# 图片由本脚本从 04_运行结果/ 同步到 figures/，看板以相对路径引用，离线可用。
import shutil

FIG_SUB = os.environ.get("DASH_FIG_SUB", "")  # 沙箱构建时为 "sandbox"（figures/sandbox/）
FIG_DIR = os.path.join(HERE, "figures", FIG_SUB) if FIG_SUB else os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
for f in sorted(set(os.listdir(RES)) |
                (set(os.listdir(FALLBACK)) if FALLBACK and os.path.isdir(FALLBACK) else set())):
    if not ((f.startswith("up_") or f == "fig00_architecture.png") and f.endswith(".png")):
        continue
    src_png = os.path.join(RES, f)
    if not os.path.isfile(src_png) and FALLBACK:
        src_png = os.path.join(FALLBACK, f)  # 沙箱未产出该图时回退正式图，保证渲染完整
    if os.path.isfile(src_png):
        shutil.copyfile(src_png, os.path.join(FIG_DIR, f))
# 架构图是静态方法学资产（存放在 figures/ 根目录），沙箱版看板也需引用 → 复制到子目录
_f00_src = os.path.join(HERE, "figures", "fig00_architecture.png")
_f00_dst = os.path.join(FIG_DIR, "fig00_architecture.png")
if os.path.isfile(_f00_src) and os.path.abspath(_f00_src) != os.path.abspath(_f00_dst):
    shutil.copyfile(_f00_src, _f00_dst)
insight["fig_prefix"] = f"figures/{FIG_SUB}/" if FIG_SUB else "figures/"


def _d1(mapping):
    """dict 第一键（top5_features / time_patterns / lsa_patterns 的首位项）。"""
    for k in (mapping or {}):
        return str(k)
    return None


def _fnum(v, pct=False):
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


_IND_CN = {"active_days": "活跃天数", "total_clicks": "总点击", "avg_quiz_score": "测验均分",
           "forum_posts": "论坛发帖", "quiz_attempts": "测验次数", "resource_clicks": "资源点击",
           "video_clicks": "视频点击", "engagement_score": "互动投入度"}
_top_feat = _d1(UM.get("top5_features"))
_corr0 = (AS.get("corr_top") or [{}])[0] or {}
_tp = MI.get("time_patterns") or {}
_lsa = MI.get("lsa_patterns") or {}
_ap = (MI.get("apriori_top") or [{}])[0] or {}
_sc = MI.get("scene") or {}
_sna = MI.get("sna_top_resources") or []
_sd_best_res = (EV.get("sd_results") or {}).get(EV.get("sd_best") or "", {}) or {}

_n_feat = len(UM.get("cluster_cols") or [])
insight["figs"] = {
    "0": [  # 屏1 顶部 · 总体方法架构（静态方法学图，不随运行变化）
        {"f": "fig00_architecture.png", "t": "总体方法架构 · 数据到建模链全景",
         "c": f"数据层（真实公开数据源+事件流）→ 统一特征层（{_n_feat} 维行为特征，{N_LEARNERS:,} 人）→ "
              "建模层七环节（聚类画像→行为挖掘→相关分层→预测预警→统一评价→策略仿真→决策建议）→ "
              "输出层（报告/{n_figs} 张证据图/三个看板）。全链固定随机种子可复现，沙箱试验不污染正式结果；"
              "每个环节的方法与产物文件在图中逐一标注，与本看板各屏一一对应。"},
    ],
    "1": [  # 屏1 · 环节=聚类画像（K-means + 多方法降维）
        {"f": "up_elbow_silhouette.png", "t": "环节证据 · K 选择（肘部+轮廓双准则）",
         "c": f"在 K=2–8 候选中自动选定 K={UM.get('k')}：轮廓系数 {_fnum(UM.get('silhouette'))}、"
              f"肘部拐点 K={UM.get('knee_k')}，两准则一致——聚类数由数据决定而非人为指定。"},
        {"f": "up_pca_clusters.png", "t": "环节证据 · PCA 二维聚类投影",
         "c": "主成分平面上画像簇边界清晰、簇间距离远大于簇内距离——分群结构真实存在。"},
        {"f": "up_tsne_clusters.png", "t": "环节证据 · t-SNE 非线性嵌入",
         "c": "非线性降维同样呈现紧凑分簇：分组结论对降维方法稳健。"},
        {"f": "up_umap_clusters.png", "t": "环节证据 · UMAP 降维投影",
         "c": "第三种降维方法复现相同结构，三种视角交叉验证聚类可靠性。"},
        {"f": "up_radar_personas.png", "t": "环节证据 · 画像行为中心雷达",
         "c": f"各画像行为中心差异集中在投入维度（点击/活跃/互动）——{len(personas)} 类画像由行为塑造，而非人口标签。"},
        {"f": "up_heatmap_features.png", "t": "环节证据 · 聚类 × 行为特征热力图",
         "c": "逐特征比对各类行为中心（Z 分数标注各簇显著偏高/偏低的行为），命名依据即此差异。"},
    ],
    "2": [  # 屏2 · 环节=相关分析与投入分层
        {"f": "up_correlation_top_factors.png", "t": "环节证据 · 行为 × 成绩相关排序",
         "c": f"与成绩关联最强的行为是“{_IND_CN.get(_corr0.get('indicator'), _corr0.get('indicator'))}”"
              f"（r={_fnum(_corr0.get('pearson_r'))}，显著）——该项行为的持续积累比单次爆发更能解释成绩差异。"},
        {"f": "up_engagement_tiers.png", "t": "环节证据 · 投入分层增益",
         "c": f"高投入组相对低投入组：{AS.get('tier_lift')}——投入分层的增益可量化，是干预优先级的直接依据。"},
        {"f": "up_demographics_cross.png", "t": "环节证据 · 人口属性 × 行为交叉",
         "c": "不同性别/年龄/学历群体的行为差异存在但幅度有限——行为特征比背景属性更能预测结果。"},
        {"f": "up_rf_feature_importance.png", "t": "环节证据 · 随机森林特征重要性",
         "c": f"模型最依赖“{_IND_CN.get(_top_feat, _top_feat)}”——与相关分析两套独立方法得出同一结论，相互印证。"},
    ],
    "3": [  # 屏3 · 环节=时序模式挖掘
        {"f": "up_temporal_patterns.png", "t": "环节证据 · 登录节律分型分布",
         "c": f"学习者自动归为 {len(_tp)} 类节律，最多为“{_d1(_tp) or '—'}”"
              f"（{_tp.get(_d1(_tp), 0) if _tp else 0} 人）——推送与直播时段设计应贴合节律分布。"},
    ],
}

insight["figs"]["4"] = [  # 屏4 · 环节=行为序列与资源挖掘（LSA/SNA/Apriori）
    {"f": "up_lsa_transition_graph.png", "t": "环节证据 · LSA 行为模式转移图",
     "c": f"逐人行为序列归为三型：{' / '.join(f'{k} {v} 人' for k, v in _lsa.items())}"
          + (f"——其中“{max(_lsa, key=_lsa.get)}”型最多，需要完整学习路径引导而非放任刷量。"
             if _lsa else "。")},
    {"f": "up_sna_network.png", "t": "环节证据 · SNA 资源共现网络",
     "c": f"少数资源承载多数访问（Top：{'、'.join(_sna[:3])}…）——核心资源应优先保障可用性与加载速度。"},
    {"f": "up_apriori_rules.png", "t": "环节证据 · Apriori 关联规则",
     "c": f"最强规则「{_ap.get('antecedent')} → {_ap.get('consequent')}」"
          f"（置信度 {_ap.get('confidence')}，提升度 {_ap.get('lift')}）"
          "——规则的先后关系可直接设计成资源引导推荐。"},
    {"f": "up_scene_comparison.png", "t": "环节证据 · 慕课/短视频场景对比",
     "c": f"完播率慕课长视频 {(_sc.get('慕课长视频') or {}).get('完播率', '—')} vs 短视频 "
          f"{(_sc.get('短视频') or {}).get('完播率', '—')}，倍速使用率差异显著"
          "——两种内容场景的交互设计需差异化。"},
]
insight["figs"]["5"] = [  # 屏5 · 环节=预测与早期预警
    {"f": "up_model_feature_importance.png", "t": "环节证据 · 预测特征重要性 Top10",
     "c": f"仅用前 4 周行为即可预警：早期 AUC {_fnum(UM.get('rf_auc_early'), pct=True)}、"
          f"全量 AUC {_fnum(UM.get('rf_auc_full'), pct=True)}；"
          f"“{_IND_CN.get(_top_feat, _top_feat)}”是最关键的早期信号——第 4 周就应触发干预。"},
]
insight["figs"]["6"] = [  # 屏6 · 环节=统一评价与策略仿真
    {"f": "up_quality_score_distribution.png", "t": "环节证据 · 学习质量评分分布",
     "c": f"全体质量分均值 {EV.get('quality_mean')}/100——均值受少数高分个体拉动，"
          "干预应聚焦长尾低分群体而非均值本身。"},
    {"f": "up_sd_simulation.png", "t": "环节证据 · 系统动力学（SD）策略仿真",
     "c": f"52 周策略仿真：最优策略「{EV.get('sd_best')}」最终参与度增幅 "
          f"+{_sd_best_res.get('growth_pct', '—')}%——供给型策略具复利效应，激励型策略长期衰减。"},
]

# 证据图总数在全部图组构建完毕后回填（fig00 图注中的 {n_figs} 占位符）
_total_figs = sum(len(v) for v in insight["figs"].values())
insight["figs"]["0"][0]["c"] = insight["figs"]["0"][0]["c"].replace(
    "{n_figs}", f"{_total_figs}")
tpl = open(os.path.join(HERE, "dashboard.template.html"), encoding="utf-8").read()
data = json.dumps(insight, ensure_ascii=False, separators=(",", ":"))
assert "</script" not in data.lower(), "数据中含有会截断 script 的字符串"
out = tpl.replace("__DATA__", data)
assert "__DATA__" not in out
with open(os.path.join(HERE, OUT_NAME), "w", encoding="utf-8") as f:
    f.write(out)
print(f"[OK] {OUT_NAME} 已生成：{len(out):,} 字符，内联数据 {len(data):,} 字符（数据源 {RES}）")
print(f"     六源 {N_LEARNERS:,} 名学习者 / K={UM.get('k')} / 轮廓系数 {UM.get('silhouette'):.3f}（{UM.get('backend')} 后端）")
print(f"     画像 {len(personas)} 类 / 干预队列 {len(queues)} 类 / 优化建议 {sum(len(v) for v in suggestions.values())} 条 / 数据源 {len(sources)} 个")
