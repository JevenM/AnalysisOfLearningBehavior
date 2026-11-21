# -*- coding: utf-8 -*-
"""构建交互式「多形态学习者画像分析看板」。

数据来源均为真实实跑结果：
  - 02_数据与数据集/multimodal_all.csv      六源统一五维样本（形态内 z-score）
  - 04_运行结果/multimodal_results.json     各形态 K-means 画像、达标驱动系数、口径说明

本脚本只做数据裁剪与画像命名，不改动任何实跑数字；
画像群中心直接取自 mult imodal_results.json，散点按最近中心归属（等价 K-means 指派）。
产出单文件离线 HTML：05_图表与看板/persona_dashboard.html
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "02_数据与数据集")
RES = os.environ.get("DASH_SOURCE") or os.path.join(ROOT, "04_运行结果")
OUT = os.environ.get("DASH_OUT", "persona_dashboard.html")

_req = os.path.join(RES, "multimodal_results.json")
if not os.path.isfile(_req):
    # 沙箱构建（DASH_SOURCE 指向沙箱目录）且沙箱范围不跑四形态 → 无数据源，跳过
    print(f"SKIP {os.path.basename(RES)} 无 multimodal_results.json，"
          "跳过沙箱版形态画像看板（该看板仅反映四形态测评）")
    raise SystemExit(0)
mm = json.load(open(_req, encoding="utf-8"))
df = pd.read_csv(os.path.join(DATA, "multimodal_all.csv"))

DIMS = mm["meta"]["dims"]                       # engage persist complete interact context
DIM_CN = mm["meta"]["dim_cn"]
DIM_TAG_HI = {"engage": "高投入", "persist": "强持续", "complete": "高完成",
              "interact": "强互动", "context": "高情境"}
DIM_TAG_LO = {"engage": "低投入", "persist": "弱持续", "complete": "低完成",
              "interact": "弱互动", "context": "低情境"}

FORM_CN = {"MOOC": "慕课 MOOC", "微课": "微课", "短视频": "短视频", "课堂对照": "课堂对照"}
FORM_ORDER = ["MOOC", "微课", "短视频", "课堂对照"]

# 每形态散点抽样上限，保证单文件轻量且交互流畅
SAMPLE_CAP = {"MOOC": 1400, "微课": 723, "短视频": 1469, "课堂对照": 1044}

UNIT_SHORT = {
    "Open University (OU VLE)": "OU VLE",
    "edX (HarvardX/MITx)": "edX",
    "学堂在线 XuetangX": "学堂在线",
    "知识类短视频": "泛知识短视频",
    "葡萄牙语": "葡萄牙语",
    "数学": "数学",
}


def unit_label(form, unit):
    if form == "微课":
        return "章节微资源"
    return UNIT_SHORT.get(str(unit), str(unit)[:14])


def persona_name(dvals, label_rate, form_rate):
    """基于五维 z-score 签名生成中文画像名与一句话刻画（确定性规则，不引入新口径）。

    命名取「最强正向维度 + 最强负向维度」组合，保证每个画像群名称可区分，
    且直接对应其在形态内标准化空间中的真实位置。
    """
    order = sorted(dvals.items(), key=lambda kv: kv[1], reverse=True)
    hi = [k for k, v in order if v > 0.35]                       # 由高到低
    lo = [k for k, v in reversed(order) if v < -0.35]           # 由低到高
    if hi and lo:
        base = DIM_TAG_HI[hi[0]] + "·" + DIM_TAG_LO[lo[0]]
    elif hi:
        base = "·".join(DIM_TAG_HI[k] for k in hi[:2])
    elif lo:
        base = "·".join(DIM_TAG_LO[k] for k in lo[:2])
    else:
        base = "常态参与"
    name = base + "型"
    if label_rate >= form_rate * 1.35:
        tier = "高转化"
    elif label_rate <= form_rate * 0.45:
        tier = "低转化"
    else:
        tier = "中转化"
    # 描述：突出最强与最弱维度 + 达标率相对定位
    top_k, top_v = order[0]
    desc = f"{DIM_TAG_HI.get(top_k, '各项')}特征突出（z={top_v:+.2f}）" if top_v > 0.2 \
        else f"各维度接近本形态均值（最强 {DIM_TAG_HI.get(top_k, top_k)} z={top_v:+.2f}）"
    low_k, low_v = order[-1]
    if low_v < -0.35:
        desc += f"，{DIM_TAG_LO.get(low_k, low_k)}偏弱（z={low_v:+.2f}）"
    rel = "高于" if label_rate > form_rate else "低于"
    desc += f"。达标率 {label_rate*100:.1f}%，{rel}本形态总体 {form_rate*100:.1f}%。"
    return name, tier, desc


bundle = {
    "meta": {
        "seed": mm["meta"]["seed"],
        "dims": DIMS,
        "dim_cn": DIM_CN,
        "n_total": mm["meta"]["n_total"],
        "label_note": "四类形态标签语义不同：MOOC=取得认证/通过；微课=单元测验通过；"
                      "短视频=后2日知识类内容回访（行为持续性，非学习成效）；课堂对照=期末G3≥12。"
                      "因此跨形态只比较标准化五维画像，不混称完课率。",
    },
    "forms": {},
}

for fi, form in enumerate(FORM_ORDER):
    m = mm["models"][form]
    sub = df[df["form"] == form].copy()
    prof = m["cluster"]["profile"]
    centers = np.array([[p[DIM_CN[d]] for d in DIMS] for p in prof], dtype=float)
    form_rate = float(m["positive_rate"])

    personas = []
    for p in prof:
        dvals = {d: float(p[DIM_CN[d]]) for d in DIMS}
        name, tier, desc = persona_name(dvals, float(p["label_rate"]), form_rate)
        personas.append({
            "idx": int(p["cluster"]),
            "name": name,
            "tier": tier,
            "n": int(p["n"]),
            "share": round(int(p["n"]) / int(m["n"]), 4),
            "label_rate": float(p["label_rate"]),
            "dims": [round(dvals[d], 3) for d in DIMS],
            "desc": desc,
        })
    personas.sort(key=lambda x: -x["n"])

    # 分层抽样：按标签分层，保证散点里两类都能看到
    cap = SAMPLE_CAP[form]
    pos = sub[sub["label"] == 1]
    neg = sub[sub["label"] == 0]
    n_pos = min(len(pos), max(30, int(cap * form_rate)))
    n_neg = min(len(neg), cap - n_pos)
    sample = pd.concat([pos.sample(n_pos, random_state=mm["meta"]["seed"]),
                        neg.sample(n_neg, random_state=mm["meta"]["seed"] + 1)])
    sample = sample.sample(frac=1, random_state=mm["meta"]["seed"]).reset_index(drop=True)

    X = sample[DIMS].to_numpy(dtype=float)
    # 最近画像中心指派（与 K-means 预测规则一致）
    dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = dist.argmin(axis=1)
    points = []
    for i, row in sample.iterrows():
        points.append([
            round(float(row[DIMS[0]]), 3), round(float(row[DIMS[1]]), 3),
            round(float(row[DIMS[2]]), 3), round(float(row[DIMS[3]]), 3),
            round(float(row[DIMS[4]]), 3),
            int(row["label"]), int(assign[i]), unit_label(form, row["unit"]),
        ])

    coef = [{"dim": d, "cn": DIM_CN[d], "v": float(m["metrics"]["coef"][d])} for d in DIMS]

    units = []
    for u, grp in sub.groupby("unit", sort=False):
        units.append({"unit": unit_label(form, u), "n": int(len(grp)),
                      "rate": round(float(grp["label"].mean()), 4)})
    units.sort(key=lambda x: -x["n"])

    src = next((s for s in mm["sources"] if s["form"] == form), {})

    # 来源明细：不同形态的可比维度不同，按各自真实字段选择合适的细分口径
    if form == "短视频" and src.get("root_counts"):
        breakdown = {"title": "内容类目构成（原始样本）", "axis": "原始条数",
                     "items": [{"label": k, "value": int(v), "disp": f"{int(v):,}"}
                               for k, v in src["root_counts"].items()]}
    elif form == "微课":
        # 微课为单一课程 527 个资源单元、完备度为常量，改按投入强度分箱看达标率
        eng = sub["engage"]
        qs = eng.quantile([0, .25, .5, .75, 1]).tolist()
        labels = ["低投入 Q1", "中低 Q2", "中高 Q3", "高投入 Q4"]
        items = []
        for i in range(4):
            lo_v, hi_v = qs[i], qs[i + 1]
            mask = (eng >= lo_v) & (eng <= hi_v if i == 3 else eng < hi_v)
            g = sub[mask]
            if len(g):
                items.append({"label": labels[i], "value": round(float(g["label"].mean()), 4),
                              "disp": f"{g['label'].mean()*100:.1f}%"})
        breakdown = {"title": "按投入强度分层的达标率（微课）", "axis": "达标率", "items": items}
    else:
        breakdown = {"title": ("来源平台" if form == "MOOC" else "学科") + "分布与达标率",
                     "axis": "达标率",
                     "items": [{"label": u["unit"], "value": u["rate"], "disp": f"{u['rate']*100:.1f}%"}
                               for u in units]}

    bundle["forms"][form] = {
        "cn": FORM_CN[form],
        "n": int(m["n"]),
        "sampled": len(points),
        "positive_rate": form_rate,
        "label_name": src.get("label", ""),
        "auc": float(m["metrics"].get("best_cv_auc") or 0),
        "auc_model": m["metrics"].get("best_model", ""),
        "f1": float(m["metrics"].get("Logistic", {}).get("f1", 0)),
        "silhouette": float(m["cluster"]["silhouette"]),
        "k": int(m["cluster"]["best_k"]),
        "dataset": src.get("dataset", ""),
        "window": src.get("window", ""),
        "caveat": src.get("caveat", ""),
        "field_availability": src.get("field_availability", {}),
        "units": units,
        "breakdown": breakdown,
        "personas": personas,
        "coef": coef,
        "points": points,
    }

bundle["compare"] = {
    "headline": mm["headline"],
    "platform": mm["platform_effect"],
    "transfer": {"mean_transfer": mm["transfer"]["mean_transfer_auc"],
                 "mean_indomain": mm["transfer"]["mean_indomain_auc"]},
    "robustness": mm["robustness"],
    "note": "各形态标签语义不同，AUC 与达标率不可直接横向比较；" 
            "此处对比用于观察形态间行为结构差异，平台内口径为本形态主要证据。"
}

data = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
assert "</script" not in data.lower()
tpl_path = os.path.join(HERE, "persona_dashboard.template.html")
tpl = open(tpl_path, encoding="utf-8").read()
out = tpl.replace("__PERSONA_DATA__", data)
assert "__PERSONA_DATA__" not in out
out_path = os.path.join(HERE, OUT)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(out)
print(f"[OK] {OUT} 已生成：{len(out):,} 字符，内联数据 {len(data):,} 字符（数据源 {RES}）")
for form in FORM_ORDER:
    b = bundle["forms"][form]
    print(f"     {form:<6} 总体 {b['n']:>5} 条 / 画像 {b['k']} 类 / 散点 {b['sampled']:>4} / AUC {b['auc']:.4f}")
