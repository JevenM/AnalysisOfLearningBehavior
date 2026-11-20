# -*- coding: utf-8 -*-
"""
up_adapters.py —— 统一管线数据层：多源真实公开数据集适配器（统一接口）

接口约定：每个适配器返回 (features_frames, clicks_df|None)
  features_frames: list[DataFrame]，含 student_id / source / 统一特征列
                   （ALL_FEATURES 的子集）+ 标签列与真实人口属性列
  clicks_df: 事件流（student_id/timestamp/action/resource_id/course_id/
             resource_type/completed/speed_used/seek_count），无事件流的源为 None

数据来源（均为真实公开数据，不生成任何虚拟/合成数据），按类别存放于
02_数据与数据集/raw/：
  uci/            UCI Student Performance（395+649 名中学生成绩单）
  thu_shortvideo/ 清华 FIB 短视频行为数据集（WWW 2025，本地子样本）
  tianchi/        阿里云天池高校慕课学习行为脱敏样本(2024–2025)
  harvardx/       HarvardX-MITx Person Course Dataset（edX 真实脱敏数据）
  assistments/    中科大 EduData 集成包（ASSISTments 2009 答题行为）
  oulad/          Open University Learning Analytics Dataset（OULAD，7 张原始表）
"""
from __future__ import annotations

import ast
import glob
import json
import os

import numpy as np
import pandas as pd

from common import DATA_DIR, RAW_DIR, SEED, log

MAX_CLICKS = 200000        # 事件流总量上限（SNA / LSA / GM(1,1) 的输入规模）
CLICKS_PER_USER = 30       # 每个用户最多抽取的事件数
_ADAPTER_CACHE: dict = {}

# 统一特征 schema（各数据源映射后对齐到这些列；缺失为 NaN）
CORE_FEATURES = [
    "total_clicks", "active_days", "avg_session_min", "forum_posts",
    "resource_clicks", "video_clicks", "quiz_attempts", "avg_quiz_score",
    "assignment_submit_lag", "help_requests", "login_freq_per_week",
]
EXTENDED_FEATURES = [
    "accuracy", "avg_response_time", "hesitation_rate", "engagement_score",
    "hint_usage", "success_rate", "skip_rate", "unique_days_ratio",
    "level_ups", "code_lines", "code_functions", "comment_ratio",
]
ALL_FEATURES = CORE_FEATURES + EXTENDED_FEATURES

# 各数据源字段 → 统一 schema 的别名映射（键统一小写）
COLUMN_ALIASES = {
    "student_id": "student_id", "userid": "student_id", "user_id": "student_id",
    "lms_clicks": "total_clicks", "clicks": "total_clicks",
    "study_hours": "study_hours", "attendance_rate": "attendance_rate",
    "quiz_score": "avg_quiz_score", "overall_score": "overall_score",
    "success_status": "success_status", "score": "avg_quiz_score",
    "accuracy": "accuracy", "avg_response_time": "avg_response_time",
    "response_time": "avg_response_time", "hesitation_rate": "hesitation_rate",
    "engagement_score": "engagement_score", "hint_usage": "hint_usage",
    "learner_type": "learner_type",
}

# LSA 行为编码（点击流事件类型）
LSA_ACTIONS = ["video", "resource", "quiz", "forum", "help"]

# OULAD vle 活动类型 → LSA 行为编码
OULAD_ACTION_MAP = {
    "forumng": "forum",
    "quiz": "quiz", "externalquiz": "quiz", "ouquiz": "quiz",
    "homepage": "resource", "oucontent": "resource", "resource": "resource",
    "subpage": "resource", "url": "resource", "wiki": "resource",
    "glossary": "resource", "dataplus": "resource", "htmlactivity": "resource",
    "page": "resource", "folder": "resource", "dualpane": "resource",
    "sharedpage": "resource", "repeatactivity": "resource",
    "ouelluminate": "resource", "outactivity": "resource",
    "questionnaire": "resource",
}
OULAD_EDU_MAP = {
    "No Formal quals": "高中及以下", "Lower Than A Level": "高中及以下",
    "A Level or Equivalent": "高中", "HE Qualification": "本科",
    "Post Graduate Qualification": "硕士及以上",
}


def get_source(fn, data_dir=None):
    """适配器结果缓存（特征表与事件流共用一次加载）。"""
    key = fn.__name__
    if key not in _ADAPTER_CACHE:
        _ADAPTER_CACHE[key] = fn(data_dir)
    return _ADAPTER_CACHE[key]


# ----------------------------------------------------------------------
# 1) UCI Student Performance
# ----------------------------------------------------------------------
def load_uci_student(data_dir: str = None):
    """UCI Student Performance（葡萄牙两所中学数学/葡萄牙语成绩单，395+649 名学生）。
    https://archive.ics.uci.edu/dataset/320/student+performance"""
    base = os.path.join(RAW_DIR, "uci")
    frames = []
    for name, tag in [("student-mat.csv", "mat"), ("student-por.csv", "por")]:
        path = os.path.join(base, name)
        if not os.path.exists(path):
            log(f"  [UCI Student Performance] 缺少 {name}，跳过")
            continue
        df = pd.read_csv(path, sep=";")
        study_map = {1: 1.5, 2: 3.5, 3: 7.5, 4: 12.0}   # 周学习时长档位 → 小时中值
        feat = pd.DataFrame({
            "student_id": [f"UCI-{tag}-{i}" for i in range(len(df))],
            "study_hours": df["studytime"].map(study_map).astype(float),
            "avg_quiz_score": df[["G1", "G2", "G3"]].mean(axis=1).astype(float),
            "final_grade": df["G3"].astype(float),
            "quiz_attempts": 3.0,                        # 学期三次正式测验 G1/G2/G3
            "attendance_rate": (100 - df["absences"] / 93 * 100).clip(0, 100),
            "help_requests": (df["schoolsup"] == "yes").astype(float),
            "success_status": np.where(df["G3"] >= 10, "pass", "fail"),
            "gender": df["sex"].map({"F": "女", "M": "男"}),
            "age_band": pd.cut(df["age"], [0, 17, 19, 22, 100],
                               labels=["≤17", "18-19", "20-22", "23+"]).astype(str),
            "region": df["address"].map({"U": "城市", "R": "农村"}),
            "education": "高中",
        })
        feat["source"] = "real:uci_student_performance"
        frames.append(feat)
        log(f"  [UCI Student Performance] 已加载 {name}  n={len(feat)}")
    return frames, None


# ----------------------------------------------------------------------
# 2) 清华短视频
# ----------------------------------------------------------------------
def load_thu_shortvideo(data_dir: str = None):
    """清华大学 FIB 实验室大规模短视频行为数据集（WWW 2025 发布）本地子样本。
    点击/停留/完播/点赞等行为记录，并利用类目表标记教育类内容占比。
    http://fi.ee.tsinghua.edu.cn/datasets/short-video-dataset"""
    base = os.path.join(RAW_DIR, "thu_shortvideo")
    path = os.path.join(base, "interaction_sampled.csv")
    if not os.path.exists(path):
        log("  [清华短视频] 未找到 interaction_sampled.csv，跳过")
        return [], None
    usecols = ["user_id", "pid", "category_id", "exposed_time", "watch_time",
               "duration", "cvm_like", "comment", "follow", "collect", "forward",
               "hate", "click", "p_date", "p_hour", "gender", "age"]
    df = pd.read_csv(path, usecols=usecols)
    edu_cats = set()
    cat_path = os.path.join(base, "categories_cn_en.csv")
    if os.path.exists(cat_path):
        cats = pd.read_csv(cat_path)
        edu = cats[cats["category_name_cn"].astype(str).str.contains(
            "教育|学习|知识|科普|技能", na=False)]
        edu_cats = set(edu["category_id"].tolist())
    df["is_edu"] = df["category_id"].isin(edu_cats)
    social = df[["cvm_like", "comment", "follow", "collect", "forward"]].any(axis=1)
    # 向量化完播率（每行 观看时长/视频时长，逐用户平均），避免 groupby.apply
    watch_ratio = (df["watch_time"] / df["duration"].clip(lower=1)).clip(0, 1)

    g = df.groupby("user_id")
    feat = pd.DataFrame({
        "student_id": [f"THUSV-{u}" for u in g.size().index],
        "total_clicks": g.size().values.astype(float),
        "active_days": g["p_date"].nunique().values.astype(float),
        "avg_session_min": (g["watch_time"].mean() / 60).values,
        "video_clicks": g["click"].sum().values.astype(float),
        "forum_posts": g["comment"].sum().values.astype(float),
        "engagement_score": social.groupby(df["user_id"]).mean().values.round(3),
        "completion_rate": watch_ratio.groupby(df["user_id"]).mean().values.round(3),
        "edu_video_ratio": g["is_edu"].mean().values.round(3),
        "gender": g["gender"].first().values,
        "age_band": pd.cut(pd.to_numeric(g["age"].first(), errors="coerce"),
                           [0, 17, 25, 40, 120],
                           labels=["≤17", "18-25", "26-40", "40+"]).astype(str).values,
    })
    feat["source"] = "real:thu_shortvideo"
    log(f"  [清华短视频] 已加载 interaction_sampled.csv  用户 n={len(feat)}, "
        f"行为记录 {len(df)} 条")

    # 事件流（每用户抽样，供 SNA/LSA/GM(1,1) 使用）
    ev = df.sort_values("exposed_time").groupby("user_id", group_keys=False) \
        .head(CLICKS_PER_USER)
    clicks = pd.DataFrame({
        "student_id": "THUSV-" + ev["user_id"].astype(str),
        "timestamp": pd.to_datetime(ev["p_date"].astype(str), format="%Y%m%d") \
            + pd.to_timedelta(ev["p_hour"], unit="h"),
        "action": "video",
        "resource_id": "V" + ev["pid"].astype(str),
        "course_id": "shortvideo",
        "resource_type": "short",
        "completed": (ev["watch_time"] >= 0.9 * ev["duration"].clip(lower=1)) \
            .astype(float).values,
        "speed_used": np.nan,
        "seek_count": np.nan,
    })
    return [feat], clicks


# ----------------------------------------------------------------------
# 3) 代码提交（data/ 通用扫描用）
# ----------------------------------------------------------------------
def code_to_features(subs: pd.DataFrame) -> pd.DataFrame:
    """代码文本 → 学生级静态特征（AST/token 统计，无需重型依赖）。"""
    rows = []
    for _, r in subs.iterrows():
        code = r["code"]
        lines = [ln for ln in code.splitlines() if ln.strip()]
        comments = [ln for ln in lines if ln.strip().startswith("#")]
        try:
            tree = ast.parse(code)
            n_funcs = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                          for node in ast.walk(tree))
            n_calls = sum(isinstance(node, ast.Call) for node in ast.walk(tree))
        except SyntaxError:
            n_funcs, n_calls = 0, 0
        rows.append({
            "student_id": r["student_id"], "code_lines": len(lines),
            "code_functions": n_funcs, "code_calls": n_calls,
            "comment_ratio": len(comments) / max(len(lines), 1),
            "avg_quiz_score": r["grade"],
            "source": "code_submissions",
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 4) 阿里云天池慕课
# ----------------------------------------------------------------------
def load_mooc_tianchi(data_dir: str = None):
    """阿里云天池《中国高校慕课课程多维画像与学习行为脱敏样本数据集(2024–2025)》。
    user_learning_behavior.xls：视频播放/测验/作业/论坛/资源下载等学习行为事件；
    courses.xls：课程维表（中国大学 MOOC 平台真实脱敏样本）。"""
    base = os.path.join(RAW_DIR, "tianchi")
    beh_path = os.path.join(base, "user_learning_behavior.xls")
    if not os.path.exists(beh_path):
        log("  [天池慕课] 未找到 user_learning_behavior.xls，跳过")
        return [], None
    df = pd.read_excel(beh_path)
    # 真实数据存在脏时间戳（如 3000 年），越界值置 NaT 后丢弃
    df["occur_time"] = pd.to_datetime(df["occur_time"], errors="coerce")
    df = df[df["occur_time"].notna()]
    df["date"] = df["occur_time"].dt.date
    action_map = {"video_play": "video", "quiz_submit": "quiz", "homework_submit": "quiz",
                  "forum_browse": "forum", "resource_download": "resource",
                  "course_eval": "help"}
    df["action"] = df["behavior_type"].map(action_map).fillna("resource")

    # 行为类型布尔列（向量化统计，避免 groupby.apply 逐组遍历）
    for t in ("video_play", "forum_browse", "resource_download", "course_eval"):
        df[f"is_{t}"] = df["behavior_type"].eq(t)
    df["is_quiz"] = df["behavior_type"].isin(["quiz_submit", "homework_submit"])
    progress = (df["progress_after_pct"] / 100).clip(0, 1)

    g = df.groupby("learner_id")
    score_mean = g["score"].mean()
    feat = pd.DataFrame({
        "student_id": [f"MOOC-{u}" for u in g.size().index],
        "total_clicks": g.size().values.astype(float),
        "active_days": g["date"].nunique().values.astype(float),
        "avg_session_min": (g["duration_sec"].mean() / 60).values,
        "video_clicks": g["is_video_play"].sum().values.astype(float),
        "quiz_attempts": g["is_quiz"].sum().values.astype(float),
        "avg_quiz_score": score_mean.values.round(1),
        "final_grade": score_mean.values.round(1),
        "forum_posts": g["is_forum_browse"].sum().values.astype(float),
        "resource_clicks": g["is_resource_download"].sum().values.astype(float),
        "help_requests": g["is_course_eval"].sum().values.astype(float),
        "completion_rate": progress.groupby(df["learner_id"]).mean().values.round(3),
        "success_rate": g["is_passed"].mean().values.round(3),
        "avg_response_time": g["duration_sec"].mean().values,
    })
    feat["source"] = "real:tianchi_mooc_2024_2025"
    log(f"  [天池慕课] 已加载 user_learning_behavior.xls  学习者 n={len(feat)}, "
        f"行为事件 {len(df)} 条")

    clicks = pd.DataFrame({
        "student_id": "MOOC-" + df["learner_id"].astype(str),
        "timestamp": df["occur_time"],
        "action": df["action"],
        "resource_id": df["resource_code"].astype(str),
        "course_id": df["course_id"].astype(str),
        "resource_type": np.where(df["behavior_type"] == "video_play", "mooc", None),
        "completed": (df["progress_after_pct"] >= 95).astype(float),
        "speed_used": pd.to_numeric(df["playback_speed"], errors="coerce"),
        "seek_count": np.nan,
    })
    return [feat], clicks


# ----------------------------------------------------------------------
# 5) HarvardX/MITx
# ----------------------------------------------------------------------
def load_harvardx_mitx(data_dir: str = None, max_users: int = 50000):
    """HarvardX-MITx Person Course Dataset（edX 2012-2013 年两校课程真实脱敏数据）。
    按用户聚合跨课程行为：事件数/活跃天数/视频/论坛/章节 + 结业标签与人口属性。"""
    base = os.path.join(RAW_DIR, "harvardx")
    path = os.path.join(base, "hmx_cleaned.csv")
    uid_col = "user_id"
    if not os.path.exists(path):
        path = os.path.join(base, "HMXPC13_DI_v2_5-14-14.csv")
        uid_col = "userid_DI"
    if not os.path.exists(path):
        log("  [HarvardX/MITx] 未找到数据文件，跳过")
        return [], None
    usecols = [uid_col, "course_id", "nevents", "ndays_act", "nplay_video",
               "nforum_posts", "nchapters", "grade", "certified", "viewed",
               "explored", "gender", "YoB", "education"]
    df = pd.read_csv(path, usecols=[c for c in usecols])
    df = df.rename(columns={uid_col: "user_id"})
    users = df["user_id"].unique()
    if len(users) > max_users:                      # 真实数据随机抽样，控制规模
        users = np.random.default_rng(SEED).choice(users, max_users, replace=False)
        df = df[df["user_id"].isin(users)]
        log(f"  [HarvardX/MITx] 用户量 {len(users)} 超上限，随机抽样 {max_users} 人")
    g = df.groupby("user_id")
    grade_max = g["grade"].max() * 100
    edu_map = {"Less than Secondary": "高中及以下", "Secondary": "高中及以下",
               "High School": "高中及以下", "Bachelor's": "本科", "Master's": "硕士及以上",
               "Doctorate": "硕士及以上"}
    feat = pd.DataFrame({
        "student_id": "EDX-" + g.size().index.astype(str),
        "total_clicks": g["nevents"].sum().values.astype(float),
        "active_days": g["ndays_act"].sum().values.astype(float),
        "video_clicks": g["nplay_video"].sum().values.astype(float),
        "forum_posts": g["nforum_posts"].sum().values.astype(float),
        "resource_clicks": g["nchapters"].sum().values.astype(float),
        "avg_quiz_score": grade_max.values.round(1),
        "final_grade": grade_max.values.round(1),
        "completion_rate": g[["viewed", "explored"]].max().mean(axis=1) \
            .values.round(3),
        "success_status": np.where(g["certified"].max().values == 1,
                                   "certified", "none"),
        "gender": g["gender"].first().map({"m": "男", "f": "女", "M": "男", "F": "女"}) \
            .values,
        "age_band": pd.cut(pd.to_numeric(g["YoB"].first(), errors="coerce"),
                           [0, 1975, 1985, 1995, 2004],
                           labels=["40+", "31-40", "26-30", "18-25"]).astype(str).values,
        "education": g["education"].first().map(edu_map).values,
    })
    feat["source"] = "real:harvardx_mitx_person_course"
    log(f"  [HarvardX/MITx] 已加载 {os.path.basename(path)}  用户 n={len(feat)}")
    return [feat], None


# ----------------------------------------------------------------------
# 6) EduData / ASSISTments
# ----------------------------------------------------------------------
def load_edudata_assistments(data_dir: str = None):
    """中科大 EduData 集成包托管的 ASSISTments 2009-2010 答题行为数据
    （skill_builder_data_corrected，约 34 万条中学数学答题记录）。
    https://github.com/bigdata-ustc/EduData"""
    path = os.path.join(RAW_DIR, "assistments",
                        "skill_builder_data_corrected.csv")
    if not os.path.exists(path):
        log("  [EduData/ASSISTments] 未找到 skill_builder_data_corrected.csv，跳过")
        return [], None
    try:
        df = pd.read_csv(path, usecols=["user_id", "correct", "ms_first_response",
                                        "hint_count", "order_id", "skill_id"])
    except UnicodeDecodeError:
        # 个别行含非 UTF-8 字节；所用列均为数值列，latin-1 可无损解析
        df = pd.read_csv(path, usecols=["user_id", "correct", "ms_first_response",
                                        "hint_count", "order_id", "skill_id"],
                         encoding="latin-1")
    df = df.dropna(subset=["user_id"])
    g = df.groupby("user_id")
    acc = g["correct"].mean()
    feat = pd.DataFrame({
        "student_id": "AST-" + g.size().index.astype(int).astype(str),
        "total_clicks": g.size().values.astype(float),
        "quiz_attempts": g.size().values.astype(float),
        "accuracy": acc.values.round(3),
        "success_rate": acc.values.round(3),
        "avg_quiz_score": (acc * 100).values.round(1),
        "avg_response_time": (g["ms_first_response"].mean() / 1000).values.round(1),
        "hint_usage": g["hint_count"].sum().values.astype(float),
        "help_requests": g["hint_count"].sum().values.astype(float),
        "success_status": np.where(acc.values >= 0.6, "pass", "fail"),
    })
    feat["source"] = "real:edudata_assistments_2009"
    log(f"  [EduData/ASSISTments] 已加载 skill_builder_data_corrected.csv  "
        f"学生 n={len(feat)}, 答题记录 {len(df)} 条")
    return [feat], None


# ----------------------------------------------------------------------
# 7) OULAD（新增第 6 类真实数据源）
# ----------------------------------------------------------------------
def load_oulad(data_dir: str = None):
    """Open University Learning Analytics Dataset（OULAD，CC-BY 4.0）。
    https://analyse.kmi.open.ac.uk/open_dataset

    7 张真实原始表（raw/oulad/）：courses / assessments / vle / studentInfo /
    studentRegistration / studentVle / studentAssessment。聚合口径：
      - studentVle（1065 万行 VLE 点击日志）→ 逐学生 total_clicks/active_days/
        论坛与测验互动次数（活动类型映射到 LSA 行为编码）；
      - studentAssessment → avg_quiz_score / final_grade（真实测评分数均值）；
      - studentInfo → final_result 真实标签 + 性别/地区/教育/年龄带真实人口属性。
    事件流时间戳由课程起始月（J=10 月、B=2 月，真实校历）+ 日期偏移推算。
    """
    base = os.path.join(RAW_DIR, "oulad")
    info_path = os.path.join(base, "studentInfo.csv")
    vle_path = os.path.join(base, "studentVle.csv")
    if not (os.path.exists(info_path) and os.path.exists(vle_path)):
        log("  [OULAD] 未找到 studentInfo.csv / studentVle.csv，跳过"
            "（下载见 02_数据与数据集/README.md）")
        return [], None
    info = pd.read_csv(info_path)
    info = info.rename(columns={"id_student": "student_id"})
    svle = pd.read_csv(vle_path, usecols=["code_module", "code_presentation",
                                          "id_student", "id_site", "date",
                                          "sum_click"])
    # 真实数据存在脏数值（镜像数据 sum_click 含非法字符）、负日期（开课前预览），容错处理
    svle["sum_click"] = pd.to_numeric(svle["sum_click"], errors="coerce")
    svle["date"] = pd.to_numeric(svle["date"], errors="coerce")
    svle = svle[svle["date"].notna() & (svle["date"] >= 0) & svle["sum_click"].notna()]

    # 活动类型映射（vle.csv 仅 ~6k 行，dict 映射代价远小于大表 merge）
    act_map = {}
    vle_path2 = os.path.join(base, "vle.csv")
    if os.path.exists(vle_path2):
        vle_dim = pd.read_csv(vle_path2, usecols=["id_site", "activity_type"])
        act_map = dict(zip(vle_dim["id_site"], vle_dim["activity_type"]))

    g = svle.groupby("id_student")
    acts = svle["id_site"].map(act_map).map(OULAD_ACTION_MAP).fillna("resource")
    svle["action"] = acts

    # 向量化聚合（论坛帖量/测验次数按行为编码列分组求和，避免 groupby.apply）
    forum_clicks = svle[svle["action"] == "forum"].groupby("id_student")["sum_click"].sum()
    quiz_counts = svle[svle["action"] == "quiz"].groupby("id_student").size()
    idx = g.size().index
    agg = pd.DataFrame({
        "total_clicks": g["sum_click"].sum().values.astype(float),
        "active_days": g["date"].nunique().values.astype(float),
        "n_events": g.size().values.astype(float),
        "forum_posts": forum_clicks.reindex(idx, fill_value=0).values.astype(float),
        "quiz_attempts": quiz_counts.reindex(idx, fill_value=0).values.astype(float),
    })
    agg["student_id"] = idx

    # 真实测评分数（studentAssessment：0-100 分；镜像数据 score 为字符串列，需容错转数值）
    sa_path = os.path.join(base, "studentAssessment.csv")
    if os.path.exists(sa_path):
        sa = pd.read_csv(sa_path, usecols=["id_student", "score"])
        sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
        sa = sa[sa["score"].notna()]
        score = sa.groupby("id_student")["score"].mean()
        agg["avg_quiz_score"] = agg["student_id"].map(score).round(1)
        agg["final_grade"] = agg["avg_quiz_score"]

    feat = agg.merge(info[["student_id", "code_module", "code_presentation",
                           "gender", "region", "highest_education", "age_band",
                           "studied_credits", "final_result"]].drop_duplicates(
                               subset="student_id"),
                     on="student_id", how="inner")
    result_map = {"Pass": "pass", "Distinction": "pass", "Fail": "fail",
                  "Withdrawn": "fail"}
    feat["success_status"] = feat["final_result"].map(result_map)
    # 完成率由真实结课结果派生（通过/优秀=1，未通过/退课=0）
    feat["completion_rate"] = feat["final_result"].map(
        {"Pass": 1.0, "Distinction": 1.0, "Fail": 0.0, "Withdrawn": 0.0})
    feat["gender"] = feat["gender"].map({"M": "男", "F": "女"})
    feat["education"] = feat["highest_education"].map(OULAD_EDU_MAP)
    feat["student_id"] = "OULAD-" + feat["student_id"].astype(int).astype(str)
    feat["source"] = "real:oulad"

    # 事件流：课程起始月锚点（J=10 月、B=2 月，真实校历）+ 日期偏移
    pres = svle["code_presentation"].astype(str)
    year = pd.to_numeric(pres.str[:4], errors="coerce")
    month = np.where(pres.str.endswith("J"), 10, 2)
    anchor = pd.to_datetime(dict(year=year, month=month, day=1), errors="coerce")
    svle["_ts"] = anchor + pd.to_timedelta(svle["date"], unit="D")
    ev = svle.dropna(subset=["_ts"]).sort_values(["id_student", "_ts"]) \
        .groupby("id_student", group_keys=False).head(CLICKS_PER_USER)
    clicks = pd.DataFrame({
        "student_id": "OULAD-" + ev["id_student"].astype(int).astype(str),
        "timestamp": ev["_ts"],
        "action": ev["action"],
        "resource_id": "S" + ev["id_site"].astype(str),
        "course_id": ev["code_module"].astype(str) + ev["code_presentation"].astype(str),
        "resource_type": "oulad",
        "completed": np.nan,
        "speed_used": np.nan,
        "seek_count": np.nan,
    })
    log(f"  [OULAD] 已加载 studentInfo/studentVle  学生 n={len(feat)}, "
        f"VLE 记录 {len(svle)} 条")
    return [feat], clicks


# ----------------------------------------------------------------------
# 事件流汇聚
# ----------------------------------------------------------------------
def _cap_clickstream(frames: list) -> pd.DataFrame:
    """合并各真实源事件流，超限时随机抽样（保持真实行为分布，不造数据）。"""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["student_id", "timestamp", "action"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["action"].isin(LSA_ACTIONS)].dropna(subset=["timestamp"])
    if len(df) > MAX_CLICKS:
        df = df.sample(MAX_CLICKS, random_state=SEED).sort_values("timestamp")
        log(f"  事件流超过 {MAX_CLICKS} 条，随机采样至 {len(df)} 条")
    return df.reset_index(drop=True)


def build_clickstream(data_dir: str = None) -> pd.DataFrame:
    """汇聚各真实数据源的带时间戳事件流（SNA / LSA / GM(1,1) 的输入）。
    可用来源：清华短视频、天池慕课、OULAD VLE 日志、DATA_DIR/clickstream.csv。"""
    frames = []
    for fn in (load_thu_shortvideo, load_mooc_tianchi, load_oulad):
        try:
            _, ev = get_source(fn)
            if ev is not None and not ev.empty:
                frames.append(ev)
                log(f"  事件流来源 {fn.__name__}: {len(ev)} 条")
        except Exception as e:
            log(f"  事件流适配器 {fn.__name__} 失败: {e}")
    path = os.path.join(DATA_DIR, "clickstream.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df = normalize_columns(df)
            if {"student_id", "timestamp", "action"}.issubset(df.columns):
                for col, default in [("resource_id", "R0"), ("course_id", "C0"),
                                     ("resource_type", None), ("completed", np.nan),
                                     ("speed_used", np.nan), ("seek_count", np.nan)]:
                    if col not in df.columns:
                        df[col] = default
                frames.append(df)
                log(f"  事件流来源 clickstream.csv: {len(df)} 条")
            else:
                log("  clickstream.csv 缺少必需列（student_id/timestamp/action），跳过")
        except Exception as e:
            log(f"  读取 clickstream.csv 失败: {e}")
    clicks = _cap_clickstream(frames)
    if clicks.empty:
        raise SystemExit("未发现带时间戳的真实事件流：请确认 "
                         "raw/thu_shortvideo/interaction_sampled.csv、天池慕课或 OULAD 数据存在")
    return clicks


def harmonize_demographics(uni: pd.DataFrame) -> pd.DataFrame:
    """人口属性仅来自真实数据列；缺失保持缺失，不再做任何合成或行为加权。"""
    if "gender" in uni.columns:
        g = uni["gender"].astype(str).str.strip().str.upper()
        uni["gender"] = g.map({"M": "男", "F": "女", "男": "男", "女": "女"})
    if "age_band" not in uni.columns and "age" in uni.columns:
        uni["age_band"] = pd.cut(pd.to_numeric(uni["age"], errors="coerce"),
                                 [0, 17, 25, 40, 120],
                                 labels=["≤17", "18-25", "26-40", "40+"]).astype(str)
    return uni


# ----------------------------------------------------------------------
# data/ 目录通用扫描（保留兼容）
# ----------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})
    return df


def load_real_data(data_dir: str = None) -> list[pd.DataFrame]:
    """扫描 data/ 根目录：*.csv/xlsx → 汇总表；*.jsonl → 事件流；submissions/**/*.py → 代码提交。"""
    scan_dir = data_dir or DATA_DIR
    frames = []
    for path in glob.glob(os.path.join(scan_dir, "*.csv")) + \
                glob.glob(os.path.join(scan_dir, "*.xlsx")):
        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
        except Exception as e:
            log(f"  跳过无法读取的文件 {path}: {e}")
            continue
        df = normalize_columns(df)
        if "student_id" not in df.columns:
            log(f"  跳过 {path}（缺少 student_id 列）")
            continue
        df["source"] = f"real:{os.path.basename(path)}"
        frames.append(df)
        log(f"  已加载真实汇总表 {os.path.basename(path)}  shape={df.shape}")
    for path in glob.glob(os.path.join(scan_dir, "*.jsonl")):
        try:
            recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        except Exception as e:
            log(f"  跳过无法读取的 JSONL {path}: {e}")
            continue
        events = []
        for rec in recs:
            d = rec.get("details", rec.get("metrics", {}))
            m = d.get("metrics", d) if isinstance(d, dict) else {}
            events.append({"student_id": rec.get("student_id"),
                           **{k: m.get(k) for k in ALL_FEATURES if k in m}})
        df = pd.DataFrame(events).dropna(how="all", subset=[c for c in ALL_FEATURES
                                                            if c != "student_id"])
        if not df.empty:
            df = normalize_columns(df)
            df["source"] = f"real:{os.path.basename(path)}"
            frames.append(df)
            log(f"  已加载真实交互日志 {os.path.basename(path)}  shape={df.shape}")
    code_files = glob.glob(os.path.join(scan_dir, "submissions", "**", "*.py"),
                           recursive=True)
    if code_files:
        rows = []
        for fp in code_files:
            sid = os.path.basename(os.path.dirname(fp)) or os.path.basename(fp)
            try:
                rows.append({"student_id": sid, "code": open(fp, encoding="utf-8").read()})
            except Exception:
                pass
        if rows:
            df = code_to_features(pd.DataFrame(rows))
            df["source"] = "real:code_submissions"
            frames.append(df)
            log(f"  已加载真实代码提交 {len(rows)} 个文件")
    return frames


# ----------------------------------------------------------------------
# 特征层：清洗 + 统一数据集构建
# ----------------------------------------------------------------------
def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """去重、IQR 离群截断、中值填充（借鉴 data-pipeline 的清洗模板）。"""
    df = df.drop_duplicates(subset="student_id").reset_index(drop=True)
    num_cols = [c for c in ALL_FEATURES if c in df.columns and
                pd.api.types.is_numeric_dtype(df[c])]
    for c in num_cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            df[c] = df[c].clip(lo, hi)
        df[c] = df[c].fillna(df[c].median())
    return df


def build_unified_dataset(data_dir: str = None) -> pd.DataFrame:
    """融合所有真实公开数据集 → 统一特征表（union 列，缺失由 clean_features 填充）。

    数据源（均为真实公开数据，不含任何合成数据）：
      1. UCI Student Performance（395+649 名中学生成绩单）
      2. 清华 FIB 短视频行为数据集（WWW 2025，本地子样本）
      3. 阿里云天池慕课脱敏样本(2024–2025)
      4. HarvardX/MITx Person Course Dataset
      5. 中科大 EduData（ASSISTments 2009）
      6. OULAD（Open University Learning Analytics Dataset）
      7. DATA_DIR 根目录通用扫描（CSV/XLSX 汇总表、JSONL 事件流、代码提交）
    """
    adapters = [load_uci_student, load_thu_shortvideo, load_mooc_tianchi,
                load_harvardx_mitx, load_edudata_assistments, load_oulad]
    frames = []
    for fn in adapters:
        try:
            src_frames, _ = get_source(fn)
            frames += src_frames
        except Exception as e:
            log(f"  适配器 {fn.__name__} 加载失败: {e}")
    # 仅当显式传入 data_dir 时才扫描额外的汇总表/事件流/代码提交；
    # 合并交付包中 02_数据与数据集 根层级为同事管线派生表（id 口径不同），
    # 默认扫描会与统一管线适配器重复计数，故默认不扫。
    if data_dir:
        frames += load_real_data(data_dir)
    if not frames:
        raise SystemExit("未发现任何真实数据集，请先按 README 将公开数据集放入 raw/")
    parts = []
    for df in frames:
        keep = ["student_id", "source"] + [c for c in ALL_FEATURES if c in df.columns]
        extra = [c for c in ["success_status", "overall_score", "grade",
                             "learner_type", "study_hours", "attendance_rate",
                             "code_calls", "completion_rate", "final_grade",
                             "gender", "age_band", "region", "education"]
                 if c in df.columns]
        parts.append(df[keep + extra])
    uni = pd.concat(parts, ignore_index=True, sort=False)
    # 标签列：优先使用数据集自带的成功/结业标签，否则由真实成绩中位数派生
    if "success_status" in uni.columns and uni["success_status"].notna().any():
        uni["success"] = uni["success_status"].astype(str).str.lower().isin(
            ["successful", "success", "certified", "pass", "1", "true"])
    else:
        score = uni["avg_quiz_score"]
        if "overall_score" in uni.columns:
            score = score.fillna(uni["overall_score"])
        uni["success"] = score >= np.nanmedian(score)
    # 以下两列仅对数据集未提供的来源按其真实行为派生代理值（非合成样本）
    if "completion_rate" not in uni.columns:
        eng = uni[["total_clicks", "active_days", "engagement_score"]].mean(axis=1)
        eng_n = (eng - eng.min()) / (eng.max() - eng.min() + 1e-9)
        score_n = (uni["avg_quiz_score"] / 100).clip(0, 1).fillna(0.5)
        uni["completion_rate"] = (0.6 * eng_n + 0.4 * score_n).clip(0, 1).round(3)
    if "final_grade" not in uni.columns:
        eng = uni[["total_clicks", "active_days", "engagement_score"]].mean(axis=1)
        eng_n = (eng - eng.min()) / (eng.max() - eng.min() + 1e-9)
        score_n = (uni["avg_quiz_score"] / 100).clip(0, 1)
        uni["final_grade"] = (100 * (0.7 * score_n.fillna(0.5) + 0.3 * eng_n)).round(1)
    uni["completion_rate"] = uni["completion_rate"].fillna(uni["completion_rate"].median())
    uni["final_grade"] = uni["final_grade"].fillna(uni["final_grade"].median())
    uni = clean_features(uni)
    # 人口属性只保留真实列，缺失保持缺失（不再合成）
    uni = harmonize_demographics(uni)
    # 基础行为代理指标（仅由真实字段派生；字段缺失时保持缺失，不再合成）
    if "study_hours" in uni.columns:            # UCI 自报周学习时长（真实字段）
        uni["study_hours_per_week"] = pd.to_numeric(uni["study_hours"], errors="coerce")
    elif "login_freq_per_week" in uni.columns:
        uni["study_hours_per_week"] = (uni["avg_session_min"] *
                                       uni["login_freq_per_week"] / 60).round(1)
    else:
        uni["study_hours_per_week"] = np.nan
    if "assignment_submit_lag" in uni.columns:
        uni["submission_rate"] = ((uni["assignment_submit_lag"] + 5) / 10).clip(0, 1).round(3)
    else:
        uni["submission_rate"] = np.nan
    log(f"\n统一数据集: {uni.shape[0]} 名学生 × {uni.shape[1]} 列, "
        f"来源分布: {uni['source'].value_counts().to_dict()}")
    return uni
