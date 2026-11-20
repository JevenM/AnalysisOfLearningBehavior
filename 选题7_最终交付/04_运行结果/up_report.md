# 学习者画像分析报告（统一管线 up_）
生成时间: 2026-11-20 22:02:57

## 数据概览
- 学生总数: 66644，数据来源: {'real:harvardx_mitx_person_course': 50000, 'real:thu_shortvideo': 6654, 'real:edudata_assistments_2009': 4217, 'real:oulad': 4014, 'real:uci_student_performance': 1044, 'real:tianchi_mooc_2024_2025': 715}
- 聚类数 K = 4，Silhouette = 0.552，计算后端 = torch

## 主成分降维（PCA）
- 主成分命名: []
- 方差解释率: []

## 行为模式挖掘（SNA + LSA）
- SNA: PageRank Top 资源 = ['S542811', 'S703721', 'S703737', 'S542864', 'S542807']
- LSA 行为模式分布: {'刷课': 6298, '正常': 4830, '跳步学习': 229}
- LSA Top 显著转换: [{'transition': 'video→video', 'z': 427.68}, {'transition': 'resource→video', 'z': -283.1}, {'transition': 'video→resource', 'z': -281.02}, {'transition': 'resource→resource', 'z': 238.44}, {'transition': 'video→forum', 'z': -226.5}]

## 预测性能
- LogisticRegression 成功预测: acc=0.954, AUC=0.983
- RF 学业表现预测: 最终成绩 R²=nan, 完成率 R²=nan, 完成判定 acc=nan
- RandomForest 风险预测 5-fold AUC: 全特征=0.978, 早期特征=0.882
- 行为特征预测画像标签 acc = 0.975
- GM(1,1) 日活跃趋势: {}

## 学习质量评价（AHP + 模糊综合评判）
- AHP 权重: {'学习行为': 0.32, '学习效果': 0.558, '互动参与': 0.122}, 一致性 CR = 0.016
- 评分均值: 45.5, 等级分布: {'差': 56693, '良': 5325, '中': 4235, '优': 391}

## 学习者画像

## 扩展分析摘要

- 核心学习者群体: {'gender': '男', 'age_band': '26-30', 'region': '城市', 'education': '本科'}
- 时间行为模式分布: {'常规型学习': 4439, '夜间型学习': 3517, '集中式周末学习': 3309, '碎片化高频学习': 92}
- 关联规则: 论坛互动、读资源 → OULAD 课程资源 (support=0.319, conf=0.988, lift=2.8)
- 关联规则: OULAD 课程资源、论坛互动 → 读资源 (support=0.319, conf=0.998, lift=2.75)
- 关联规则: OULAD 课程资源、做测验 → 读资源 (support=0.084, conf=0.996, lift=2.74)
- 只看短视频跳过完整慕课的学生占比: 58.5%
- 场景行为差异: {'慕课长视频': {'完播率': 0.39, '倍速使用率': 0.424, '平均拖拽次数': nan}, '短视频': {'完播率': 0.348, '倍速使用率': 0.0, '平均拖拽次数': nan}}
- 与成绩最相关的行为因素: active_days(r=0.715), avg_quiz_score(r=0.711), total_clicks(r=0.537), resource_clicks(r=0.35), forum_posts(r=0.313)
- 投入分层提升幅度: 高投入组较低投入组 完成率 +50.0%, 成绩 +23.7 分
- SD 仿真最优策略: 加强资源供给（52 周投入水平 1.0)
- 优化建议: 10 条（详见 up_optimization_suggestions.csv）
                                  persona  n_students  share_pct                description       content_recommendation
                     At-Risk / Disengaged       53281       79.9 点击与活跃天数极低、得分差、几乎无互动，流失风险高。  短微内容单元、游戏化激励、主动推送通知、导师早期介入。
                           Guided Learner        1165        1.7    中等投入、偏好视频、频繁求助、依赖结构化指导。 分步视频教程、例题演示、即时反馈测验、脚手架式学习路径。
                 Independent Deep Learner       11445       17.2     高投入、深度探索内容、测验得分高、极少求助。  研究性阅读材料、开放性项目、同伴讨论论坛、挑战级测评。
Low-Engagement Strong-Performance Learner         753        1.1                        NaN                 按簇中心特征定制内容策略

## 干预队列统计

- none: 45292 人
- enrichment: 11980 人
- review: 6468 人
- support: 2904 人

## 需要教师复核的学生（review/support 队列示例，前 20）

student_id                  persona  avg_quiz_score  queue suggested_action                      evidence
   THUSV-1     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
   THUSV-2     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
   THUSV-3     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
   THUSV-4     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
   THUSV-5     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
   THUSV-6     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
   THUSV-8     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
   THUSV-9     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
  THUSV-10     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
  THUSV-11     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
  THUSV-12     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
  THUSV-13     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
  THUSV-14     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
  THUSV-16     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
  THUSV-18     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
  THUSV-19 Independent Deep Learner             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
  THUSV-20     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
  THUSV-21     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
  THUSV-23     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)
  THUSV-24     At-Risk / Disengaged             0.0 review   teacher_review abnormal_behavior_pattern(刷课)