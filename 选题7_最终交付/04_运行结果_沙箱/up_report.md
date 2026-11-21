# 学习者画像分析报告（统一管线 up_）
生成时间: 2026-11-21 10:36:24

## 数据概览
- 学生总数: 8000，数据来源: {'real:harvardx_mitx_person_course': 8000}
- 聚类数 K = 3，Silhouette = 0.814，计算后端 = torch

## 主成分降维（PCA）
- 主成分命名: []
- 方差解释率: []

## 行为模式挖掘（SNA + LSA）
- SNA: PageRank Top 资源 = ['S542811', 'S703721', 'S703737', 'S542864', 'S542807']
- LSA 行为模式分布: {'刷课': 6298, '正常': 4830, '跳步学习': 229}
- LSA Top 显著转换: [{'transition': 'video→video', 'z': 427.68}, {'transition': 'resource→video', 'z': -283.1}, {'transition': 'video→resource', 'z': -281.02}, {'transition': 'resource→resource', 'z': 238.44}, {'transition': 'video→forum', 'z': -226.5}]

## 预测性能
- LogisticRegression 成功预测: acc=0.995, AUC=1.000
- RF 学业表现预测: 最终成绩 R²=nan, 完成率 R²=nan, 完成判定 acc=nan
- RandomForest 风险预测 5-fold AUC: 全特征=1.000, 早期特征=0.949
- 行为特征预测画像标签 acc = 0.995
- GM(1,1) 日活跃趋势: {}

## 学习质量评价（AHP + 模糊综合评判）
- AHP 权重: {'学习行为': 0.32, '学习效果': 0.558, '互动参与': 0.122}, 一致性 CR = 0.016
- 评分均值: 40.7, 等级分布: {'差': 7674, '中': 326}

## 学习者画像

## 扩展分析摘要

- 核心学习者群体: {'gender': '男', 'age_band': '26-30', 'education': '本科'}
- 时间行为模式分布: {'常规型学习': 4439, '夜间型学习': 3517, '集中式周末学习': 3309, '碎片化高频学习': 92}
- 关联规则: 论坛互动、读资源 → OULAD 课程资源 (support=0.319, conf=0.988, lift=2.8)
- 关联规则: OULAD 课程资源、论坛互动 → 读资源 (support=0.319, conf=0.998, lift=2.75)
- 关联规则: OULAD 课程资源、做测验 → 读资源 (support=0.084, conf=0.996, lift=2.74)
- 只看短视频跳过完整慕课的学生占比: 58.5%
- 场景行为差异: {'慕课长视频': {'完播率': 0.39, '倍速使用率': 0.424, '平均拖拽次数': nan}, '短视频': {'完播率': 0.348, '倍速使用率': 0.0, '平均拖拽次数': nan}}
- 与成绩最相关的行为因素: avg_quiz_score(r=1.0), active_days(r=0.694), total_clicks(r=0.692), resource_clicks(r=0.69), video_clicks(r=0.449)
- 投入分层提升幅度: 高投入组较低投入组 完成率 +41.0%, 成绩 +16.5 分
- SD 仿真最优策略: 加强资源供给（52 周投入水平 1.0)
- 优化建议: 7 条（详见 up_optimization_suggestions.csv）
                                   persona  n_students  share_pct            description      content_recommendation
                  Independent Deep Learner         125        1.6 高投入、深度探索内容、测验得分高、极少求助。 研究性阅读材料、开放性项目、同伴讨论论坛、挑战级测评。
Independent Deep Learner (High Engagement)         509        6.4 高投入、深度探索内容、测验得分高、极少求助。 研究性阅读材料、开放性项目、同伴讨论论坛、挑战级测评。
   Low-Engagement Weak-Performance Learner        7366       92.1                    NaN                按簇中心特征定制内容策略

## 干预队列统计

- none: 6395 人
- enrichment: 1605 人

## 需要教师复核的学生（review/support 队列示例，前 20）

Empty DataFrame
Columns: [student_id, persona, avg_quiz_score, queue, suggested_action, evidence]
Index: []