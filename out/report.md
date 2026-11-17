# 学习者画像分析报告
生成时间: 2026-11-17 15:32:07

## 数据概览
- 学生总数: 62630，数据来源: {'real:harvardx_mitx_person_course': 50000, 'real:thu_shortvideo': 6654, 'real:edudata_assistments_2009': 4217, 'real:uci_student_performance': 1044, 'real:tianchi_mooc_2024_2025': 715}
- 聚类数 K = 5，Silhouette = 0.579

## 主成分降维（PCA）
- 主成分命名: ['学习投入度', '互动活跃度', '学业表现', '学业表现(次要)']
- 方差解释率: ['24.3%', '22.4%', '13.7%', '11.3%']

## 行为模式挖掘（SNA + LSA）
- SNA: PageRank Top 资源 = ['V97998', 'V32880', 'V10164', 'V33286', 'V98799']
- LSA 行为模式分布: {'无点击流数据': 55261, '刷课': 6479, '正常': 776, '跳步学习': 114}
- LSA Top 显著转换: ['video→video(z=190.49)', 'video→quiz(z=-165.15)', 'quiz→video(z=-161.99)', 'quiz→quiz(z=139.69)', 'forum→video(z=-72.63)']

## 预测性能
- LogisticRegression 成功预测: acc=0.974, AUC=0.993
- RF 学业表现预测: 最终成绩 R²=0.793, 完成率 R²=0.844, 完成判定 acc=0.966
- RandomForest 风险预测 5-fold AUC: 全特征=0.981, 早期特征=0.872
- 行为特征预测画像标签 acc = 0.969
- GM(1,1) 日活跃趋势: {'平稳': 7311, '上升': 41, '下降': 17}

## 学习质量评价（AHP + 模糊综合评判）
- AHP 权重: {'学习行为': 0.32, '学习效果': 0.558, '互动参与': 0.122}, 一致性 CR = 0.016
- 评分均值: 45.0, 等级分布: {'差': 54255, '良': 4164, '中': 3820, '优': 391}

## 学习者画像

## 扩展分析摘要

- 核心学习者群体: {'gender': '男', 'age_band': '26-30', 'region': '城市', 'education': '本科'}
- 时间行为模式分布: {'常规型学习': 3049, '集中式周末学习': 2471, '碎片化高频学习': 1453, '夜间型学习': 396}
- 关联规则: 做测验、看视频 → 慕课长视频 (support=0.086, conf=1.0, lift=10.51)
- 关联规则: 做测验 → 慕课长视频 (support=0.086, conf=0.978, lift=10.28)
- 关联规则: 慕课长视频 → 做测验 (support=0.086, conf=0.902, lift=10.28)
- 只看短视频跳过完整慕课的学生占比: 90.3%
- 场景行为差异: {'慕课长视频': {'完播率': 0.375, '倍速使用率': 0.429, '平均拖拽次数': nan}, '短视频': {'完播率': 0.347, '倍速使用率': 0.0, '平均拖拽次数': nan}}
- 与成绩最相关的行为因素: resource_clicks(r=0.588), avg_quiz_score(r=0.554), active_days(r=0.54), total_clicks(r=0.451), video_clicks(r=0.279)
- 投入分层提升幅度: 高投入组较低投入组 完成率 +50.0%, 成绩 +12.5 分
- 学业表现预测: 最佳模型 GradientBoosting acc=0.969/0.970（RF/GBDT），Top5 关键行为: ['resource_clicks', 'avg_quiz_score', 'avg_session_min', 'avg_response_time', 'active_days']
- 早期预警: 0 名高风险学生
- SD 仿真最优策略: 加强资源供给（52 周投入水平 1.0)
- 优化建议: 11 条（详见 optimization_suggestions.csv）
                                  persona  n_students  share_pct                description       content_recommendation
                     At-Risk / Disengaged       50339       80.4 点击与活跃天数极低、得分差、几乎无互动，流失风险高。  短微内容单元、游戏化激励、主动推送通知、导师早期介入。
    At-Risk / Disengaged (Low Engagement)         783        1.3 点击与活跃天数极低、得分差、几乎无互动，流失风险高。  短微内容单元、游戏化激励、主动推送通知、导师早期介入。
                           Guided Learner         931        1.5    中等投入、偏好视频、频繁求助、依赖结构化指导。 分步视频教程、例题演示、即时反馈测验、脚手架式学习路径。
                 Independent Deep Learner        9826       15.7     高投入、深度探索内容、测验得分高、极少求助。  研究性阅读材料、开放性项目、同伴讨论论坛、挑战级测评。
Low-Engagement Strong-Performance Learner         751        1.2                                            按簇中心特征定制内容策略

## 干预队列统计

- none: 44702 人
- enrichment: 8431 人
- review: 6593 人
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