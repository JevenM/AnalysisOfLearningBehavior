# AnalysisOfLearningBehavior — 统一学习行为分析与学习者画像系统

从在线学习行为数据中识别学生画像（Learner Persona）的完整分析管线：支持**多平台、多形态数据源**接入，覆盖数据清洗、EDA、聚类画像、预测预警、干预决策与报告输出全流程，入口为单个可直接运行的 `main.py`。

本项目整合了 7 个开源学习分析项目的方法论：

| 来源仓库 | 主要方向 | 整合进本项目的功能 |
|---|---|---|
| [Learner-Persona-Detection-via-Behavioral-Clustering](https://github.com/najigit/Learner-Persona-Detection-via-Behavioral-Clustering) | 学习行为聚类识别学习者类型 | K-Means 画像管线、肘部法 + 轮廓系数选 K、PCA 散点 / 雷达图 / 特征热力图、4 类画像定义与内容推荐 |
| [Learning-analytics-data-pipeline](https://github.com/AloneMask001/Learning-analytics-data-pipeline) | 学习数据清洗、EDA、PCA、成功预测与画像 | 数据清洗模板（去重、IQR 离群截断、中值填充）、LogisticRegression 成功/风险预测 |
| [student-learning-analytics-dss](https://github.com/wyou0410/student-learning-analytics-dss) | 动态画像、教师复核和报告生成 | 规则引擎干预队列（support / enrichment / review）、教师复核记录闭环、Markdown 报告输出 |
| [ai-personalized-learning](https://github.com/elmersonrajan/ai-personalized-learning) | 学习者画像与个性化学习推荐 | 交互日志数据源（accuracy / hesitation / engagement / hint_usage）、分画像内容推荐 |
| [MSc_Thesis_Learning_Analytics](https://github.com/shirgreif/MSc_Thesis_Learning_Analytics) | 行为日志分析与早期预测 | 事件流特征工程（成功率 / 活跃天数 / 升降级等）、RF 由行为特征预测画像标签、"早期特征 vs 全特征"对比预警 |
| [Studylytics](https://github.com/shwetha-researcher/Studylytics) | 学习行为分析与认知画像（无代码，仅设计参考） | 响应延迟、engagement 等认知类指标纳入统一特征 schema |
| [user2code2vec](https://github.com/dazcona/user2code2vec) | 从学生编程提交构建学习者画像 | 代码提交数据源：AST / token 统计特征（代码行数、函数数、调用数、注释率） |

> **说明**：各仓库形态差异大（notebook / Streamlit 应用 / 仅 PDF 报告）且许可证不一，本项目**不搬运源码**，而是自研统一实现上述功能。参考仓库克隆在 `_repos/` 目录（仅供参考对照，可删除，不影响运行）。

---

## 快速开始

```bash
pip install -r requirements.txt   # numpy / pandas / scikit-learn / matplotlib / seaborn + tabulate
python main.py                    # 默认：合成数据 + 自动扫描 data/ 目录中的真实数据
python main.py --n 600 --k 4      # 自定义合成样本量与聚类数
python main.py --k 0              # K=0 表示由轮廓系数自动选择最优 K
```

命令行参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-dir` | `data` | 真实数据扫描目录 |
| `--out-dir` | `out` | 输出目录（自动创建） |
| `--n` | `400` | OULAD 风格合成数据样本量 |
| `--k` | `4` | 聚类数；`0` = 自动选择 |

---

## 一、整合方案

### 1.1 设计原则

- **单入口**：全部功能收敛到一个 `main.py`，一条命令跑通六层管线
- **多源统一**：不同平台的公开数据集形态各异，通过"适配器 + 别名映射 + 统一 schema"对齐
- **功能保留**：各参考项目的核心功能（清洗、聚类、预测、规则队列、推荐、报告）全部保留并串联
- **零强制依赖真实数据**：内置各数据源的合成生成器，无数据也能完整演示；放入真实数据即自动切换

### 1.2 数据源整合

| 源形态 | 对应参考项目 | 接入方式 | 识别的关键字段 |
|---|---|---|---|
| A. 学生级汇总表 CSV/XLSX | Learner-Persona-Detection、data-pipeline | `data/*.csv`、`data/*.xlsx`，须含 `student_id` 列 | 列名经别名映射（如 `LMS_Clicks`→`total_clicks`、`Quiz_Score`→`avg_quiz_score`） |
| B. 交互日志 JSONL | ai-personalized-learning | `data/*.jsonl`，每行含 `student_id` + 指标 | `accuracy`、`avg_response_time`、`hesitation_rate`、`engagement_score`、`hint_usage` |
| C. 答题事件流 | MSc_Thesis_Learning_Analytics | 合成生成（可扩展读取真实日志） | `action_type`：1=答对 2=答错 3=跳过 9=升级 10=降级 |
| D. 代码提交 | user2code2vec | `data/submissions/<学生ID>/*.py` | 代码文本 → AST/token 静态特征 |

无真实数据时，四类源各自动生成合成数据（OULAD 风格为主力源，其余按 `n/2`、`n/4` 比例），并打上 `source` 列标记来源。

### 1.3 统一特征 schema（23 列）

**核心行为特征（11 列）**：`total_clicks`、`active_days`、`avg_session_min`、`forum_posts`、`resource_clicks`、`video_clicks`、`quiz_attempts`、`avg_quiz_score`、`assignment_submit_lag`、`help_requests`、`login_freq_per_week`

**扩展特征（12 列）**：`accuracy`、`avg_response_time`、`hesitation_rate`、`engagement_score`、`hint_usage`、`success_rate`、`skip_rate`、`unique_days_ratio`、`level_ups`、`code_lines`、`code_functions`、`comment_ratio`

各源仅填充自己拥有的列，其余为 NaN，由清洗步骤统一插补——这就是"不同平台公开数据集整合"的实现方式。

---

## 二、管线结构（八阶段）

```
┌─────────────────────────────────────────────────────────────┐
│ [1/8] 数据层  多源适配器 + 合成生成器 + data/ 自动扫描        │
│        → 统一 schema 合并（union 列 + source 标记）           │
├─────────────────────────────────────────────────────────────┤
│ [2/8] 特征层  去重 → IQR 离群截断 → 中值填充                  │
│        → 按缺失率筛选聚类特征列 → 派生完成率/最终成绩目标      │
├─────────────────────────────────────────────────────────────┤
│ [3/8] 分析层  命名主成分（PCA 降维）→ 肘部法 + 轮廓系数选 K    │
│        → K-Means 聚类 → PCA 散点 / 雷达图 / 热力图 → 画像命名  │
├─────────────────────────────────────────────────────────────┤
│ [4/8] 挖掘层  社会网络分析 SNA（度中心性/PageRank/路径关联度） │
│        + 滞后序列分析 LSA（转换图谱 + 刷课/跳步异常检测）      │
├─────────────────────────────────────────────────────────────┤
│ [5/8] 预测层  RF 学业表现预测（完成率/最终成绩+特征重要性）    │
│        ① LogReg 成功预测 ② RF 风险预警（早期 vs 全特征）       │
│        ③ RF 预测画像标签 ④ 灰色 GM(1,1) 小样本趋势预测        │
├─────────────────────────────────────────────────────────────┤
│ [6/8] 评价层  AHP 两两比较定权（CR 一致性检验）               │
│        + 模糊综合评判 → 学习质量量化评分（0-100）             │
├─────────────────────────────────────────────────────────────┤
│ [7/8] 决策层  规则引擎 → support / enrichment / review 队列   │
│        （含 LSA 行为异常证据）→ 教师复核 → 内容推荐           │
├─────────────────────────────────────────────────────────────┤
│ [8/8] 输出层  out/ 目录：图表 PNG、CSV 结果集、report.md、    │
│        pipeline_log.jsonl 审计日志                            │
└─────────────────────────────────────────────────────────────┘
```

### 各层实现要点

**数据层**
- `synthetic_oulad()`：按 4 类潜在学习类型的分布参数生成 OULAD 风格数据（含 `latent_type` 真值，仅用于评估 ARI）
- `synthetic_interaction()`：交互系统指标 + 部分重叠的 OULAD 字段，模拟真实系统间特征重叠
- `synthetic_chatbot_events()` → `events_to_features()`：事件流按学生聚合出 `success_rate`、`skip_rate`、`avg_response_time`、`active_days`、`unique_days_ratio`、`avg_interactions_per_day`、`level_ups/downs` 等
- `synthetic_code_submissions()` → `code_to_features()`：生成代码文本后用 `ast` 模块解析，统计 `code_lines`、`code_functions`、`code_calls`、`comment_ratio`
- `load_real_data()`：扫描 `data/` 下 CSV/XLSX（`normalize_columns()` 做小写化 + 别名映射）、JSONL（提取 `details.metrics`）、`submissions/**/*.py`，无法解析的文件安全跳过并记录日志
- `synthetic_clickstream()` / `load_clickstream()`：点击流事件源（video/resource/quiz/forum/help），学生 ID 与 OULAD 主力源对齐，内置"正常/刷课/跳步"三种行为模式；若存在 `data/clickstream.csv`（须含 `student_id/timestamp/action` 列）则优先加载真实点击流

**特征层**
- `clean_features()`：按 `student_id` 去重；每列按 IQR（±3 倍）截断离群值；中值填充缺失
- `select_cluster_columns()`：仅保留非缺失率 ≥ 60% 的列参与聚类，避免单一来源独有的特征主导聚类

**分析层**
- `pca_named_components()`：对 20+ 维原始行为指标 PCA 降维提取 4 个综合主成分，按载荷模式自动命名（学习投入度 / 资源偏好度 / 互动活跃度 / 学业表现），输出载荷矩阵与逐学生主成分得分，大幅降低后续建模的维度冗余
- `cluster_personas()`：基于学习时长、访问频次、资源偏好等核心特征，K∈[2,8] 扫描惯性 + 轮廓系数并绘图；`KMeans(n_init=20)` 最终聚类；PCA 2D 投影（记录方差解释率）；雷达图取各簇均值归一化；热力图展示簇级 Z-score
- `label_personas()`：基于簇中心 Z-score 的启发式签名命名——低投入低表现→At-Risk（≈浅尝辄止型）；高投入高分低求助→Independent Deep Learner（≈深度投入型）；高求助/高视频→Guided Learner；测验尝试率相对活跃度偏高→Exam-Driven（≈突击备考型）；同签名多簇时追加 `(High/Low Engagement)` 限定词；无法匹配时回退为 `High/Low-Engagement Strong/Weak-Performance Learner`

**挖掘层（SNA + LSA）**
- `sna_analysis()`：将学习者、课程、学习资源抽象为网络节点（学习者-资源二部图，边权=访问次数），计算节点度中心性与加权 PageRank（幂迭代，无 networkx 依赖）；按学生内时序统计资源→资源转换次数得到路径关联度 top 边；输出学习者/资源两套指标表与资源网络可视化图
- `lsa_analysis()`：对点击流行为序列编码（video/resource/quiz/forum/help），统计 lag-1 转换频次矩阵并计算调整残差 z 分数（标准 LSA 显著性检验），绘制行为转换图谱（边宽 ∝ |z|）；逐学生检测异常模式——**刷课**（最长连续视频段 ≥ 5 且视频占比 ≥ 0.6）、**跳步学习**（quiz 前 3 步内无资源/视频的比例 > 0.5），结果合并进统一画像表并作为干预队列证据

**预测层**
- `rf_performance_model()`：以历史行为特征为输入、课程完成率与最终成绩为输出——RF 回归预测最终成绩（R²/MAE）与完成率，RF 分类判定是否完成（完成率 ≥ 0.75），并输出特征重要性排序表与条形图
- `run_predictions()`：① LogReg 成功预测（标准化 + 30% 测试集，输出 acc/AUC 及 Top-10 系数）；② RF 风险预警（5 折 CV AUC，对比"仅活动类早期特征"与"全特征"，模拟学期早期预警能力）；③ RF 由行为特征预测画像标签的 5 折准确率
- `gm11_forecast()`：灰色 GM(1,1) 模型——针对每位学习者的周活跃度序列（少数据、部分信息缺失场景）建模，发展系数 a 判定趋势（上升/下降/平稳），后验差比 C 评定模型精度（优/良/合格），预测未来 2 周活跃度，弥补机器学习模型对小样本群体预测差的短板

**评价层（AHP + 模糊综合评判）**
- `ahp_fuzzy_scores()`：构建学习行为（点击/活跃/登录频次/会话时长）、学习效果（测验分数/正确率/成功率）、互动参与（论坛/求助/参与度）三维度指标体系；`AHP_MATRIX` 两两比较矩阵经主特征向量法求权重并做 CR 一致性检验（默认 CR=0.016 通过）；各指标 min-max 归一（求助、跳过率等反向指标取反）后代入优/良/中/差梯形隶属函数，模糊合成 B = W·R，按等级分值向量加权得到每位学习者 0-100 的学习质量综合评分与最优等级

**决策层**
- `rule_queues()`：以中位数为基准的阈值规则——得分远低于中位数或高频求助→support（导师介入/引导资源）；高分高投入→enrichment（拓展资源）；临界或低投入低分、LSA 检出刷课/跳步异常→review（教师复核），每条附证据字符串
- `generate_recommendations()`：按画像汇总人数、占比、行为描述与内容推荐
- 教师复核表：review/support 队列生成 `review_key`（学生+动作，幂等）与 `decision=pending`，作为复核闭环起点

**输出层**
- `log()`：所有过程信息同时打印并追加写入 `out/pipeline_log.jsonl`（JSON 审计日志）
- `write_report()`：生成 `out/report.md`，含数据概览、预测性能、画像表（Markdown 表格）、队列统计与复核名单

---

## 三、修改说明（相对原仓库的变更）

原仓库是单一 K-Means 管线（`analysis/learner_profiling.py` + `data/simulate_oulad.py`），本次整合的变更：

1. **新增 `main.py` 统一入口**：原脚本按模块拆分为六层函数（原为顺序执行的顶层代码），并加入 CLI 参数
2. **数据源从 1 个扩展到 4 个**：原仅有 OULAD 合成数据；新增交互日志、答题事件流、代码提交三类源及其统一 schema 映射
3. **特征从 11 列扩展到 23 列**：新增认知类（响应延迟、犹豫率、专注度）与代码类特征；新增缺失率驱动的聚类特征筛选
4. **清洗增强**：新增去重与 IQR 离群截断（原脚本无清洗）
5. **新增预测层**：LogReg 成功预测、RF 风险预测与早期预警对比、画像可预测性评估
6. **新增决策层**：规则引擎队列 + 教师复核 + 内容推荐表
7. **输出规范化**：结果统一写入 `out/`（原脚本写在根目录），新增 report.md 与 JSONL 审计日志
8. **绘图兼容化**：强制 Agg 后端（无显示器可运行）；雷达图支持任意 K 的子图布局；热力图改用 imshow（去除 seaborn 强依赖）
9. **原有文件保留**：`analysis/learner_profiling.py`、`data/simulate_oulad.py`、`notebooks/` 仍可独立运行，作为单源基线
10. **新增行为模式挖掘层**：SNA（度中心性/PageRank/路径关联度）、LSA（行为编码转换图谱 + 刷课/跳步异常检测）、命名主成分（投入度/偏好度/活跃度/表现四综合维度）
11. **预测层扩展**：新增 RF 学业表现预测（课程完成率 + 最终成绩回归 + 完成判定 + 特征重要性）与灰色 GM(1,1) 小样本趋势预测；新增派生目标变量 `completion_rate` / `final_grade`（真实数据提供同名列时优先使用）
12. **新增评价层**：AHP 层次分析法（两两比较定权 + CR 一致性检验）+ 模糊综合评判 → 学习质量量化评分
13. **管线从 6 阶段扩展为 8 阶段**：新增挖掘层与评价层；点击流事件成为第 5 类数据源（`synthetic_clickstream` / `data/clickstream.csv`）；干预队列证据新增 LSA 行为异常模式

原项目的 4 类画像定义与内容推荐（沿用为默认命名体系）：

| 画像 | 行为签名 | 内容推荐 |
|---|---|---|
| Independent Deep Learner | 高投入、长会话、高分、少求助 | 研究性材料、开放项目、论坛、挑战测评 |
| Guided Learner | 中投入、高视频、高频求助 | 分步视频、例题、即时反馈测验、脚手架路径 |
| Exam-Driven Learner | 低日常活跃、考前爆发、高测验率 | 精简摘要、记忆卡片、模拟卷、间隔重复 |
| At-Risk / Disengaged | 极低活跃、低分、无互动 | 微内容、游戏化、推送通知、导师介入 |

---

## 四、使用方式

### 4.1 直接运行

```bash
python main.py
```

终端会依次打印八阶段进度、主成分命名、聚类指标（Silhouette / ARI）、SNA/LSA 挖掘结果、预测性能、GM(1,1) 趋势分布、AHP 权重与质量评分统计，最后列出 `out/` 全部输出文件。

### 4.2 接入真实数据

把数据放到 `data/` 目录即可被自动识别与合并：

```
data/
├── 平台A汇总表.csv          # 必须含 student_id 列，其余列名自动别名映射
├── 平台B交互日志.jsonl      # 每行: {"student_id": "...", "details": {"metrics": {...}}}
├── clickstream.csv          # 可选：真实点击流，须含 student_id/timestamp/action
│                            #   列，可选 resource_id/course_id；供 SNA/LSA/GM 使用
└── submissions/
    ├── S0001/task1.py       # 每个学生一个子目录，.py 文件提静态特征
    └── S0002/task1.py
```

- 列名匹配不区分大小写与下划线空格差异；`LMS_Clicks`、`clicks`、`Quiz_Score`、`success_status` 等常见字段已内置别名
- 无 `student_id` 列或无法解析的文件会被跳过并记录日志，不会中断管线
- 标签优先级：存在 `success_status` 列则用作监督标签（Successful/Pass/True 等视为成功），否则按 `avg_quiz_score` 中位数阈值派生

### 4.3 输出文件清单（`out/`）

| 文件 | 内容 |
|---|---|
| `eda_summary.csv` | 各特征描述统计 + 缺失数 |
| `elbow_silhouette.png` | 肘部法与轮廓系数曲线（标注所选 K） |
| `pca_loadings.csv` / `pca_component_scores.csv` | 主成分载荷矩阵（含命名）/ 逐学生主成分得分 |
| `pca_clusters.png` / `pca_projection.csv` | 2D PCA 聚类散点 / 各样本投影坐标 |
| `radar_personas.png` | 每个簇的行为雷达图（特征均值归一化） |
| `heatmap_features.png` | 簇级特征 Z-score 热力图 |
| `sna_learner_metrics.csv` / `sna_resource_metrics.csv` | SNA 学习者 / 资源节点度中心性与 PageRank |
| `sna_top_paths.csv` / `sna_network.png` | 资源访问路径关联度 Top 边 / 资源网络可视化 |
| `lsa_transition_counts.csv` / `lsa_transition_zscores.csv` | LSA 转换频次矩阵 / 调整残差 z 分数 |
| `lsa_transition_graph.png` / `lsa_behavior_patterns.csv` | 行为转换图谱 / 逐学生刷课/跳步模式判定 |
| `learner_profiles.csv` | 全部学生：统一特征 + cluster + persona + LSA 模式 + 主成分得分 + 质量评分 |
| `logreg_coefficients.csv` | 成功预测 Top-10 特征系数 |
| `rf_grade_feature_importance.csv` / `rf_feature_importance.png` | RF 成绩预测特征重要性排序 / 条形图 |
| `rf_feature_importance.csv` | RF 风险预测特征重要性 |
| `gm11_forecast.csv` | 逐学生周活跃度 GM(1,1) 趋势与 2 周预测值 |
| `learning_quality_scores.csv` / `quality_score_distribution.png` | AHP+模糊评判维度得分/等级/综合评分 / 分布直方图 |
| `queues.csv` | 干预队列：queue / suggested_action / evidence（含行为异常） |
| `teacher_reviews.csv` | 教师复核记录（decision 初始为 pending） |
| `persona_recommendations.csv` | 各画像人数、占比、描述、内容推荐 |
| `report.md` | 汇总图文报告（含全部方法结果摘要） |
| `pipeline_log.jsonl` | 全流程 JSON 审计日志 |

### 4.4 编程调用示例

```python
from main import synthetic_oulad, clean_features, rule_queues, label_personas

df = synthetic_oulad(n=200)          # 仅生成 OULAD 风格数据
df = clean_features(df)              # 清洗（此处直接可用，缺 student_id 时需先补）
```

---

## 五、注意事项

1. **依赖**：`requirements.txt` 之外，`report.md` 的 Markdown 表格依赖 `tabulate`（`pip install tabulate`）。绘图使用 Agg 后端，无需显示器，Linux 服务器可直接运行。
2. **Python 版本**：需 Python 3.9+（代码使用了 `list[str]` 等类型注解写法）。已在 Python 3.14 下验证通过。
3. **跨源可比性**：不同平台特征尺度差异大，聚类前已做 StandardScaler；但"中值填充 + IQR 截断"会把单一来源的特有特征拉向全局中位数，混合来源较多时 Silhouette 会显著低于单源（实测约 0.2 vs 单源 0.49），属预期现象。建议关注各簇的相对结构而非绝对分数。
4. **合成数据仅用于演示**：`latent_type`、合成代码文本等均为随机生成，预测指标（如 AUC≈0.99）反映的是合成分布的可分性，不代表真实数据上的性能；接入真实数据后请以 `report.md` 中重新计算的指标为准。
5. **事件流数据源暂为内置生成**：`action_type` 语义（1 对 / 2 错 / 3 跳 / 9 升 / 10 降）与 MSc 论文仓库一致；如有真实的答题日志（CSV/Excel），可参照 `events_to_features()` 的字段聚合逻辑接入。
6. **旧输出文件**：`data/learner_profiles.csv` 是原单源管线的产物，因缺 `student_id` 列会被真实数据扫描器安全跳过；新的结果统一输出到 `out/`。
7. **规则阈值**：干预队列的阈值（中位数倍数、求助频率 90 分位）为经验默认值，可在 `rule_queues()` 中按业务调整；后续可参考 dss 仓库引入 YAML 配置与阈值敏感性分析。
8. **许可证**：本项目代码为自研实现，未复制参考仓库源码；使用各仓库的方法论时，请自行确认其原始许可证（如 ai-personalized-learning、student-learning-analytics-dss 带 LICENSE）在你们场景下的合规性。
9. **AHP 与 LSA 参数可调**：AHP 两两比较矩阵 `AHP_MATRIX`、三维度指标集 `AHP_INDICATORS`、模糊隶属参数 `FUZZY_GRADES` 均在 `main.py` 顶部配置区，修改后权重会自动重算并做 CR 检验（CR ≥ 0.1 时日志会提示矩阵不一致）；LSA 异常判定阈值（连续视频段长度、视频占比、跳步比例）在 `lsa_analysis()` 内，可按业务口径调整。
10. **GM(1,1) 适用条件**：灰色模型适合小样本（每生 ≥ 4 个周数据点）指数型序列；合成点击流的周活跃度近似随机游走，趋势判定（上升/平稳/下降）仅供演示，接入真实周序列后趋势与预测值才有业务含义；`model_grade` 为后验差比 C 的精度等级（C < 0.35 优、< 0.5 良、否则合格），C 过大说明序列波动剧烈不宜外推。
11. **PCA 命名是启发式**：主成分按各语义域 |载荷| 总和自动归属命名，若两个主成分载荷模式接近会出现"（次要）"后缀；业务上应以 `pca_loadings.csv` 的实际载荷为准，命名可在 `pca_named_components()` 的 `DOMAIN_MAP` 中调整。

---

## 项目结构

```
AnalysisOfLearningBehavior/
├── main.py                       # 统一入口：八阶段管线（数据/特征/分析/挖掘/预测/评价/决策/输出）
├── data/
│   ├── simulate_oulad.py         # 原单源合成生成器（保留）
│   └── learner_profiles.csv      # 原管线输出（保留）
│   # ← 真实数据放这里：*.csv / *.xlsx / *.jsonl / submissions/<学生ID>/*.py
├── analysis/
│   └── learner_profiling.py      # 原单源聚类管线（保留，可独立运行）
├── notebooks/
│   └── exploration.ipynb         # 原交互式探索
├── out/                          # main.py 全部输出（自动创建）
├── _repos/                       # 7 个参考仓库克隆（仅供对照，可删除）
├── requirements.txt
└── README.md
```

## 后续方向

- 真实 OULAD 公开数据集接入验证（`https://analyse.kmi.open.ac.uk/open_dataset`）
- 规则阈值外置 YAML 配置 + 敏感性分析（对齐 dss 仓库）
- 高斯混合模型（软簇归属）与画像随学期迁移的时序分析
- 代码提交源升级为向量嵌入（对齐 user2code2vec 的 BOW / embedding 思路）
- Streamlit 仪表盘（对齐 dss / ai-personalized-learning 的展示形态）


## 实现总结

### 第一部分

基础数据统计与特征画像（basic_stats_analysis + add_demographics）

人口属性：gender/age_band/region/education 四维度频数统计，真实数据含同名列则沿用，否则按典型在线教育结构合成并与行为轻度关联；核心群体自动识别（本次：男/18-25/华东/本科）
行为基线：登录频次、周均学习时长、完播率、提交率、论坛互动等指标的均值/中位数/极值（behavior_baseline.csv）
交叉对比：按性别/地区/教育分组对比学习时长、视频偏好、完成率（demographics_cross.png 展示了教育背景分化：硕士及以上时长最高、高中及以下成绩明显偏低）

### 第二部分

行为模式挖掘（管线 [4/9]）

时序特征：早/中/晚/深夜分布 + 周度周期性（周末高亮）+ 逐学生模式标记（常规型/碎片化高频/夜间型/集中式周末，见 temporal_patterns.png）
K-Means 精细化画像：已有能力复用，产出深度投入型/浅尝辄止型/突击备考型/刷课型（LSA 联动）标签
Apriori 路径挖掘：自研轻量 Apriori（无 mlxtend 依赖），行为+资源类型项集 → 224 条规则（support/confidence/lift），并统计“只看短视频跳过完整慕课”负向行为（apriori_rules.png）
场景对比：慕课/微课/短视频三类资源的完播率、倍速使用率、拖拽次数（scene_comparison.png：短视频完播率 80% 最高、慕课拖拽 3.5 次最多）

### 第三部分

关联与相关性（管线 [5/9]）

皮尔逊相关：全部行为指标 vs 最终成绩/认证状态，含 p 值显著性（correlation_top_factors.png：测验分 r=0.77 最强、提交率 r=0.68）
异质性分析：分性别/地区/教育子群体的“时长-成绩”相关系数表
投入分层：低/中/高三分位对比——高投入组较低投入组完成率 +26.0%、成绩 +18.5 分（engagement_tiers.png）

### 第四部分

预测与评价建模（管线 [6/9]、[7/9]）

扩展建模：RF vs GBDT 成绩等级预测（acc=0.850，召回 0.698）+ 认证判定（acc=0.956）+ Top5 关键行为（测验分、正确率、点击量、视频、资源点击）+ 高辍学风险预警名单 early_warning_students.csv
模糊综合评价：AHP 权重（已有）→ 每人 0-100 质量评分
SD 系统动力学仿真：学习投入/资源供给/平台激励三条反馈回路，欧拉法模拟 4 种策略 52 周演化（sd_simulation.png：基线策略投入持续衰减，综合优化策略 8 周内收敛到满投入）

### 第五部分

优化建议（generate_optimization_suggestions）

输出 11 条落地策略（optimization_suggestions.csv）：画像差异化推送、完成率最低两地区的区域运营、早期预警触发机制（7 天未登录/完播率骤降/作业逾期自动介入）、分场景资源设计（短视频加知识点锚点、慕课拆分为 8-12 分钟微模块）、SD 仿真结论支撑的平台长期策略
一点说明：人口属性与播放行为目前为合成数据演示，接入真实数据（CSV 含 gender/region/education 列、点击流含 resource_type/completed/speed_used/seek_count 列）后全部自动切换为真实统计。如需把本次扩展写进 README，告诉我即可。