# AnalysisOfLearningBehavior — 统一学习行为分析与学习者画像系统

从在线学习行为数据中识别学生画像（Learner Persona）的完整分析管线：融合**多个真实公开数据集**，覆盖数据清洗、EDA、聚类画像、行为模式挖掘、关联分析、预测预警、AHP 评价、干预决策与报告输出全流程，入口为单个可直接运行的 `main.py`。


本项目整合了 4 个开源学习分析项目的方法论：

| 来源仓库 | 主要方向 | 整合进本项目的功能 |
|---|---|---|
| [Learner-Persona-Detection-via-Behavioral-Clustering](https://github.com/najigit/Learner-Persona-Detection-via-Behavioral-Clustering) | 学习行为聚类识别学习者类型 | K-Means 画像管线、肘部拐点 + 轮廓系数综合选 K、PCA 散点 / 雷达图 / 特征热力图、4 类画像定义与内容推荐 |
| [Learning-analytics-data-pipeline](https://github.com/AloneMask001/Learning-analytics-data-pipeline) | 学习数据清洗、EDA、成功预测与画像 | 数据清洗（去重、IQR 离群截断、中值填充）、LogisticRegression 成功/风险预测 |
| [ai-personalized-learning](https://github.com/elmersonrajan/ai-personalized-learning) | 学习者画像与个性化学习推荐 | 交互指标（accuracy / hesitation / engagement / hint_usage）纳入统一 schema、分画像内容推荐 |
| [MSc_Thesis_Learning_Analytics](https://github.com/shirgreif/MSc_Thesis_Learning_Analytics) | 行为日志分析与早期预测 | 事件流特征工程思路、RF 由行为特征预测画像标签、"早期特征 vs 全特征"对比预警 |

> **说明**：各仓库形态差异大且许可证不一，本项目不搬运源码，而是自研统一实现上述功能。参考仓库克隆在 `_repos/` 目录（其中天池慕课与 HarvardX/MITx 数据集也存放于此，见下文）。

---

## 快速开始

```bash
pip install -r requirements.txt

python main.py                    # 融合 data/real/ 等目录下的全部真实公开数据集
python main.py --k 4              # 指定聚类数（默认自动：肘部拐点 + 轮廓系数综合选择）
python main.py --device gpu       # GPU 加速（自动探测 PyTorch CUDA / cuML，无则回退 CPU）
python main.py --device cpu       # 强制 CPU（sklearn 多核并行）
```

命令行参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-dir` | `data` | 真实数据扫描目录 |
| `--out-dir` | `out` | 输出目录（自动创建） |
| `--k` | `0` | 聚类数；`0` = 自动（肘部拐点 + 轮廓系数两路证据等权融合） |
| `--device` | `auto` | 计算设备：`auto` 自动探测 GPU；`gpu` 强制 GPU（无后端回退 CPU 并告警）；`cpu` 强制 CPU |

环境要求：Python 3.8+（已做类型注解兼容，3.8 / 3.9 / 3.10+ 均可运行；已在 Python 3.8.8 与 3.14 下验证）。

---

## 一、数据集：多源真实公开数据

### 1.1 已接入的数据源（统一适配器接口）

| 数据源 | 规模 | 内容 | 存放位置 |
|---|---|---|---|
| UCI Student Performance | 395+649 名中学生 | 数学/葡萄牙语成绩单（G1/G2/G3、学习时长、出勤、资助） | `data/real/uci_student/` |
| 清华 FIB 实验室短视频行为数据集（WWW 2025） | 6,654 用户 / 79 万+条行为 | 点击、停留、完播、点赞/评论/转发等，含教育类目标记 | `data/real/thu_shortvideo/` |
| 阿里云天池《中国高校慕课课程多维画像与学习行为脱敏样本(2024–2025)》 | 715 学习者 / 5,000 事件 | 视频播放/测验/作业/论坛/资源下载/课程评价 | `_repos/阿里云天池…/` |
| HarvardX/MITx Person Course Dataset | 抽样 5 万用户（全量 32 万可调） | edX 2012-2013 两校课程：事件数/活跃天数/视频/论坛/章节/结业标签/人口属性 | `_repos/HarvardX_MITx…/` |
| EduData 集成包（中科大）托管的 ASSISTments 2009 | 4,217 学生 / 40 万条答题 | 中学数学答题行为：正确性、首答时长、提示使用、知识技能 | `data/real/edudata/assistment_2009/` |

每个适配器实现统一接口，返回 `(features_frames, clicks_df|None)`：
- 特征表含 `student_id`（加源前缀防冲突）/ `source` / 统一特征列 + 标签与人口属性列；
- 事件流含 `student_id / timestamp / action / resource_id / course_id / resource_type / completed / speed_used / seek_count`，供 SNA / LSA / GM(1,1) 使用，总量超 20 万条时随机采样；
- 适配器结果带缓存，特征表与事件流共用一次加载。

### 1.2 数据集获取方式

```bash
# 1) UCI Student Performance（官方 zip 内含 student-mat.csv / student-por.csv）
curl -L -o data/real/uci_student/student+performance.zip \
  https://archive.ics.uci.edu/static/public/320/student+performance.zip
# 解压后需再解一层 student.zip

# 2) EduData 集成包（GitHub 直连不通时用 ghproxy 镜像）
curl -L -o data/real/edudata/edudata_repo.zip \
  https://ghproxy.net/https://github.com/bigdata-ustc/EduData/archive/refs/heads/master.zip
# ASSISTments 2009 答题行为数据（中科大镜像可直连）
curl -o data/real/edudata/assistment_2009.zip \
  http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip

# 3) 清华短视频数据集：官网登录下载（全量 1000 万+ 条，本地已放置官方子样本）
#    http://fi.ee.tsinghua.edu.cn/datasets/short-video-dataset

# 4) 天池慕课 / HarvardX/MITx：已在 _repos/ 目录内

# 5）OULAD data，无法下载，需要教职工才行 (download from [here](https://analyse.kmi.open.ac.uk/open_dataset))
```

### 1.3 统一特征 schema

**核心行为特征**：`total_clicks`、`active_days`、`avg_session_min`、`forum_posts`、`resource_clicks`、`video_clicks`、`quiz_attempts`、`avg_quiz_score`、`assignment_submit_lag`、`help_requests`、`login_freq_per_week`

**扩展特征**：`accuracy`、`avg_response_time`、`hesitation_rate`、`engagement_score`、`hint_usage`、`success_rate`、`skip_rate`、`code_lines`、`code_functions`、`comment_ratio` 等

各源仅填充自己拥有的列，其余为 NaN 由清洗步骤中值插补；标签列优先使用数据集自带的真实标签（`success_status` / `certified`），否则由真实成绩中位数派生；`completion_rate` / `final_grade` 在数据集未提供时按该来源的真实行为派生代理值（代码中已明确标注，非合成样本）。人口属性（gender/age_band/region/education）**只保留真实列，缺失保持缺失**。

---

## 二、管线结构（九阶段）

```
┌─────────────────────────────────────────────────────────────┐
│ [1/9] 数据层  6 类真实数据集适配器（统一接口 + 缓存）          │
│        → union 列合并（source 标记）→ 真实事件流汇聚           │
├─────────────────────────────────────────────────────────────┤
│ [2/9] 特征层  EDA → 去重 → IQR 离群截断 → 中值填充            │
│        → 按缺失率筛选聚类特征列                               │
├─────────────────────────────────────────────────────────────┤
│ [3/9] 分析层  命名主成分（PCA）→ 肘部拐点 + 轮廓系数综合选 K   │
│        → K-Means（GPU 可选）→ PCA 散点 / 雷达图 / 热力图       │
├─────────────────────────────────────────────────────────────┤
│ [4/9] 挖掘层  SNA（PageRank/路径关联度）+ LSA（转换图谱 +      │
│        刷课/跳步异常）+ 时序模式 + Apriori + 场景对比          │
├─────────────────────────────────────────────────────────────┤
│ [5/9] 关联层  皮尔逊相关（含 p 值）+ 异质性分析 + 投入度分层    │
├─────────────────────────────────────────────────────────────┤
│ [6/9] 预测层  RF/GBDT 学业表现预测 + LogReg 成功预测 +         │
│        RF 风险预警（早期 vs 全特征）+ GM(1,1) 日活跃趋势       │
├─────────────────────────────────────────────────────────────┤
│ [7/9] 评价层  AHP 层次分析法（CR 一致性检验）+ 模糊综合评判 +   │
│        SD 系统动力学仿真                                      │
├─────────────────────────────────────────────────────────────┤
│ [8/9] 决策层  规则引擎干预队列 → 教师复核 → 分画像内容推荐 →    │
│        优化建议                                               │
├─────────────────────────────────────────────────────────────┤
│ [9/9] 输出层  out/：图表 PNG、CSV 结果集、report.md、          │
│        pipeline_log.jsonl 审计日志                            │
└─────────────────────────────────────────────────────────────┘
```

### 关键实现要点

**数据层**
- `load_uci_student()` / `load_thu_shortvideo()` / `load_mooc_tianchi()` / `load_harvardx_mitx()` / `load_edudata_assistments()`：五个真实数据集适配器，含脏数据容错（越界时间戳置 NaT 丢弃、非 UTF-8 字节 latin-1 回退、超量用户种子随机抽样）
- `build_clickstream()`：汇聚清华短视频、天池慕课事件与 `data/clickstream.csv` 的真实事件流
- `harmonize_demographics()`：人口属性仅来自真实数据列
- `load_real_data()`：通用扫描 `data/` 下 CSV/XLSX（别名映射）、JSONL、代码提交

**分析层（选 K 逻辑）**
- 自动模式综合**肘部拐点（kneedle 垂距法）与轮廓系数**两路证据等权融合
- 显式指定的 `k` 做合法性校验并夹取到 `[2, min(8, n-1)]`
- 大数据集下 silhouette（O(n²)）在 1.5 万固定随机子样本上拟合与评估，全量样本用最近质心分配标签
- `label_personas()`：基于簇中心 Z-score 的启发式签名命名（At-Risk / Guided Learner / Independent Deep Learner / 低投入高成绩型等）

**性能与加速**
- 数据加载全部向量化（消除 `groupby.apply(lambda)` 热点，79 万行清华数据秒级聚合）
- 所有 RandomForest 与 5 折交叉验证启用 `n_jobs=-1` 多核并行
- `--device gpu`：`ClusterEngine` 统一引擎自动选择后端——RAPIDS cuML（Linux/WSL，全 GPU）→ PyTorch CUDA（Windows 可用：GPU KMeans、分块轮廓系数、最近质心分配）→ CPU 回退

**挖掘 / 关联 / 预测 / 评价 / 决策层**
- SNA：学习者-资源二部图、加权 PageRank、路径关联度 Top 边（无 networkx 依赖）
- LSA：lag-1 转换频次矩阵 + 调整残差 z 检验，逐学生检出刷课/跳步异常，作为干预证据
- Apriori：自研轻量实现（无 mlxtend 依赖），行为+资源类型项集挖掘
- GM(1,1)：**按天**活跃度序列小样本趋势预测（清华公开子样本事件仅覆盖约一周，周粒度序列过短；日粒度每用户最多 7 点，满足建模最少 4 点要求）
- AHP + 模糊综合评判 → 0-100 学习质量评分；SD 系统动力学仿真 4 种策略 52 周演化
- 规则引擎队列（support/enrichment/review）+ 教师复核（幂等 review_key）+ 内容推荐 + 落地策略建议

---

## 三、真实数据运行结果（62,630 名学生，5 个真实来源）

```
来源分布: HarvardX/MITx 50000 · 清华短视频 6654 · ASSISTments 4217
          UCI 1044 · 天池慕课 715
```

- **自动选 K=4**：肘部拐点与轮廓系数一致指向 4（Silhouette=0.571，各 K 综合得分 K=4 最高 0.9）
- **四类画像**：At-Risk/流失风险 80.6%（HarvardX 大量零活跃注册者，符合 edX 真实分布）、独立深度学习者 15.7%、引导型 2.5%、低投入高成绩型 1.2%
- **行为→效果**：资源点击 r=0.588、活跃天数 r=0.54 与成绩正相关；高投入组较低投入组完成率 +50%、成绩 +12.5 分
- **预测**：RF 最终成绩 R²=0.793、完成判定 acc=0.966；仅 4 个早期特征即可达 AUC 0.872
- **碎片化学习**：事件流用户中 90.3% 只看短视频；短视频完播率 0.347，慕课长视频 0.374 且 42.8% 使用倍速
- **GM(1,1)**：平稳/上升/下降三档趋势 + 后验差比精度评级

> 指标随数据集更新而变，请以 `out/report.md` 中最新计算结果为准。

---

## 四、输出文件清单（`out/`）

| 文件 | 内容 |
|---|---|
| `eda_summary.csv` | 各特征描述统计 + 缺失数 |
| `elbow_silhouette.png` | 肘部法与轮廓系数曲线（标注肘部拐点与所选 K） |
| `pca_loadings.csv` / `pca_component_scores.csv` | 主成分载荷矩阵（含命名）/ 逐学生主成分得分 |
| `pca_clusters.png` / `pca_projection.csv` | 2D PCA 聚类散点 / 各样本投影坐标 |
| `tsne_clusters.png` / `tsne_projection.csv` | t-SNE 非线性嵌入散点（簇分离度视图，坐标尺度无业务含义）/ 子样本投影坐标 |
| `umap_clusters.png` / `umap_projection.csv` | UMAP 非线性嵌入散点（保留更多全局结构，需 `pip install umap-learn`，缺失时跳过） |
| `radar_personas.png` / `heatmap_features.png` | 各簇行为雷达图 / 簇级特征 Z-score 热力图 |
| `sna_*.csv` / `sna_network.png` | SNA 学习者/资源指标、路径 Top 边、网络可视化 |
| `lsa_transition_*.csv` / `lsa_transition_graph.png` / `lsa_behavior_patterns.csv` | LSA 转换矩阵、z 分数、转换图谱、逐学生异常模式 |
| `temporal_patterns.png` / `temporal_student_patterns.csv` | 时间行为模式（早中晚/周末）与逐学生标记 |
| `apriori_rules.csv` / `apriori_rules.png` | 关联规则（support/confidence/lift）与可视化 |
| `scene_comparison.csv` / `scene_comparison.png` / `scene_per_student.csv` | 慕课/短视频场景完播率、倍速、拖拽对比 |
| `correlation_results.csv` / `correlation_top_factors.png` | 全指标 vs 成绩皮尔逊相关（含 p 值） |
| `heterogeneity_analysis.csv` | 分性别/地区/教育子群体的时长-成绩相关系数 |
| `engagement_tier_analysis.csv` / `engagement_tiers.png` | 低/中/高投入分层的完成率与成绩提升幅度 |
| `rf_grade_feature_importance.csv` / `model_feature_importance.png` | RF/GBDT 特征重要性 |
| `logreg_coefficients.csv` | 成功预测 Top-10 特征系数 |
| `early_warning_students.csv` | 高辍学风险学生预警名单（结课概率 + 风险等级） |
| `gm11_forecast.csv` | 逐学生日活跃度 GM(1,1) 趋势与预测值 |
| `learning_quality_scores.csv` / `quality_score_distribution.png` | AHP+模糊评判评分与等级分布 |
| `sd_simulation_results.csv` / `sd_simulation.png` | 系统动力学 4 策略 52 周仿真 |
| `queues.csv` / `teacher_reviews.csv` | 干预队列（含证据）/ 教师复核记录 |
| `persona_recommendations.csv` / `optimization_suggestions.csv` | 分画像内容推荐 / 落地优化建议 |
| `demographics_distribution.csv` / `demographics_cross.png` | 真实人口属性分布与交叉分析 |
| `learner_profiles.csv` | 全部学生：统一特征 + cluster + persona + LSA 模式 + 主成分得分 + 质量评分 |
| `report.md` | 汇总图文报告 |
| `pipeline_log.jsonl` | 全流程 JSON 审计日志 |

---

## 五、注意事项

1. **真实数据原则**：管线不生成任何虚拟/合成数据；仅当数据集未提供 `completion_rate` / `final_grade` 时按该来源真实行为派生代理指标（代码中有标注）。
2. **Python 版本**：3.8+（`from __future__ import annotations` 兼容 PEP 604 注解）。旧版 scipy 可能提示 numpy 版本警告，无害；介意可 `pip install -U scipy`。
3. **依赖**：`requirements.txt` 之外，`report.md` 的 Markdown 表格依赖 `tabulate`（缺失时自动降级为纯文本表格）；`.xls` 读取需要 `xlrd`。绘图使用 Agg 后端，Linux 服务器可直接运行。
4. **GPU 加速**（可选）：
   - Windows：`pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`（Python 3.8 上限为 torch 2.4），加速 KMeans / 轮廓系数 / 质心分配；
   - Linux/WSL2：`pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com`，完整 GPU 栈；
   - 无 GPU 时自动回退 CPU 多核并行，不中断管线。
5. **跨源可比性**：不同平台特征尺度差异大，聚类前已做 StandardScaler；中值填充 + IQR 截断会把单一来源的特有特征拉向全局中位数，混合来源的 Silhouette 天然低于单源，属预期现象。
6. **数据真实口径**：HarvardX/MITx 全量 32 万用户，默认种子随机抽样 5 万（`load_harvardx_mitx(max_users=…)` 可调）；清华公开子样本事件时间仅覆盖一周（GM(1,1) 因此采用日粒度）；天池脱敏样本存在越界脏时间戳（已容错丢弃）；ASSISTments 部分行含非 UTF-8 字节（已 latin-1 回退）。
7. **规则阈值**：干预队列阈值（中位数倍数、求助频率 90 分位）为经验默认值，可在 `rule_queues()` 中调整；AHP 矩阵 `AHP_MATRIX`、LSA 异常判定阈值同理在 `main.py` 配置区，修改后权重自动重算并做 CR 检验。
8. **许可证**：本项目代码为自研实现，未复制参考仓库源码；使用各数据集/仓库时请自行确认原始许可证与数据使用协议的合规性。

---

## 项目结构

```
AnalysisOfLearningBehavior/
├── main.py                       # 统一入口：九阶段管线（数据/特征/分析/挖掘/关联/预测/评价/决策/输出）
├── data/
│   ├── real/
│   │   ├── uci_student/          # UCI Student Performance（student-mat/por.csv）
│   │   ├── thu_shortvideo/       # 清华短视频行为数据集（interaction_sampled.csv 等）
│   │   └── edudata/              # EduData 托管的 ASSISTments 2009 + 参考代码
│   # ← 通用扫描：*.csv / *.xlsx / *.jsonl / clickstream.csv / submissions/<学生ID>/*.py
├── notebooks/                    # 原交互式探索
├── out/                          # main.py 全部输出（自动创建）
├── _repos/                       # 参考仓库 + 天池慕课/HarvardX 数据集（对照用）
├── requirements.txt
└── README.md
```

## 后续方向

- 清华短视频全量数据集接入（官网下载 1000 万+ 条完整版）
- OULAD 真实公开数据集接入验证（`https://analyse.kmi.open.ac.uk/open_dataset`）
- cuML 后端扩展到 RandomForest / LogisticRegression（Linux/WSL 全 GPU 管线）
- 规则阈值外置 YAML 配置 + 敏感性分析
- 高斯混合模型（软簇归属）与画像随学期迁移的时序分析
- Streamlit 仪表盘（对齐 dss / ai-personalized-learning 的展示形态）
