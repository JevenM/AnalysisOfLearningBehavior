# 选题 7 · 基于线上数据的学习者行为分析 —— 最终交付包

《系统工程原理》课程实践项目（2026 年实践项目指南 · 选题 7）。
本目录是**自洽、可复现、可现场改参重跑**的完整交付物：看板、图表、结果与支撑文档共用同一组实跑结果。

> **版本说明（2026-11 六源统一重构）**：全项目以"**六类真实公开数据集合并总表**"（`up_` 统一管线，
> 66,644 名学习者）为唯一主分析口径，**不再单独拆分分析任何一个课程数据集（含 OULAD）**。
> 原 OULAD 单源链（analysis.py / app.py / forecast.py / fuzzy_eval.py / figures.py / demo_runs.py
> 及 results.json 等产物、fig04–fig10 与 fig15–fig19 插图）已按确认删除；
> `common.py`（GPU 聚类引擎）与 `evaluators.py`（AHP/熵权唯一实现）由统一管线与多形态链共用。

## 零、一键运行（推荐从这里开始）

参照借鉴仓库 `main.py` 的单入口模式，本项目提供 `run_all.py`：一条命令复现整条分析链（结果 → 图表 → 看板）。

```bash
# 1. 安装依赖（一次性）
python -m pip install -r 03_源代码/requirements.txt

# 2. 先体检（只读，确认数据与依赖齐全）
python 03_源代码/run_all.py --check

# 3. 一键全链路（默认用随包派生数据，完全离线可复现）
python 03_源代码/run_all.py

# 常用变体
python 03_源代码/run_all.py --device gpu          # 统一管线聚类启用 GPU 加速
```

一键链的执行顺序与依赖：`multimodal.py（四形态测评）→ profile_modeling.py（内容级画像）→ **统一管线 up_1..up_8（主线）** → 三个看板`，每步独立子进程、失败即停、可重复运行（幂等覆盖）。默认**不重新下载、不生成仿真数据**。

### 建模控制台（可交互验证程序，答辩现场改参重跑）

对应指南 4.4（1）（2）"可交互界面 + 现场更换输入数据或参数重新运行"：

```bash
python 03_源代码/model_console.py      # 启动本地控制台（自动打开浏览器 http://127.0.0.1:8000/）
```

- **改参数**：聚类数 K（0=自动）/ 计算设备 / Apriori 支持度与置信度 / SD 仿真周数 / 多形态预算比例；
- **换数据**：上传自定义学习者总表（up_unified.csv 同 schema），一键恢复默认六源数据；
  **内置测试文件** `02_数据与数据集/测试上传数据/test_upload_harvardx_holdout.csv`
  （8,000 名未参与建模的留出学习者，真实数据零重叠；`python 03_源代码/build_test_upload.py` 可再生成）；
  上传格式要求与 6 步测试流程见 `06_支撑文档/运行说明.md` 方式 0.6；
- **重跑**：沙箱模式（写入 `04_运行结果_沙箱/`，不动正式结果，约 1–3 分钟）或正式模式
  （写入 `04_运行结果/` 并自动刷新三个看板）；
- **看板自动刷新**：控制台内置监视器（`auto_refresh_dashboards.py`，启动时自动开后台线程）——
  每次管线跑完、运行产物比看板新时自动重建三个看板，**任何方式跑完代码都无需手动运行
  build_*.py**；也可单独常驻：`python 03_源代码/auto_refresh_dashboards.py`（绕过控制台
  手动跑 `up_*.py` 各阶段时使用）；
- **看变化**：每次运行自动与上一次快照对比，预测指标 / 四类画像 / 干预队列 / 优化建议的
  变化逐行高亮（▲▼），直接支撑答辩"解释结果变化"；
- **看板随范围切换**：三个看板链接随沙箱/正式范围联动——沙箱范围打开按沙箱结果
  单独构建的沙箱版看板，正式范围打开正式版；每个看板顶部有醒目数据源横幅
  （绿色正式 / 橙色沙箱），不会混淆；沙箱版由自动刷新监视器自动构建；
- **看图表**："⑥ 中间图表"随运行实时产出 18 张管线图（聚类 / 挖掘 / 预警 / SD 等），
  运行中每 15 秒轮询刷新、点击放大，沙箱与正式两套独立切换；
- 页面与三个看板互链（`model_console.html`，由控制台服务托管）。

### 统一管线（主线，`03_源代码/up_*.py`）

融合 **6 类真实公开数据集**（HarvardX-MITx 50,000 / 清华短视频 6,654 / ASSISTments 4,217 /
OULAD 4,014 / UCI 课堂对照 1,044 / 天池慕课 715，合计 **66,644 名学习者**）：

```
up_build_unified.py   数据层+特征层 → up_unified.csv / up_clicks.csv
up_cluster.py         PCA 命名 + 肘部/轮廓融合选 K + ClusterEngine 聚类（GPU 可选）+ t-SNE/UMAP
up_mine.py            SNA / LSA / 时序模式 / Apriori / 场景对比
up_associate.py       相关性 / 异质性 / 投入分层
up_predict.py         RF/GBDT/LogReg / 早期预警 / GM(1,1)
up_evaluate.py        AHP+模糊综合评判（委托 evaluators）+ SD 系统动力学仿真
up_decide.py          干预队列 / 内容推荐 / 优化建议
up_report.py          汇总报告 → up_report.md
```

**GPU 加速**（`--device auto|cpu|gpu`，经环境变量 `WB_DEVICE` 贯通全部子进程）：
自动探测 RAPIDS cuML（Linux/WSL）→ PyTorch CUDA（Windows 可用）→ CPU 回退；
加速范围为 KMeans / 轮廓系数 / 最近质心分配。聚类与选 K 全链共用 `common.ClusterEngine`。

## 一、先看什么

| 交付物 | 路径 |
|---|---|
| **主看板（推荐先看）** | `05_图表与看板/dashboard.html`（7 屏六源统一叙事：四类画像 → 投入与结果 → 节律与兴趣 → 行为序列 → 早期预警 → 评价与行动 → 数据来源；屏 1 顶部含**总体方法架构图**（六源数据→统一特征→七环节建模链→报告看板全景，`build_architecture_figure.py` 生成 fig00_architecture.png）；每屏底部嵌入该环节的**管线证据图**——18 张实跑图表按环节分组、逐图标注实跑结论、点击放大，全部图表中文标注（降维图/雷达图附维度含义声明）；先给结论再给证据，零建模基础也能看懂） |
| **形态画像看板** | `05_图表与看板/persona_dashboard.html`（慕课/微课/短视频/课堂四形态切换 + 画像群探索 + 散点联动） |
| **内容画像看板** | `05_图表与看板/profile_dashboard.html`（观看时段 / 兴趣爱好 / 短视频人口属性 / 天池倍速标签，可排序表格 + 差异检验） |
| **全图表解读手册** | `05_图表与看板/conclusion.md`（全部图表逐一标注：算法全称、输入数据、实跑结果、怎么读、与后续环节关系、结论；按系统工程分析步骤与代码运行流程组织） |
| **建模控制台（改参重跑）** | `05_图表与看板/model_console.html`（需先 `python 03_源代码/model_console.py` 启动本地服务；改参数/换数据→重跑→结果对比，对应课程可交互验证要求） |

### 三看板导读：区别、联系与讲解顺序

**数据可信度承诺**：三个看板的每一个数字均由构建脚本从 `04_运行结果/` 实跑 JSON/CSV 读取注入（构建脚本内无编造值，已做逐数字溯源比对全部一致）；`seed=20260101` 只用于分层抽样（可复现），不生成任何仿真数据。

| | 主看板 `dashboard.html` | 形态画像看板 `persona_dashboard.html` | 内容画像看板 `profile_dashboard.html` |
|---|---|---|---|
| **回答什么** | 全景结论：分几类、投入与结果的关系、平台该怎么帮 | 同一方法在 4 种学习形态下是否都成立 | "这个人是谁"：时段/兴趣/人口属性/平台标签刻画 |
| **数据粒度** | 六源合并 66,644 名学习者（主线） | 多形态样本 9,236 条（MOOC 6,000/微课 723/短视频 1,469/课堂对照 1,044） | 国内短视频 6,654 用户 + 天池微课 |
| **交互方式** | 7 屏顺序叙事 | 四形态标签切换、画像群点击联动散点 | 四维度标签、可排序表格 + 联动条形图 |
| **数据来源** | up_*.json / up_*.csv / profile_results.json | multimodal_results.json + multimodal_all.csv | profile_results.json |

**联系**：三看板互链（主看板头部可跳另两个）；**同一套实跑结果**的三种视角——主看板给"结论"，两个专题看板给"证据细节"（形态间对比、内容级差异检验）。任何一屏的数字都能在 conclusion.md（全图表解读手册）找到算法与数据来源解释。

**讲解顺序（演示/答辩建议）**：① 从 `dashboard.html` 第 1 屏"四类学习者画像卡"开场（3 分钟讲完四类人是谁、谁有风险）；② 第 5-6 屏早期预警与行动建议收尾（这是决策者最关心的"怎么办"）；③ 观众追问"换个场景还成立吗"→ 打开 `persona_dashboard.html` 切到短视频/课堂形态；④ 追问"学习的人是怎样的"→ 打开 `profile_dashboard.html` 展示时段/兴趣/人口属性差异检验。

### 主看板：结论先行的六源统一叙事（推荐先看）

`dashboard.html` 按"让不懂建模的人也能看懂"组织为 7 屏叙事：**四类学习者画像卡（六源统一聚类：风险流失型 53,281 / 自主深度型 11,445 / 引导依赖型 1,165 / 低投入高绩效型 753）→ 投入与结果（分层 +50% 完成率/+23.7 分）→ 节律与兴趣 → 行为序列与资源（LSA/SNA/Apriori）→ 第 4 周早期预警（区分度 AUC 88.2%，全量 97.8%）→ 统一评价与行动建议（AHP+模糊质量分层 / 干预队列 / SD 仿真 / 10 条优化建议）→ 数据来源（六源构成与伦理边界）**。每屏先给一句结论、再给直观图表；指标卡全部是"学习者数 / 画像类数 / 用户数 / 城市数"这类人话。

### 形态画像分析看板

`persona_dashboard.html` 是面向「慕课 / 微课 / 短视频 / 课堂对照」四形态的交互式画像工具，单文件离线可用：

- **形态切换**：顶部标签在 MOOC（6,000 条）、微课（723 条）、短视频（1,469 条）、课堂对照（1,044 条）间切换，全部图表联动；
- **用户群体画像**：K-means 画像群按规模排序，展示每群人数、占比、达标率与五维行为中心雷达；点击任一画像群，雷达与散点同步高亮；
- **个体分布探索**：可切换 X/Y 轴的二维散点，按画像群或达标与否着色，悬停查看完整五维取值；
- **达标驱动与形态对比**：分形态 Logistic 系数、跨形态平均画像与 CV-AUC 对照，并显式标注各形态标签语义不同、不可混算。

所有数字直接取自 `04_运行结果/multimodal_results.json` 与 `multimodal_all.csv`，无仿真数据。

### 内容画像看板（回应"不识其人，即知其理"）

`profile_dashboard.html` 对国内内容级真实数据做画像刻画与差异检验，顶部切换标签、可排序表格与联动条形图：

- **观看时段偏好**：短视频 6,654 用户小时级主观看时段（峰值在晚间与下午，合计约 65%）；
- **兴趣爱好**：短视频 37 个一级类目观看占比 + 知识求知型/均衡杂食型/娱乐休闲型三类兴趣画像（行为代理，非心理测量）；
- **短视频用户属性（国内）**：城市等级（一线~五线）/ 性别 / 年龄 / 观看时段 / 城乡，365 座城市 6,654 用户的观看完成度与互动率；
- **天池倍速标签（国内）**：倍速 / 设备 / OS / 网络 / 预习（含"倍速伪影"口径修正：严格限定视频事件对照，检验后倍速与预习差异均不显著）。

数字取自 `04_运行结果/profile_results.json`（由 `03_源代码/profile_modeling.py` 实跑）；短视频/天池细粒度输入表为 `02_数据与数据集/shortvideo_profile.csv`、`tianchi_profile.csv`、`shortvideo_cat_watch.csv`（由 `build_profile_inputs.py` 从真实原始数据派生，随包交付、离线可复现）。所有群体差异为观察性关联，伦理边界在页面底部显式标注。

## 二、数字唯一来源

| 结果文件 | 内容 | 产生脚本 |
|---|---|---|
| `04_运行结果/up_*.json` 系列 | **主线**：六源统一管线聚类/挖掘/关联/预测/评价全指标 | `03_源代码/up_*.py`（8 阶段） |
| `04_运行结果/up_learners.csv / up_unified.csv / up_queues.csv / up_report.md` | 逐学习者画像表 / 合并总表 / 干预队列 / 汇总报告 | `03_源代码/up_*.py`（8 阶段） |
| `04_运行结果/multimodal_results.json` | 四形态跨场景测评（9,236 条） | `03_源代码/multimodal.py` |
| `04_运行结果/profile_results.json` | 内容级画像：时段/兴趣/短视频国内属性/天池倍速标签 | `03_源代码/profile_modeling.py` |

**全文无仿真数据**：六源合并总表回答"课程级行为与结果"；国内内容级证据（清华短视频 6,654 名用户、天池微课）回答"什么内容被持续观看"；
四形态对比（9,236 条）回答"换个场景还成立吗"。`seed=20260101` 仅用于算法可复现，不生成数据。

## 三、完整复现（在交付包根目录执行）

**推荐一键**：`python 03_源代码/run_all.py`（等价于下列全部步骤，自动按正确依赖顺序执行）。
以下命令供需要**单步调试**或**只重跑某一环**时使用：

```bash
pip install -r 03_源代码/requirements.txt

python 03_源代码/multimodal.py             # 四形态跨场景测评
python 03_源代码/multimodal_figures.py     # fig11—fig14
python 03_源代码/build_profile_inputs.py   # 派生短视频/天池细粒度画像输入表（随包已附，可选重跑）
python 03_源代码/profile_modeling.py       # 内容级画像（读随包派生表，离线可复现）

python 03_源代码/up_build_unified.py       # 统一管线 8 阶段（主线），也可用 run_all.py 一次跑完
python 03_源代码/up_cluster.py --device gpu
python 03_源代码/up_mine.py
python 03_源代码/up_associate.py
python 03_源代码/up_predict.py
python 03_源代码/up_evaluate.py
python 03_源代码/up_decide.py
python 03_源代码/up_report.py

python 05_图表与看板/build_dashboard.py    # 刷新主看板 dashboard.html
python 05_图表与看板/build_persona_dashboard.py  # 刷新形态画像看板
python 05_图表与看板/build_profile_dashboard.py  # 刷新内容画像看板
```

> `build_profile_inputs.py` 需要短视频/天池原始表（工作区内，不随包分发）重新派生细粒度画像输入表；
> 派生后的 `shortvideo_profile.csv` / `tianchi_profile.csv` / `shortvideo_cat_watch.csv` 已随包交付，
> **只重跑 `profile_modeling.py` 与看板不需要原始数据**。

## 四、目录结构

```
选题7_最终交付/
├── README.md
├── 01_研究报告/
│   ├── 《系统工程原理》实践项目研究报告-选题7.docx / .md
│   │                                   （六源统一口径：摘要+9章+参考文献+附录A-D，26 表/27 图/2 公式，
│   │                                    全部数字由 build_final_report.py 从 04_运行结果/ 实跑 JSON/CSV 注入）
│   └── build_final_report.py          报告生成器（不硬编码结果；重建命令见运行说明）
├── 02_数据与数据集/
│   ├── raw/                            六类真实公开数据集（按来源分目录，git 忽略）
│   │   ├── uci/                        UCI Student Performance（student-mat/por.csv）
│   │   ├── thu_shortvideo/             清华 FIB 短视频行为数据集本地子样本（WWW 2025）
│   │   ├── tianchi/                    阿里云天池高校慕课脱敏样本(2024–2025)
│   │   ├── harvardx/                   HarvardX-MITx Person Course Dataset
│   │   ├── assistments/                中科大 EduData（ASSISTments 2009）
│   │   └── oulad/                      OULAD 7 张原始表（GitHub 镜像，studentVle 为子集）
│   ├── real_learners.csv               OULAD 派生主表（统一管线的六源输入之一）
│   ├── mooc.csv / micro.csv / shortvideo.csv / classroom.csv / multimodal_all.csv
│   ├── shortvideo_narrow.csv          严格教育类窄口径样本
│   ├── shortvideo_profile.csv         短视频用户级画像输入 6,654 人（城市等级/性别/年龄/时段/互动）
│   ├── tianchi_profile.csv            天池事件级标签输入 5,000 条（倍速/设备/OS/网络/预习）
│   ├── shortvideo_cat_watch.csv       短视频类目观看汇总 37 类
│   ├── sources.json / sources_narrow.json        数据源与筛选口径
│   ├── 数据源清单.csv / 字段对齐表.csv / 字段字典.csv / 方法适配矩阵.csv
│   └── README.md
├── 03_源代码/
│   ├── run_all.py                     ★ 一键运行入口（--check 体检 / --device GPU）
│   ├── model_console.py               ★ 建模控制台服务（可交互改参/换数据重跑，标准库实现）
│   ├── common.py                      GPU 聚类引擎 ClusterEngine / 路径 / 日志 / 绘图公共库
│   ├── evaluators.py                  AHP/熵权唯一实现（统一管线与多形态链共用）
│   ├── up_adapters.py                 6 类真实数据源适配器（统一 schema + 事件流）
│   ├── up_stages.py                   统一管线全部阶段函数实现库
│   ├── up_build_unified.py … up_report.py  统一管线 8 个阶段 CLI（run_all 编排调用）
│   ├── multimodal.py                  四形态对齐、分布偏移、迁移与决策
│   ├── multimodal_figures.py          fig11—fig14
│   ├── build_real_learners.py         OULAD 原始表 → 统一管线输入主表
│   ├── build_profile_inputs.py        从真实原始数据派生短视频/天池细粒度画像输入表
│   ├── profile_modeling.py            内容级画像：时段/兴趣/短视频属性/天池标签
│   └── requirements.txt
├── 04_运行结果/
│   ├── up_unified.csv / up_learners.csv / up_clicks.csv   统一管线数据产物（主线）
│   ├── up_metrics.json / up_mining_summary.json / up_association_summary.json / up_eval_summary.json
│   ├── up_queues.csv / up_persona_recommendations.csv / up_optimization_suggestions.csv / up_report.md
│   ├── up_*.png                        统一管线图表 16 张
│   ├── multimodal_results.json         四形态测评
│   └── profile_results.json            内容级画像实算结果
├── 05_图表与看板/
│   ├── dashboard.html / dashboard.template.html / build_dashboard.py
│   │                                   （主看板：7 屏六源统一叙事，数据全部来自 up_* 实跑结果）
│   ├── persona_dashboard.html          形态画像看板（四形态切换 / 画像 / 散点联动）
│   ├── persona_dashboard.template.html / build_persona_dashboard.py
│   ├── profile_dashboard.html          内容画像看板（时段/兴趣/短视频属性/天池标签）
│   ├── profile_dashboard.template.html / build_profile_dashboard.py
│   ├── conclusion.md                  全图表解读手册（逐一标注算法/数据/结果/结论）
│   ├── model_console.html             建模控制台页面（由 03_源代码/model_console.py 托管）
│   └── figures/                        fig01—fig03（方法学）、fig11—fig14（多形态）、fig20—fig22（内容画像）
└── 06_支撑文档/
    ├── 数据说明.md / 多形态数据说明.md
    ├── 模型—代码对照表.md / 答辩速查卡.md / 运行说明.md
    ├── 生成式AI使用声明表.docx / 实践项目选题申报表.docx
```

## 五、报告框架（霍尔逻辑维七步）

摘要 → 1 问题界定 → 2 方法论与总体框架 → 3 支撑数据与分析对象（六源统一口径）→
**4 系统建模**（目标树 + ISM 递阶）→ **5 系统分析**（K-means 统一画像 + 内容级画像）→
**6 系统预测**（早期预警/GM(1,1)）→ **7 系统评价**（AHP+熵权+模糊）→ **8 系统决策与优化建议**（干预队列 + SD 仿真）→ 9 可验证成果与结论 →
参考文献 → 附录 A 数据说明 / B AI 声明 / C 代码清单 / D 贡献度。

## 六、提交前需人工补齐

1. 封面与附录 D 的组员姓名、学号、贡献度与签名；
2. 附录 B 生成式 AI 声明表的手写签名与日期；
3. `06_支撑文档/实践项目选题申报表.docx` 的指导教师意见。

> 六源重构遗留待办已于 2026-11-21 全部完成：研究报告（docx/md + build_final_report.py）
> 已按六源统一口径重写生成；`06_支撑文档/` 五份文档已通读修订为六源口径。
