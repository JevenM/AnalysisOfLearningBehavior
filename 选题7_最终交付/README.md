# 选题 7 · 基于线上数据的学习者行为分析 —— 最终交付包

《系统工程原理》课程实践项目（2026 年实践项目指南 · 选题 7）。
本目录是**自洽、可复现、可现场改参重跑**的完整交付物：研究报告、源码、数据、结果、图表看板与支撑文档共用同一组实跑结果。

> **版本说明（2026-11 双管线合并）**：本交付包与另一套单入口分析管线（9 阶段统一
> 数据管线，含 GPU 加速聚类）已完成合并——共用 `03_源代码/common.py`（GPU 聚类引擎）
> 与 `evaluators.py`（AHP/熵权/TOPSIS 唯一实现），并新增 `raw/` 六类真实数据源目录与
> 统一管线 `up_*` 脚本组。两条证据链独立可跑、互不影响；改造后原主链结果 JSON 与
> 改造前逐位一致。

## 零、一键运行（推荐从这里开始）

参照借鉴仓库 `main.py` 的单入口模式，本项目提供 `run_all.py`：一条命令复现整条分析链（结果 → 图表 → 看板 → 报告），无需逐条记忆脚本顺序。

```bash
# 1. 安装依赖（一次性）
python -m pip install -r 03_源代码/requirements.txt

# 2. 先体检（只读，确认数据与依赖齐全）
python 03_源代码/run_all.py --check

# 3. 一键全链路（默认用随包派生数据，完全离线可复现）
python 03_源代码/run_all.py

# 常用变体
python 03_源代码/run_all.py --web                 # 全链路后顺便启动交互平台 http://127.0.0.1:8765
python 03_源代码/run_all.py --skip-report         # Word 正占用 docx 时跳过报告重建
python 03_源代码/run_all.py --from-raw --oulad-dir <OULAD原始目录>   # 含 443MB 原始表派生（可选）
python 03_源代码/run_all.py --device gpu          # 统一管线聚类启用 GPU 加速（合并新增）
python 03_源代码/run_all.py --skip-unified        # 只跑 OULAD 主链，跳过统一管线（合并新增）
```

一键链的执行顺序与依赖：`multimodal → app.py --batch（挂载多形态结果）→ forecast → fuzzy → profile → demo_runs → **统一管线 up_1..up_8（合并新增）** → figures → 三个看板 → 报告`，每步独立子进程、失败即停、可重复运行（幂等覆盖）。默认**不重新下载、不生成仿真数据**；`--from-raw` 才走 OULAD 原始表派生。

### 统一管线（合并新增，`03_源代码/up_*.py`）

独立于 OULAD 主线的第二条全链路，融合 `raw/` 下 **6 类真实公开数据集**
（UCI / 清华短视频 / 天池慕课 / HarvardX-MITx / ASSISTments / OULAD，共约 6.7 万学习者）：

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
加速范围为 KMeans / 轮廓系数 / 最近质心分配（GTX 1050 Ti 4GB 实测可用）。
聚类与选 K 全链共用 `common.ClusterEngine`；OULAD 主链（analysis.py / multimodal.py）
的 KMeans 亦已接入同一引擎，CPU 后端下结果与合并前逐位一致。

## 一、先看什么

| 交付物 | 路径 |
|---|---|
| 主报告（按附件 A 模板，9 章 + 附录 A—D） | `01_研究报告/《系统工程原理》实践项目研究报告-选题7.docx` |
| 可编辑源稿 | `01_研究报告/《系统工程原理》实践项目研究报告-选题7.md` |
| **主看板（推荐先看）** | `05_图表与看板/dashboard.html`（8 屏结论先行的画像解读，先给结论再给证据，零建模基础也能看懂；含时钟图/甜甜圈/金字塔/树图等多种信息图；第 8 屏为统一管线六源融合分析（合并新增），统一管线未跑时自动隐藏；建模细节与在线验证区可折叠展开） |
| **学习者画像分析看板** | `05_图表与看板/persona_dashboard.html`（形态切换 + 画像群探索 + 散点联动，详见下） |
| **六维画像看板** | `05_图表与看板/profile_dashboard.html`（地区 / 专业背景 / 学习习惯 / 兴趣爱好 / 短视频用户属性 / 天池倍速标签，详见下） |
| **全图表解读手册** | `05_图表与看板/conclusion.md`（38 张图逐一标注：算法全称、输入数据、实跑结果、怎么读、与后续环节关系、结论；按系统工程分析步骤与代码运行流程组织） |
| 可交互验证平台 | `03_源代码/app.py --no-browser`（浏览器打开 http://127.0.0.1:8765） |

### 三看板导读：区别、联系与讲解顺序（2026-11 数据溯源核查后整理）

**数据可信度承诺**：三个看板的每一个数字均由构建脚本从 `04_运行结果/` 实跑 JSON/CSV 读取注入（构建脚本内无编造值），已做 39 项逐数字溯源比对全部一致；`seed=20260101` 只用于分层抽样（可复现），不生成任何仿真数据。若怀疑某个数字，可在看板内打开"数据说明"屏核对来源文件名。

| | 主看板 `dashboard.html` | 形态画像看板 `persona_dashboard.html` | 四维画像看板 `profile_dashboard.html` |
|---|---|---|---|
| **回答什么** | 全景结论：分几类、习惯与结果的关系、预算怎么花 | 同一方法在 4 种学习形态下是否都成立 | "这个人是谁"：地区/背景/习惯/兴趣四维刻画 |
| **数据粒度** | 课程级（OULAD 20,105 条）+ 六源融合 66,644 人 | 多形态样本 9,236 条（MOOC 6,000/微课 723/短视频 1,469/课堂对照 1,044） | OULAD 分组检验 + 国内短视频 6,654 用户 + 天池微课 |
| **交互方式** | 8 屏顺序叙事 + 底部折叠的在线验证控制台 | 四形态标签切换、画像群点击联动散点 | 六维度标签、可排序表格 + 联动条形图 |
| **数据来源** | results.json / forecast / fuzzy / profile / up_*.json | multimodal_results.json + multimodal_all.csv | profile_results.json |

**联系**：三看板互链（主看板头部可跳另两个）；**同一套实跑结果**的三种视角——主看板给"结论"，两个专题看板给"证据细节"（形态间对比、维度间差异检验）。任何一屏的数字都能在 conclusion.md（全图表解读手册）找到算法与数据来源解释。

**讲解顺序（演示/答辩建议）**：① 从 `dashboard.html` 第 1 屏"四类学习者人物卡"开场（3 分钟讲完四类人是谁、谁有风险）；② 第 5-6 屏早期预警与预算决策收尾（这是决策者最关心的"怎么办"）；③ 观众追问"换个场景还成立吗"→ 打开 `persona_dashboard.html` 切到短视频/课堂形态；④ 追问"人群差异可靠吗"→ 打开 `profile_dashboard.html` 展示 χ² 检验与 Cramér's V；⑤ 深挖细节时展开主看板底部"进阶验证"区的建模图表。

### 主看板：结论先行的人话叙事（推荐先看）

`dashboard.html` 已按"让不懂建模的人也能看懂"重构为 8 屏叙事：**四类学习者人物卡 → 习惯与通过率 → 兴趣与观看时段 → 人群差异 → 第4周早期预警 → 分层行动建议 → 数据说明 → 六源融合统一分析（合并新增）**，每屏先给一句绿色"一句话结论"、再给直观图表；指标卡全部是"记录数 / 画像类数 / 用户数 / 城市数"这类人话。第 8 屏展示统一管线（六类真实数据集合并 66,644 名学习者）的质量分层、行为序列模式、时间模式、GM(1,1) 趋势、特征重要性、早期预警、关联规则与 SD 策略，数据取自 `04_运行结果/up_*.json`（未跑统一管线时该屏自动隐藏、看板降级为主链数据）。专业读者可展开页面底部"进阶验证"，其中完整保留在线改参重跑控制台与 ISM/ROC/AHP 等建模图表。

### 学习者画像分析看板

`persona_dashboard.html` 是面向「慕课 / 微课 / 短视频」三类研究对象的交互式画像工具，单文件离线可用：

- **形态切换**：顶部标签在 MOOC（6,000 条）、微课（723 条）、短视频（1,469 条）、课堂对照（1,044 条）间切换，全部图表联动；
- **用户群体画像**：K-means 画像群按规模排序，展示每群人数、占比、达标率与五维行为中心雷达；点击任一画像群，雷达与散点同步高亮；
- **个体分布探索**：可切换 X/Y 轴的二维散点，按画像群或达标与否着色，悬停查看完整五维取值，来源细分随形态切换为平台 / 内容类目 / 投入分层；
- **达标驱动与形态对比**：分形态 Logistic 系数、跨形态平均画像与 CV-AUC 对照，并显式标注各形态标签语义不同、不可混算。

所有数字直接取自 `04_运行结果/multimodal_results.json` 与 `multimodal_all.csv`，无仿真数据；`seed=20260101` 仅用于分层抽样。

### 六维画像看板（回应"不识其人，即知其理"）

`profile_dashboard.html` 针对报告中关心的六类画像维度做真实数据刻画与差异检验，顶部切换标签、可排序表格与联动条形图：

- **地区差异**：OULAD 13 类英国地域的参与与通过率对照（χ²=122.085，p<1e-6，但 Cramér's V=0.078 属弱关联）；
- **专业背景**：入学学历（5 类）、课程模块（7 个，OULAD 编码）、选课门数的成效梯度，明确为代理口径；
- **学习习惯**：五类行为节律分型（规律深耕型 75.7% → 浅触即离型 24.6%）+ 短视频 6,654 用户小时级观看时段偏好；
- **兴趣爱好**：短视频 37 个一级类目观看占比 + 知识求知型/均衡杂食型/娱乐休闲型三类兴趣画像（行为代理，非心理测量）；
- **短视频用户属性（国内）**：城市等级（一线~五线）/ 性别 / 年龄 / 观看时段 / 城乡 365 座城市 6,654 用户的观看完成度与互动率；
- **天池倍速标签（国内）**：倍速 / 设备 / OS / 网络 / 预习（含"倍速伪影"口径修正：严格限定视频事件，检验后倍速与预习差异均不显著）。

**地域来源说明**：画像中既有**国内数据**（短视频 365 座中国城市、天池国内脱敏微课），也有英国 OULAD 13 大区；二者分层呈现、不合并、不互推。数字取自 `04_运行结果/profile_results.json`（由 `03_源代码/profile_modeling.py` 实跑）；短视频/天池细粒度输入表为 `02_数据与数据集/shortvideo_profile.csv`、`tianchi_profile.csv`、`shortvideo_cat_watch.csv`（由 `build_profile_inputs.py` 从真实原始数据派生，随包交付、离线可复现）。所有群体差异为观察性关联，伦理边界在页面底部显式标注。

## 二、数字唯一来源

| 结果文件 | 内容 | 产生脚本 |
|---|---|---|
| `04_运行结果/results.json` | OULAD 真实主线：清洗、ISM、画像、无泄漏预测、评价、决策、预算灵敏度 | `03_源代码/app.py --batch` |
| `04_运行结果/multimodal_results.json` | 六源真实数据多形态测评（9,236 条） | `03_源代码/multimodal.py` |
| `04_运行结果/forecast_results.json` | OULAD 周序列预测与早期预警 | `03_源代码/forecast.py` |
| `04_运行结果/fuzzy_results.json` | 模糊综合评价与 TOPSIS 一致性 | `03_源代码/fuzzy_eval.py` |
| `04_运行结果/profile_results.json` | 六维画像：地区/背景/习惯/兴趣/短视频国内属性/天池倍速标签 | `03_源代码/profile_modeling.py` |
| `04_运行结果/demo_runs/summary.json` | 三组改参重跑前后对照 | `03_源代码/demo_runs.py` |
| `04_运行结果/up_metrics.json / up_*_summary.json` | 统一管线：聚类/挖掘/关联/预测/评价全指标 | `03_源代码/up_*.py`（8 阶段） |
| `04_运行结果/up_learners.csv / up_report.md` | 统一管线逐学习者画像表 / 汇总报告 | `03_源代码/up_*.py`（8 阶段） |

**两条并重的真实证据链（全文无仿真）**：课程级证据用 OULAD 派生的 20,105 条真实记录（含地区/教育/性别/行为/真实结业标签）回答“课程能否通过”；
国内场景与内容级证据以清华 FIB 短视频的 6,654 名国内用户（365 城、城乡/年龄/时段/内容偏好）与天池微课（倍速/设备/预习）为主体，
结合 MOOC、课堂对照六源共 9,236 条多形态样本，回答“什么内容被持续观看”。两条链分层建模、不混算。
`seed=20260101` 仅用于算法可复现，不生成数据。

## 三、完整复现（在交付包根目录执行）

**推荐一键**：`python 03_源代码/run_all.py`（等价于下列全部步骤，自动按正确依赖顺序执行）。
以下命令供需要**单步调试**或**只重跑某一环**时使用；注意 `app.py --batch` 必须在 `multimodal.py` 之后，才能把多形态结果挂进 `results.json`。

```bash
pip install -r 03_源代码/requirements.txt

python 03_源代码/build_real_learners.py --oulad-dir <OULAD原始目录> --out-dir 02_数据与数据集  # 派生真实主表（已随包可跳过）
python 03_源代码/multimodal.py             # 六源多形态测评（须先于 app.py --batch）
python 03_源代码/app.py --batch            # OULAD 真实主线 → results.json（自动挂载多形态结果）
python 03_源代码/build_forecast_data.py --oulad-dir <OULAD原始目录> --out-dir 02_数据与数据集
python 03_源代码/forecast.py               # 序列预测 + 早期预警
python 03_源代码/fuzzy_eval.py             # 模糊综合评价
python 03_源代码/build_profile_inputs.py   # 派生短视频/天池细粒度画像输入表（随包已附，可选重跑）
python 03_源代码/profile_modeling.py       # 六维画像（读随包派生表，离线可复现）
python 03_源代码/demo_runs.py              # 三组改参快照
python 03_源代码/multimodal_figures.py     # fig11—fig14
python 05_图表与看板/build_dashboard.py    # 刷新 dashboard.html
python 05_图表与看板/build_persona_dashboard.py  # 刷新画像分析看板 persona_dashboard.html
python 05_图表与看板/build_profile_dashboard.py  # 刷新六维画像看板 profile_dashboard.html
python 01_研究报告/build_final_report.py   # 重新生成报告 docx / md
```

> `build_real_learners.py` / `build_forecast_data.py` 需要 OULAD 原始表（约 443 MB，不随包分发）；
> 派生后的真实主表与预测小表已随包交付，**只重跑主线与预测不需要原始数据**。
> `build_profile_inputs.py` 需要短视频/天池原始表（工作区内，不随包分发）重新派生细粒度画像输入表；
> 派生后的 `shortvideo_profile.csv` / `tianchi_profile.csv` / `shortvideo_cat_watch.csv` 已随包交付，
> **只重跑 `profile_modeling.py` 与看板不需要原始数据**。

## 四、目录结构

```
选题7_最终交付/
├── README.md
├── 01_研究报告/
│   ├── 《系统工程原理》实践项目研究报告-选题7.docx / .md
│   └── build_final_report.py          由五份结果 JSON 生成报告（不硬编码结果）
├── 02_数据与数据集/
│   ├── raw/                            [合并新增] 六类真实公开数据集（按来源分目录，git 忽略）
│   │   ├── uci/                        UCI Student Performance（student-mat/por.csv）
│   │   ├── thu_shortvideo/             清华 FIB 短视频行为数据集本地子样本（WWW 2025）
│   │   ├── tianchi/                    阿里云天池高校慕课脱敏样本(2024–2025)
│   │   ├── harvardx/                   HarvardX-MITx Person Course Dataset
│   │   ├── assistments/                中科大 EduData（ASSISTments 2009）
│   │   └── oulad/                      OULAD 7 张原始表（GitHub 镜像，studentVle 为子集）
│   ├── real_learners.csv               OULAD 真实主表 20,105 条（+ real_learners_meta.json）
│   ├── mooc.csv / micro.csv / shortvideo.csv / classroom.csv / multimodal_all.csv
│   ├── shortvideo_narrow.csv          严格教育类窄口径样本
│   ├── shortvideo_profile.csv         短视频用户级画像输入 6,654 人（城市等级/性别/年龄/时段/互动）
│   ├── tianchi_profile.csv            天池事件级标签输入 5,000 条（倍速/设备/OS/网络/预习）
│   ├── shortvideo_cat_watch.csv       短视频类目观看汇总 37 类
│   ├── weekly_activity.csv / early_warning.csv   预测用派生小表
│   ├── sources.json / sources_narrow.json        数据源与筛选口径
│   ├── 数据源清单.csv / 字段对齐表.csv / 字段字典.csv / 方法适配矩阵.csv
│   └── README.md
├── 03_源代码/
│   ├── run_all.py                     ★ 一键运行入口（--check 体检 / --web 交互 / --from-raw 原始派生 / --device GPU / --skip-unified）
│   ├── common.py                      [合并新增] GPU 聚类引擎 ClusterEngine / 路径 / 日志 / 绘图公共库
│   ├── evaluators.py                  [合并新增] AHP/熵权/TOPSIS 唯一实现（analysis 与 multimodal 共用）
│   ├── up_adapters.py                 [合并新增] 6 类真实数据源适配器（统一 schema + 事件流）
│   ├── up_stages.py                   [合并新增] 统一管线全部阶段函数实现库
│   ├── up_build_unified.py … up_report.py  [合并新增] 统一管线 8 个阶段 CLI（run_all 编排调用）
│   ├── app.py                         Web / CLI / 批跑交互验证平台
│   ├── analysis.py                    清洗、画像、无泄漏预测、评价、0-1 规划、ISM
│   ├── build_real_learners.py         OULAD 原始表 → 真实学习者主表（替换仿真主线）
│   ├── multimodal.py                  多形态对齐、分布偏移、迁移与决策
│   ├── forecast.py                    序列特征、SES/Holt/GM(1,1)、早期预警
│   ├── fuzzy_eval.py                  模糊综合评价
│   ├── build_profile_inputs.py        从真实原始数据派生短视频/天池细粒度画像输入表
│   ├── profile_modeling.py            六维画像：地区/背景/习惯/兴趣/短视频属性/天池标签
│   ├── figures.py / multimodal_figures.py
│   ├── build_forecast_data.py         OULAD → 预测用派生表
│   ├── demo_runs.py / requirements.txt
├── 04_运行结果/
│   ├── results.json / results_latest.json
│   ├── multimodal_results.json / forecast_results.json / fuzzy_results.json
│   ├── profile_results.json            六维画像实算结果（含国内地域来源说明）
│   └── demo_runs/{A_基准,B_改权重,C_改K与预算,summary}.json
├── 05_图表与看板/
│   ├── dashboard.html / dashboard.template.html / build_dashboard.py
│   │                                   （主看板第 8 屏内嵌统一管线 up_*.json 数据，缺失时自动降级）
│   ├── persona_dashboard.html          交互式画像分析看板（形态切换 / 画像 / 散点联动）
│   ├── persona_dashboard.template.html / build_persona_dashboard.py
│   ├── profile_dashboard.html          六维画像看板（地区/背景/习惯/兴趣/短视频属性/天池标签）
│   ├── profile_dashboard.template.html / build_profile_dashboard.py
│   ├── conclusion.md                  全图表解读手册（38 张图逐一标注算法/数据/结果/结论）
│   └── figures/fig01..fig22.png       22 幅报告插图（含画像 fig17—fig22）
└── 06_支撑文档/
    ├── 数据说明.md / 多形态数据说明.md
    ├── 模型—代码对照表.md / 答辩速查卡.md / 运行说明.md
    ├── 生成式AI使用声明表.docx / 实践项目选题申报表.docx
```

## 五、报告框架（霍尔逻辑维七步）

摘要 → 1 问题界定 → 2 方法论与总体框架 → 3 支撑数据与分析对象 →
**4 系统建模**（目标树 + ISM 递阶）→ **5 系统分析**（K-means 画像 + 六维画像）→
**6 系统预测** → **7 系统评价** → **8 系统决策与优化建议** → 9 可验证成果与结论 →
参考文献 → 附录 A 数据说明 / B AI 声明 / C 代码清单 / D 贡献度。

> 按教材方法论单独成章："建模"只放目标树与 ISM 这类因素结构模型（自上而下搭骨架），
> "分析"是从数据归纳群体（自下而上），所以 K-means 画像在第 5 章而不在第 4 章；七步映射见报告表 2。
> 早期曾收敛为 7 章，现按作业要求拆回 9 章；WSR 并入方法论、PEST/SWOT 与序列外推/GM/跨形态迁移/
> TOPSIS/预算扫描/SD 的结果保留在结果文件与看板折叠区，不进方法主线，数字未删。

## 六、提交前需人工补齐

1. 封面与附录 D 的组员姓名、学号、贡献度与签名；
2. 附录 B 生成式 AI 声明表的手写签名与日期；
3. `06_支撑文档/实践项目选题申报表.docx` 的指导教师意见。

AHP 判断矩阵、ISM 邻接矩阵与分层干预建议由组员依据课程理论独立确定，须能现场解释；AI 仅用于文字整理、检索与代码辅助。
