# 02_数据与数据集

本目录是报告与全部源码共用的唯一数据入口。文件名与口径固定，替换同名文件即可改数据重跑。

## 零、raw/ 原始数据集目录（双管线合并新增）

按**来源分类**存放六类真实公开数据集，供统一管线 `up_*.py`（及 `--from-raw` 派生）使用；本目录已被 git 忽略（大体量原始数据不入库），重新获取方式如下：

| 子目录 | 数据集 | 规模 | 获取方式 |
|---|---|---|---|
| `raw/uci/` | UCI Student Performance（student-mat/por.csv） | 395+649 人 | https://archive.ics.uci.edu/dataset/320/student+performance.zip（zip 内含 student.zip，需二次解压） |
| `raw/thu_shortvideo/` | 清华 FIB 短视频行为数据集本地子样本（WWW 2025） | 6,654 用户 / 79 万条 | http://fi.ee.tsinghua.edu.cn/datasets/short-video-dataset （官网登录下载） |
| `raw/tianchi/` | 阿里云天池《中国高校慕课课程多维画像与学习行为脱敏样本(2024–2025)》 | 715 学习者 / 5,000 事件 | 天池数据集页面下载（courses.xls / user_learning_behavior.xls） |
| `raw/harvardx/` | HarvardX-MITx Person Course Dataset | 5 万抽样 / 全量 32 万 | Harvard Dataverse doi:10.7910/DVN/26147（HMXPC13_DI_v2_5-14-14.csv；随包附清洗版 hmx_cleaned.csv） |
| `raw/assistments/` | 中科大 EduData 集成包（ASSISTments 2009） | 4,217 学生 / 40 万答题 | http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip 或 ghproxy 镜像拉取 https://github.com/bigdata-ustc/EduData |
| `raw/oulad/` | Open University Learning Analytics Dataset（OULAD） | 7 张表 / studentVle 子集 | 官网 https://analyse.kmi.open.ac.uk/open_dataset （官方 CDN 已迁移，当前经 GitHub 镜像 ShahzaibWaseem/OULAD-Dataset-Analysis 获取；镜像 studentVle 约为官方子集，含脏值已由适配器容错） |

> 注意：`raw/oulad/studentVle.csv` 与 vle 维表列名为官方口径 `id_site`；镜像数据
> `sum_click`/`score` 列含少量脏值（非数字字符），`up_adapters.load_oulad` 已做
> `to_numeric(errors="coerce")` 容错并记录有效规模。`raw/harvardx/` 与 `raw/tianchi/`
> 的文件同时供同事主链 `build_profile_inputs.py` 使用。


## 一、两类真实数据（全文无仿真）

| 数据块 | 文件 | 规模 | 用途 | 来源 |
|---|---|---|---|---|
| OULAD 真实主线 | `real_learners.csv`（+ `real_learners_meta.json`） | 20,105 条 / 18,634 人 / 13 门课 | 画像、无泄漏预测、课程评价、干预决策主结果 | `03_源代码/build_real_learners.py` 从 OULAD 5 张原始表派生 |
| 真实多形态 | `mooc.csv`、`micro.csv`、`shortvideo.csv`、`classroom.csv` | 6,000 / 723 / 1,469 / 1,044 | 分布偏移、迁移、跨形态评价与决策 | 6 个公开数据源，见 `sources.json` |
| 真实多形态（合并） | `multimodal_all.csv` | 9,236 条 | 统一入口，`multimodal.py` 默认读取 | 同上 |
| 短视频窄口径 | `shortvideo_narrow.csv` | 404 条 | 严格教育类稳健性对照 | 清华 FIB，root_id=14 |
| 预测派生 | `weekly_activity.csv` | 36 行 | OULAD 周人均点击与活跃率序列 | `build_forecast_data.py` 从 OULAD 派生 |
| 预警派生 | `early_warning.csv` | 20,105 行 | 2/4/6 周累计行为 + 真实结果标签 | 同上 |
| 短视频画像输入（国内） | `shortvideo_profile.csv` | 6,654 行 / 20 列 | 城市等级/性别/年龄/时段/城乡 × 观看完成度与互动 | `build_profile_inputs.py` 从清华 FIB 原始派生 |
| 短视频类目汇总 | `shortvideo_cat_watch.csv` | 37 行 | 一级类目观看时长/互动汇总 | 同上 |
| 天池画像输入（国内） | `tianchi_profile.csv` | 5,000 行 / 18 列 | 倍速/设备/OS/网络/预习 × 进度增益 | `build_profile_inputs.py` 从天池原始派生 |

## 二、字段口径

- 多形态样本统一 12 列：`form, unit, unit_id, engage, persist, complete, interact, context, label, label_name, raw_completeness, alt_label`。
- 五维指标均为**形态内标准化**值，只表示形态内相对位置，不是跨平台绝对水平。
- 四类标签语义不同：MOOC 认证/通过、微课单元测验通过、短视频后 2 日知识类回访、课堂对照期末 G3≥12；只作同构代理。
- OULAD 主表 25 列：人口属性（地区/学历/性别/年龄）+ 9 维行为特征 + 前期行为（`early_w2_*`/`early_w4_*`）+ 真实 `final_result`/`certificate` 标签；派生代理口径见 `real_learners_meta.json`。
- 预测只用前期行为 + 人口属性（无泄漏特征），quiz_avg/final_grade/assign_rate 与结果同期同生，禁止入模；人口属性只作分组变量，不参与聚类距离。
- 真实数据无预设潜类别，故不计算 ARI，改以画像真实通过率梯度验证可解释性。

## 三、元数据与字典

- `sources.json` / `sources_narrow.json`：数据源、下载日期、筛选规则、字段可得率、窗口与局限。
- `数据源清单.csv`、`字段对齐表.csv`、`字段字典.csv`、`方法适配矩阵.csv`：来源核验与选题字段逐项对应；`数据源清单.csv` 仅列报告实际采用的 6 个数据源及报告提及但未入模的 EduData，未进入报告的早期候选源已删除。
- 详细说明见 `../06_支撑文档/数据说明.md` 与 `../06_支撑文档/多形态数据说明.md`。

## 四、合规

所有数据脱敏，原始大体量日志（OULAD studentVle 约 433 MB、多形态原始约 970 MB）不随包分发；
派生后的真实主表与预测小表已随包交付，可用公开脚本复现。
数据集登录凭据未写入任何文件；个体层面结果仅限支持性干预，不用于评价或排名。
