## 问题回答
### 1、当前代码使用的数据集有哪些？

改造前代码没有使用任何真实公开数据集。build_unified_dataset()（原 main.py:505-509）的全部输入来自 5 个内置合成生成器：synthetic_oulad、synthetic_interaction、synthetic_chatbot_events、synthetic_code_submissions、synthetic_clickstream，人口属性（性别/地区/学历）和缺失特征也全部是随机生成的。data/ 目录的真实数据扫描只在有文件时生效，且当时仅有清华短视频部分数据未接入管线。

### 2、有无使用虚拟或合成数据？

有，且是主要数据来源。 已按要求全部取消：

删除了全部 5 个合成生成器及 --n 参数、add_demographics 的随机属性合成、load_clickstream 的合成回退；
删除了 data/simulate_oulad.py 和旧合成数据跑出的 data/learner_profiles.csv（避免被误当真实数据加载）；
现在管线只加载真实公开数据集，仅对个别数据集未提供的标签列（完成率/最终成绩）由该来源的真实行为派生代理值，并在代码中明确标注。
### 3、确定 K-Means 的 K 的代码是否有问题？

有，4 处问题，已修复（main.py 的 cluster_personas）：

肘部法只画图不参与选择——旧代码算完 inertia 仅绘图，自动选 K 只取轮廓系数 argmax，与文档“肘部法+轮廓系数选 K”不符。现改为肘部拐点（kneedle 垂距法）与轮廓系数两路证据归一化后等权融合；
k 无合法性校验——旧代码 --k 4 直接传入，k≥n 时 KMeans 报错、k=1 时 silhouette 报错。现夹取到 [2, min(8, n-1)] 并告警；
大数据集不可行——silhouette 是 O(n²)，HarvardX 5 万样本直接算会爆内存。现固定种子采样子集（1.5 万）拟合与评估，全量用最近质心分配标签；
n_init 不一致（选 K 用 10、最终聚类用 20）——已统一。
要求完成情况
统一接口：数据层重构为适配器模式，每个适配器返回 (features_frames, clicks|None)，结果缓存复用（特征表与事件流共用一次加载），学生 ID 加源前缀防冲突。

#### 已接入的 6 类数据源（管线实际运行验证）：

数据集	规模	位置
UCI Student Performance	395+649 学生	data/real/uci_student/（已下载解压）
清华 FIB 短视频（WWW 2025）	6,654 用户 / 79.4 万条行为	data/real/thu_shortvideo/
阿里天池高校慕课(2024–2025)	715 学习者 / 5,000 事件	_repos/阿里云天池…/
HarvardX/MITx Person Course	抽样 5 万用户	_repos/HarvardX_MITx…/
EduData 集成包	ASSISTments 2009：4,217 学生 / 40 万条答题	data/real/edudata/（已从 base.ustc.edu.cn 下载）
data/ 通用扫描	CSV/XLSX/JSONL/代码提交	保留
注：GitHub 直连失败，EduData 仓库代码经 ghproxy.net 镜像下载至 data/real/edudata/repo/；其中 math23k/OpenLUNA 是题库无学生行为记录，故接入其托管的 ASSISTments 真实答题数据。清华全量数据集需在官网登录下载，本地已有官方子样本。

#### 运行结果（真实数据结论，62,630 名学生）：

自动选 K=4（肘部拐点与轮廓系数一致指向 4，Silhouette=0.571）；
四类画像：At-Risk/流失风险 80.6%（HarvardX 大量零活跃注册者是真实分布）、独立深度学习者 15.7%、引导型 2.5%、低投入高成绩型 1.2%；
行为→效果：resource_clicks(r=0.588)、active_days(r=0.54) 与成绩正相关；高投入组完成率 +50%、成绩 +12.5 分；
预测层：RF 最终成绩 R²=0.793、完成判定 acc=0.966，早期 4 特征即可达 AUC 0.872；
碎片化证据：90.3% 的事件流用户只看短视频；短视频完播率 0.347 略低于慕课长视频 0.374，但慕课用户 42.8% 使用倍速；
GM(1,1) 趋势：5697 平稳 / 1410 上升 / 262 下降，模型精度 99.9% 合格。