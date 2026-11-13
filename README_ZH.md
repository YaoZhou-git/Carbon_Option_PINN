# 碳期权定价代码

论文 *Physics-Informed Neural Networks for carbon options: Multi-period pricing and Intertemporal Risk* 的计算代码。

## 目录

| 目录 | 内容 |
|---|---|
| single_period | 单期PINN与蒙特卡洛模型、碳价动态、期限分析及保存结果 |
| multi_period | 一至四因子模型、PINN训练与推理、方差匹配、跨期敏感度及参考定价 |
| empirical | 两期似然估计、统计推断、估计参数定价及实证绘图 |
| controls/black76 | Black（1976）期货期权期限比较 |
| data | 数据获取与输入要求 |
| docs/RESULTS_MAP.md | 论文图表与计算入口的对应关系 |

## 使用

1. 使用Python 3.10或更新版本，按 `requirements.txt` 安装依赖；执行notebook时另安装 `requirements-notebooks.txt`。详细环境配置见英文README。
2. 按各模块README运行模型推理、绘图、训练或估计。已提供保存的模型与数值参考输出。
3. 实证使用2021年观测的Dec-2022与Dec-2023 EUA期货数据，输入格式与运行步骤见 `empirical/README.md`。

## 数据

原始行情来自机构订阅的Refinitiv Datastream，受数据供应商许可限制，不随代码公开。具有相应授权的研究人员可自行获取数据并在本地运行实证。文件夹包含聚合参数估计结果和模型计算结果；原始行情及逐日变换数据应留在本地，不纳入版本控制。

使用代码时请引用对应论文。
