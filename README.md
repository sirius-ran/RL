# Boiler Reheat Temperature RL Control

这个项目用于再热汽温/烟气挡板控制的离线建模、强化学习训练和评估。

## 目录结构

```text
data/
  raw/                         原始训练与测试数据
  segments/                    闭环分段数据
models/
  env/                         LightGBM 环境模型和物理参数
  actors/                      BC/RL 训练得到的 Actor 权重
results/
  training/                    环境建模和 BC 训练图
  evaluation/                  评估时序 CSV
  figures/                     控制效果报告图
  checkpoints/                 训练过程保存的最优轨迹
src/
  train_shadow_model.py        训练直接预测 y(t+1) 的影子环境模型
  train_env_model.py           训练三阶惯性 + 扰动残差环境模型
  pretrain_actor_bc.py         行为克隆预训练 Actor
  train_controller_ddpg.py     DDPG + MPC 前瞻奖励训练控制器
  evaluate_controller.py       在保留测试数据上评估控制器
legacy/
  evaluate_generalization_old.py  历史泛化测试脚本，当前依赖缺失
```

## 推荐流程

从项目根目录运行：

```bash
python src/train_shadow_model.py
python src/train_env_model.py
python src/pretrain_actor_bc.py
python src/train_controller_ddpg.py
python src/evaluate_controller.py
```

`src/train_controller_ddpg.py` 默认启用 Q-filtered DDPG + BC actor regularization，产物标签为 `bcQReg`。如需回到普通 BC 正则，对脚本顶部的 `BC_REG_Q_FILTER` 设为 `False`；如需回到纯 RL，将 `USE_BC_REG` 设为 `False`。

当前已有模型和结果已经整理到 `models/` 和 `results/`，因此如果只是查看现有控制效果，可以直接看：

- `results/evaluation/full_controller_timeseries.csv`
- `results/figures/report_images/`
- `results/training/`

## 当前已保存评估结果

基于 `results/evaluation/full_controller_timeseries.csv`：

| 方法 | MAE | RMSE | +/-2 degC | +/-5 degC |
| --- | ---: | ---: | ---: | ---: |
| RL 控制 | 2.54 degC | 3.21 degC | 48.01% | 87.19% |
| 历史手操 | 7.32 degC | 9.26 degC | 18.53% | 42.36% |

## 备注

- 代码中的路径已经改为基于项目根目录自动定位，不再依赖当前工作目录。
- `legacy/evaluate_generalization_old.py` 引用了当前目录中不存在的旧文件，暂时作为历史记录保留。
- 目前没有整理依赖版本；如果要复现实验，建议后续补一个 `requirements.txt` 或 conda 环境文件。
