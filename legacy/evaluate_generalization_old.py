"""
Legacy note: this script was moved out of the active workflow because it references files not present in this workspace.
M1_full 控制器在多个数据集上的泛化性能测试
"""
import os
import sys
import importlib.util
import random as _random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────
# 复用 RL_dyna_5m.py 的环境定义
# ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("RL_dyna_5m", "RL_dyna_5m.py")
rl_dyna = importlib.util.module_from_spec(_spec)
sys.modules["RL_dyna_5m"] = rl_dyna
_spec.loader.exec_module(rl_dyna)

BoilerGymEnvPhysics = rl_dyna.BoilerGymEnvPhysics
ShadowAugmentor     = rl_dyna.ShadowAugmentor
decode_action       = rl_dyna.decode_action
model_d             = rl_dyna.model_d
params              = rl_dyna.params
shadow_model        = rl_dyna.shadow_model
COL_U               = rl_dyna.COL_U
COL_LOAD            = rl_dyna.COL_LOAD

# ──────────────────────────────────────────
# Actor 定义（M=1，22 维状态）
# ──────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────
# 配置：把所有要测的数据集列在这里
# ──────────────────────────────────────────
TEST_DATASETS = [
    {'name': '3月25-27日', 'path': '1111_528/520_522.csv'},
    {'name': '4月16-18日', 'path': '1111_528/416_418.csv'},
]

ACTOR_PATH   = "ablation_results_M1/actor_M1_full.pth"
SETPOINT     = 606.0
N_SHADOW     = ShadowAugmentor.N_SHADOW_FEATS  # 4
CTRL_HORIZON = 1
STATE_DIM    = 18 + N_SHADOW + 0               # = 22


# ──────────────────────────────────────────
# 评估单个数据集
# ──────────────────────────────────────────
def evaluate_one(test_csv, dataset_name):
    # 锁种子 → 保证可重复
    np.random.seed(0); _random.seed(0); torch.manual_seed(0)

    env = BoilerGymEnvPhysics(test_csv, model_d, params,
                              shadow_predictor=shadow_model)
    env.curr_u = float(env.df.iloc[0][COL_U])

    actor = Actor(STATE_DIM, CTRL_HORIZON).to(device)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    actor.eval()

    shadow_aug = ShadowAugmentor(shadow_model, n_lags=10,
                                 horizon=rl_dyna.SHADOW_HORIZON,
                                 target=env.target_sp)

    def build_state(raw_s, env_):
        sf = shadow_aug.rollout(env_)
        return np.concatenate([raw_s, sf]).astype(np.float32)

    # 强制起点 + 跑完整数据集
    s_raw = env.reset()
    EVAL_START = env.n_lags
    env.step_idx = EVAL_START
    env.current_step_count = 0
    env.accumulated_error  = 0.0
    env.prev_actual_delta  = 0.0

    idxs = slice(EVAL_START - env.n_lags, EVAL_START)
    env.y_hist.clear();         env.y_hist.extend(env.df[rl_dyna.COL_Y].iloc[idxs].values)
    env.y_disturb_hist.clear(); env.y_disturb_hist.extend(env.df['y_disturb'].iloc[idxs].values)
    env.u_hist.clear();         env.u_hist.extend(env.df[COL_U].iloc[idxs].values)

    env.curr_y = float(env.y_hist[-1])
    env.curr_u = float(env.u_hist[-1])
    env.control_state = [
        float(env.df['x1'].iloc[EVAL_START - 1]),
        float(env.df['x2'].iloc[EVAL_START - 1]),
        float(env.df['x3'].iloc[EVAL_START - 1]),
    ]
    env.u_delay_queue.clear()
    env.u_delay_queue.extend(
        env.df[COL_U].iloc[max(0, EVAL_START - env.delay - 1):EVAL_START].values)
    env.max_steps = 99999

    s_raw = env._get_state()
    s = build_state(s_raw, env)

    n_eval_steps = len(env.df) - EVAL_START - 5

    diffs_rl = []
    y_rl_list, y_origin_list = [], []
    u_rl_list, u_origin_list = [], []
    load_list = []

    with torch.no_grad():
        for _ in range(n_eval_steps):
            a = actor(torch.FloatTensor(s).unsqueeze(0).to(device)).cpu().numpy()[0]
            planned = decode_action(env.curr_u, a)
            ns_raw, _, d, info = env.step(planned[0])
            s = build_state(ns_raw, env)
            diffs_rl.append(info['diff'])
            y_rl_list.append(info['y_rl'])
            y_origin_list.append(info['y_origin'])
            u_rl_list.append(info['u_rl'])
            row = env.df.iloc[min(env.step_idx, len(env.df) - 1)]
            u_origin_list.append(float(row[COL_U]))
            load_list.append(float(row[COL_LOAD]))
            if d: break

    diffs_rl = np.array(diffs_rl)
    diffs_h  = np.array(y_origin_list) - SETPOINT

    return {
        'name':     dataset_name,
        'path':     test_csv,
        'n_steps':  len(diffs_rl),
        'hours':    len(diffs_rl) * 60 / 3600,
        'load_min': float(np.min(load_list)),
        'load_max': float(np.max(load_list)),
        'load_mean':float(np.mean(load_list)),

        'rl_MAE':   float(np.mean(np.abs(diffs_rl))),
        'rl_RMSE':  float(np.sqrt(np.mean(diffs_rl ** 2))),
        'rl_p2':    float(np.mean(np.abs(diffs_rl) <= 2.0) * 100),
        'rl_p5':    float(np.mean(np.abs(diffs_rl) <= 5.0) * 100),

        'h_MAE':    float(np.mean(np.abs(diffs_h))),
        'h_RMSE':   float(np.sqrt(np.mean(diffs_h ** 2))),
        'h_p2':     float(np.mean(np.abs(diffs_h) <= 2.0) * 100),
        'h_p5':     float(np.mean(np.abs(diffs_h) <= 5.0) * 100),

        'y_rl':     y_rl_list,
        'y_origin': y_origin_list,
        'u_rl':     u_rl_list,
        'u_origin': u_origin_list,
        'load':     load_list,
        'diff_rl':  diffs_rl.tolist(),
    }


# ──────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────
print(f"\nM1_full 控制器多数据集泛化性测试")
print(f"权重: {ACTOR_PATH}\n")

os.makedirs('generalization_results', exist_ok=True)
results = []

for ds in TEST_DATASETS:
    if not os.path.exists(ds['path']):
        print(f"[跳过] 文件不存在: {ds['path']}")
        continue
    print(f"评估 {ds['name']}  ({ds['path']}) ...")
    res = evaluate_one(ds['path'], ds['name'])
    results.append(res)
    print(f"  评估窗口: {res['n_steps']} 步 ({res['hours']:.1f} h),  "
          f"负荷范围 {res['load_min']:.0f}–{res['load_max']:.0f} MW")
    print(f"  RL  : MAE={res['rl_MAE']:.3f}  RMSE={res['rl_RMSE']:.3f}  "
          f"±2°C={res['rl_p2']:.2f}%  ±5°C={res['rl_p5']:.2f}%")
    print(f"  手操: MAE={res['h_MAE']:.3f}  RMSE={res['h_RMSE']:.3f}  "
          f"±2°C={res['h_p2']:.2f}%  ±5°C={res['h_p5']:.2f}%")

    # 保存时序
    pd.DataFrame({
        't_step':   np.arange(res['n_steps']),
        'y_rl':     res['y_rl'],
        'y_origin': res['y_origin'],
        'u_rl':     res['u_rl'],
        'u_origin': res['u_origin'],
        'load':     res['load'],
        'diff_rl':  res['diff_rl'],
    }).to_csv(f"generalization_results/timeseries_{ds['name']}.csv",
              index=False, encoding='utf-8-sig')

# ──────────────────────────────────────────
# 汇总表 + 打印
# ──────────────────────────────────────────
print("\n" + "═" * 100)
print(f"{'数据集':<14}{'时长/h':>9}{'负荷范围/MW':>16}"
      f"{'RL MAE':>10}{'手操 MAE':>11}{'MAE改善':>11}"
      f"{'RL ±2°C':>11}{'RL ±5°C':>11}")
print("─" * 100)
for r in results:
    improve = (r['h_MAE'] - r['rl_MAE']) / r['h_MAE'] * 100
    load_range = f"{r['load_min']:.0f}–{r['load_max']:.0f}"
    print(f"{r['name']:<14}{r['hours']:>9.1f}{load_range:>16}"
          f"{r['rl_MAE']:>10.2f}{r['h_MAE']:>11.2f}{improve:>10.1f}%"
          f"{r['rl_p2']:>11.2f}{r['rl_p5']:>11.2f}")
print("═" * 100)

# 保存汇总 CSV
df_summary = pd.DataFrame([{
    'dataset':    r['name'],
    'hours':      round(r['hours'], 1),
    'load_min':   round(r['load_min'], 0),
    'load_max':   round(r['load_max'], 0),
    'rl_MAE':     round(r['rl_MAE'], 3),
    'rl_RMSE':    round(r['rl_RMSE'], 3),
    'rl_p2':      round(r['rl_p2'], 2),
    'rl_p5':      round(r['rl_p5'], 2),
    'h_MAE':      round(r['h_MAE'], 3),
    'h_RMSE':     round(r['h_RMSE'], 3),
    'h_p2':       round(r['h_p2'], 2),
    'h_p5':       round(r['h_p5'], 2),
} for r in results])
df_summary.to_csv("generalization_results/summary.csv",
                  index=False, encoding='utf-8-sig')
print(f"\n✓ 汇总: generalization_results/summary.csv")

# ──────────────────────────────────────────
# 画图：每个数据集一张控制效果对比图
# ──────────────────────────────────────────
os.makedirs('figure', exist_ok=True)
for r in results:
    n   = r['n_steps']
    t_h = np.arange(n) * 60 / 3600
    diff_rl = np.array(r['diff_rl'])
    diff_h  = np.array(r['y_origin']) - SETPOINT

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(4, 1, height_ratios=[2, 1.2, 1, 1.2], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.axhline(SETPOINT, color='black', ls='--', lw=1, label='设定值 606°C')
    ax1.fill_between(t_h, SETPOINT-2, SETPOINT+2, color='green',  alpha=0.08)
    ax1.fill_between(t_h, SETPOINT-5, SETPOINT+5, color='orange', alpha=0.04)
    ax1.plot(t_h, r['y_origin'], color='gray', alpha=0.55, lw=0.9,
             label=f"历史手操 (MAE={r['h_MAE']:.2f}°C)")
    ax1.plot(t_h, r['y_rl'], color='crimson', lw=1.0,
             label=f"RL 控制器 (MAE={r['rl_MAE']:.2f}°C)")
    ax1.set_ylabel('再热汽温 (°C)')
    ax1.set_title(f"{r['name']} 数据集 ({r['hours']:.1f}h, "
                  f"负荷 {r['load_min']:.0f}–{r['load_max']:.0f} MW)",
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t_h, r['u_origin'], color='gray', alpha=0.55, lw=0.9, label='手操开度')
    ax2.plot(t_h, r['u_rl'],     color='dodgerblue', lw=1.0,    label='RL 开度')
    ax2.set_ylabel('挡板开度 (%)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t_h, r['load'], color='darkgreen', lw=0.9)
    ax3.set_ylabel('负荷 (MW)')
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.fill_between(t_h, -2, 2, color='green',  alpha=0.10, label='±2°C')
    ax4.fill_between(t_h, -5, 5, color='orange', alpha=0.05, label='±5°C')
    ax4.plot(t_h, diff_h,  color='gray', alpha=0.55, lw=0.8, label='手操偏差')
    ax4.plot(t_h, diff_rl, color='crimson',         lw=0.9, label='RL 偏差')
    ax4.axhline(0, color='black', ls='--', lw=0.6)
    ax4.set_ylabel('偏差 (°C)')
    ax4.set_xlabel('时间 (小时)')
    ax4.legend(loc='upper right', fontsize=9, ncol=2)
    ax4.grid(alpha=0.3)

    plt.savefig(f"figure/5_6_泛化_{r['name']}.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"✓ figure/5_6_泛化_{r['name']}.png")

print("\n所有数据集评估与可视化完成。")