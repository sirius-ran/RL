"""
RL_bc_pretrain.py
================================================================
行为克隆 (Behavior Cloning) 预训练:
  用电厂7天历史手操数据训练 Actor 网络,作为 DDPG 训练的起点。

依赖:
  boiler_model_d.pkl, boiler_physics_params.pkl, boiler_env_model.pkl
  RHT0121quan.csv (训练数据)

产出:
  bc_actor.pth — 预训练好的 Actor 权重,可直接 load 到 RL_dyna_6m.py 中

使用方式:
  1. 先跑 RL_pre.py, RL_env_6m.py 把三个 pkl 准备好
  2. python RL_bc_pretrain.py
  3. 在 RL_dyna_6m.py 中加载 bc_actor.pth (见 README 的 patch 说明)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joblib
from collections import deque
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ════════════════════════════════════════════
# 超参数 (必须与 RL_dyna_6m.py 中完全一致)
# ════════════════════════════════════════════
CTRL_HORIZON   = 1
SHADOW_HORIZON = 10
DEADZONE_FRAC  = 0.05
DEADZONE       = DEADZONE_FRAC * 100.0
MAX_DELTA      = 20.0

# BC 训练超参数
BC_EPOCHS      = 80
BC_BATCH_SIZE  = 256
BC_LR          = 1e-4
VAL_SPLIT      = 0.15
QUALITY_MAE_THRESHOLD = 50.0    # 放宽：保留手操工大幅调整的段（原 5.0 丢了 56% 含动作样本）
QUALITY_WINDOW = 3             # 局部窗口半宽(步,1步=60s)

# ════════════════════════════════════════════
# 加载模型与参数
# ════════════════════════════════════════════
model_d      = joblib.load(ROOT / "models/env/boiler_model_d.pkl")
params       = joblib.load(ROOT / "models/env/boiler_physics_params.pkl")
shadow_model = joblib.load(ROOT / "models/env/boiler_env_model.pkl")

K, u_ss      = params['K'], params['u_ss']
features_d   = params['features_d']
N_LAGS       = params['N_LAGS']
COL_U, COL_Y = params['col_u'], params['col_y']
COL_LOAD, COL_COAL = params['col_load'], params['col_coal']

T_bin_centers = params['T_bin_centers']
T_bin_values  = params['T_bin_values']
T_min = params.get('T_min', 30.0)
T_max = params.get('T_max', 1200.0)

def T_interp(load):
    if isinstance(load, np.ndarray):
        T = np.interp(load, T_bin_centers, T_bin_values,
                      left=T_bin_values[0], right=T_bin_values[-1])
        return np.clip(T, T_min, T_max)
    else:
        T = np.interp(load, T_bin_centers, T_bin_values,
                      left=T_bin_values[0], right=T_bin_values[-1])
        return max(T_min, min(T_max, T))


# ════════════════════════════════════════════
# Actor 网络(必须与 RL_dyna_6m.py 中完全一致)
# ════════════════════════════════════════════
class Actor(nn.Module):
    def __init__(self, state_dim, ctrl_horizon=CTRL_HORIZON):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, ctrl_horizon), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════
# 数据加载与降采样(与 RL_env_6m.py 一致)
# ════════════════════════════════════════════
print("\n[1/5] 加载数据 ...")
DT = 60
raw = pd.read_csv(ROOT / "data/raw/RHT0121quan.csv", encoding="gbk")
df  = (raw[[COL_U, COL_Y, COL_LOAD, COL_COAL]]
       .groupby(raw.index // DT)
       .agg({COL_U: 'mean', COL_Y: 'last',
             COL_LOAD: 'mean', COL_COAL: 'mean'})
       .reset_index(drop=True))

# 预计算 x1/x2/x3 与 y_disturb(与环境一致)
u_arr    = df[COL_U].values
load_arr = df[COL_LOAD].values
x1 = np.zeros(len(df)); x2 = np.zeros(len(df)); x3 = np.zeros(len(df))
for t in range(1, len(df)):
    T_t = T_interp(load_arr[t])
    a = np.exp(-DT / T_t); b = 1.0 - a
    x1[t] = a * x1[t-1] + b * K * (u_arr[t-1] - u_ss)
    x2[t] = a * x2[t-1] + b * x1[t]
    x3[t] = a * x3[t-1] + b * x2[t]
df['x1'] = x1; df['x2'] = x2; df['x3'] = x3
df['x_ctrl']    = x3
df['y_disturb'] = df[COL_Y] - x3
df['load_delta']        = df[COL_LOAD].diff().fillna(0)
df['load_delta_mean_5'] = df['load_delta'].rolling(5, min_periods=1).mean().fillna(0)

# 统计量(用于归一化)
stat_cols = [COL_U, COL_Y, COL_LOAD, COL_COAL, 'load_delta']
stats = {c + '_mean': df[c].mean() for c in stat_cols}
stats.update({c + '_std': max(df[c].std(), 1e-6) for c in stat_cols})

print(f"  降采样后长度: {len(df):,} 步")


# ════════════════════════════════════════════
# 特征提取:把 env._get_state + shadow_aug.rollout
#         独立实现一遍,避免反复 reset env
# ════════════════════════════════════════════
def normalize(val, col_name, is_bias=False):
    std = stats[col_name + '_std']
    return val / std if is_bias else (val - stats[col_name + '_mean']) / std

def build_base_state(t, target_sp=606.0, accumulated_error=0.0):
    """构造 18 维基础状态(等价于 env._get_state())"""
    row      = df.iloc[t]
    next_row = df.iloc[t + 1]
    curr_y   = float(df[COL_Y].iloc[t])
    error    = normalize(target_sp - curr_y, COL_Y, is_bias=True)

    y_hist = df[COL_Y].iloc[max(0, t-N_LAGS+1):t+1].values
    u_hist = df[COL_U].iloc[max(0, t-N_LAGS+1):t+1].values
    y_trend = [normalize(x, COL_Y) for x in y_hist[-5:]]
    u_trend = [normalize(x, COL_U) for x in u_hist[-5:]]
    while len(y_trend) < 5: y_trend.insert(0, y_trend[0] if y_trend else 0.0)
    while len(u_trend) < 5: u_trend.insert(0, u_trend[0] if u_trend else 0.0)

    load_delta_val = normalize(next_row.get('load_delta', 0.0), 'load_delta')
    x_scale = max(K * 50.0, 1e-4)

    return np.array(
        [error, np.clip(accumulated_error / 50, -2, 2)] +
        y_trend + u_trend +
        [normalize(row[COL_LOAD],     COL_LOAD),
         normalize(next_row[COL_LOAD], COL_LOAD),
         normalize(row[COL_COAL],      COL_COAL),
         normalize(next_row[COL_COAL], COL_COAL),
         load_delta_val,
         df['x3'].iloc[t] / x_scale],
        dtype=np.float32)

def build_shadow_features(t, target_sp=606.0):
    """构造 4 维影子前瞻特征(等价于 ShadowAugmentor.rollout())"""
    n_lags  = 10
    horizon = SHADOW_HORIZON
    y_std   = max(stats[COL_Y + '_std'], 1e-6)
    max_idx = len(df) - 2

    y_buf = list(df[COL_Y].iloc[max(0, t-n_lags+1):t+1].values)
    u_buf = list(df[COL_U].iloc[max(0, t-n_lags+1):t+1].values)
    curr_u = float(df[COL_U].iloc[t])

    preds = []
    for h in range(horizon):
        idx_c = min(t + h,     max_idx)
        idx_n = min(t + h + 1, max_idx)
        h_y = list(reversed(y_buf[-n_lags:]))
        h_u = [curr_u] + list(reversed(u_buf[-(n_lags - 1):]))
        while len(h_y) < n_lags: h_y.append(h_y[-1] if h_y else target_sp)
        while len(h_u) < n_lags: h_u.append(curr_u)
        feat = (h_y[:n_lags] + h_u[:n_lags] +
                [float(df.iloc[idx_c][COL_LOAD]), float(df.iloc[idx_n][COL_LOAD]),
                 float(df.iloc[idx_c][COL_COAL]), float(df.iloc[idx_n][COL_COAL])])
        y_next = float(shadow_model.predict(
            np.array(feat, dtype=np.float32).reshape(1, -1))[0])
        preds.append(y_next); y_buf.append(y_next); u_buf.append(curr_u)

    preds = np.array(preds, dtype=np.float32)
    errs  = preds - target_sp
    feats = np.array([
        np.mean(errs) / y_std,
        np.max(np.abs(errs)) / y_std,
        errs[-1] / y_std,
        (preds[-1] - preds[0]) / max(horizon, 1) / y_std,
    ], dtype=np.float32)
    return np.clip(feats, -5.0, 5.0)


# ════════════════════════════════════════════
# 提取 (state, action) 专家对
# ════════════════════════════════════════════
print("\n[2/5] 提取专家轨迹 ...")
states_list, actions_list = [], []
skipped_quality = 0
target_sp = 606.0

# t 需要满足:有 N_LAGS 步历史,有 CTRL_HORIZON+1 步未来,有 SHADOW_HORIZON 步前瞻
t_start = N_LAGS
t_end   = len(df) - max(CTRL_HORIZON + 1, SHADOW_HORIZON + 1) - 1

for t in range(t_start, t_end):
    # 质量筛选:局部跟踪窗口 MAE
    window = df[COL_Y].iloc[max(0, t-QUALITY_WINDOW):t+QUALITY_WINDOW+1].values
    local_mae = np.mean(np.abs(window - target_sp))
    if local_mae > QUALITY_MAE_THRESHOLD:
        skipped_quality += 1
        continue

    # state 部分（单步：base 18 + shadow 4 = 22 维，无 committed）
    base   = build_base_state(t, target_sp=target_sp)        # 18 维
    shadow = build_shadow_features(t, target_sp=target_sp)   # 4 维
    s = np.concatenate([base, shadow]).astype(np.float32)

    # action 部分：预测【绝对开度】u[t+1]，归一化 [0,100]→[-1,1]
    # （绝对开度信号稳定、不稀疏、闭环不积分漂移；优于学 Δu）
    a_target = [
        float(np.clip(df[COL_U].iloc[t + k + 1] / 100.0 * 2.0 - 1.0, -1.0, 1.0))
        for k in range(CTRL_HORIZON)
    ]
    states_list.append(s)
    actions_list.append(np.array(a_target, dtype=np.float32))

states  = np.array(states_list,  dtype=np.float32)
actions = np.array(actions_list, dtype=np.float32)
print(f"  保留样本: {len(states):,} 条")
print(f"  质量筛选丢弃: {skipped_quality:,} 条 "
      f"({skipped_quality/(t_end-t_start)*100:.1f}%)")
print(f"  state.shape = {states.shape}, action.shape = {actions.shape}")


# ════════════════════════════════════════════
# 训练 / 验证集划分
# ════════════════════════════════════════════
print("\n[3/5] 切分训练/验证集 ...")
N = len(states)
idx = np.random.permutation(N)
val_n = int(N * VAL_SPLIT)
val_idx, train_idx = idx[:val_n], idx[val_n:]

S_train = torch.FloatTensor(states[train_idx]).to(device)
A_train = torch.FloatTensor(actions[train_idx]).to(device)
S_val   = torch.FloatTensor(states[val_idx]).to(device)
A_val   = torch.FloatTensor(actions[val_idx]).to(device)
print(f"  训练集: {len(S_train):,}  |  验证集: {len(S_val):,}")


# ════════════════════════════════════════════
# 监督训练 Actor
# ════════════════════════════════════════════
print("\n[4/5] 行为克隆训练 ...")
STATE_DIM = states.shape[1]
actor = Actor(STATE_DIM, CTRL_HORIZON).to(device)
opt   = optim.Adam(actor.parameters(), lr=BC_LR)

best_val_loss = float('inf')
train_hist, val_hist = [], []

for ep in range(BC_EPOCHS):
    actor.train()
    perm = torch.randperm(len(S_train))
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(S_train), BC_BATCH_SIZE):
        b = perm[i:i+BC_BATCH_SIZE]
        s_b, a_b = S_train[b], A_train[b]
        pred = actor(s_b)
        loss = F.mse_loss(pred, a_b)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n_batches += 1
    train_loss = total_loss / max(n_batches, 1)

    actor.eval()
    with torch.no_grad():
        val_pred = actor(S_val)
        val_loss = F.mse_loss(val_pred, A_val).item()
        # 额外指标:Actor 在验证集上的平均动作幅度,与专家平均动作幅度对比
        mean_pred_amp   = val_pred.abs().mean().item()
        mean_expert_amp = A_val.abs().mean().item()

    train_hist.append(train_loss); val_hist.append(val_loss)
    flag = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(actor.state_dict(), ROOT / "models/actors/bc_actor.pth")
        flag = " ★保存"
    print(f"Ep {ep+1:3d}/{BC_EPOCHS} | train={train_loss:.5f} | val={val_loss:.5f} | "
          f"|a|_pred={mean_pred_amp:.3f} |a|_expert={mean_expert_amp:.3f}{flag}")


# ════════════════════════════════════════════
# 可视化
# ════════════════════════════════════════════
print("\n[5/5] 保存训练曲线 ...")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(train_hist, label='train', linewidth=1.5)
ax[0].plot(val_hist,   label='val',   linewidth=1.5)
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('MSE Loss')
ax[0].set_title('BC 训练曲线'); ax[0].legend(); ax[0].grid(alpha=0.3)

# 动作分布对比
actor.eval()
with torch.no_grad():
    pred_all = actor(S_val).cpu().numpy().flatten()
expert_all = A_val.cpu().numpy().flatten()
ax[1].hist(expert_all, bins=50, alpha=0.5, label='专家动作', density=True)
ax[1].hist(pred_all,   bins=50, alpha=0.5, label='BC预测',  density=True)
ax[1].set_xlabel('归一化动作'); ax[1].set_ylabel('密度')
ax[1].set_title('动作分布对比(验证集)'); ax[1].legend(); ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ROOT / "results/training/bc_training_curve.png", dpi=150, bbox_inches='tight')
print("\n✓ bc_actor.pth")
print("✓ bc_training_curve.png")
print(f"\n最佳验证 loss: {best_val_loss:.5f}")
print("=== BC 预训练完成 ===")