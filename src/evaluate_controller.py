import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"评估使用设备: {device}")

# ==========================================
# 0. 路径与配置
# ==========================================
TEST_DATA_PATH    = ROOT / "data/raw/holdout_2026-05-28_05-30.csv"
TRAIN_DATA_PATH   = ROOT / "data/raw/RHT0121quan.csv"   # 用于重算训练集归一化统计量
# ACTOR_PATH        = ROOT / "models/actors/bc_actor.pth"
ACTOR_PATH        = ROOT / "models/actors/best_actor_pureRL.pth"
MODEL_D_PATH      = ROOT / "models/env/boiler_model_d.pkl"
PARAMS_PATH       = ROOT / "models/env/boiler_physics_params.pkl"
SHADOW_MODEL_PATH = ROOT / "models/env/boiler_env_model.pkl"
ACTOR_PATH = Path(os.environ.get("ACTOR_PATH_OVERRIDE", ACTOR_PATH))
EVAL_TAG = os.environ.get("EVAL_TAG", ACTOR_PATH.stem.replace("best_actor_", ""))
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"
ACTION_MODE = os.environ.get("ACTION_MODE", "absolute").strip().lower()
RESIDUAL_MAX_DELTA = float(os.environ.get("RESIDUAL_MAX_DELTA", "10.0"))
PRIOR_OUTPUT_MODE = os.environ.get("PRIOR_OUTPUT_MODE", "absolute").strip().lower()
PRIOR_MAX_DELTA = float(os.environ.get("PRIOR_MAX_DELTA", "10.0"))
BC_PRIOR_ACTOR_PATH = Path(os.environ.get(
    "BC_PRIOR_ACTOR_PATH", ROOT / "models/actors/bc_actor.pth"))

if ACTION_MODE not in {"absolute", "residual", "bc_prior_residual"}:
    raise ValueError(
        f"Unsupported ACTION_MODE={ACTION_MODE!r}; "
        "use 'absolute', 'residual', or 'bc_prior_residual'.")
if PRIOR_OUTPUT_MODE not in {"absolute", "delta"}:
    raise ValueError(
        f"Unsupported PRIOR_OUTPUT_MODE={PRIOR_OUTPUT_MODE!r}; "
        "use 'absolute' or 'delta'.")

CTRL_HORIZON   = 1
SHADOW_HORIZON = 10

# RL 权重用 5%（与 RL 训练时一致）
if "bc_actor" in str(ACTOR_PATH):
    DEADZONE_FRAC = 0.0
    print("▶ 评估 BC 权重，死区设为 0%")
else:
    DEADZONE_FRAC = 0.05
    print("▶ 评估 RL 权重，死区设为 5%")
if "DEADZONE_FRAC" in os.environ:
    DEADZONE_FRAC = float(os.environ["DEADZONE_FRAC"])
    print(f"DEADZONE_FRAC override = {DEADZONE_FRAC:.3f}")

try:
    model_d      = joblib.load(MODEL_D_PATH)
    params       = joblib.load(PARAMS_PATH)
    shadow_model = joblib.load(SHADOW_MODEL_PATH)
except Exception as e:
    print(f"错误：加载模型或参数失败。{e}")
    exit()

COL_U, COL_Y     = params['col_u'], params['col_y']
COL_LOAD, COL_COAL = params['col_load'], params['col_coal']
print(f"ACTION_MODE = {ACTION_MODE}")
if ACTION_MODE in {"residual", "bc_prior_residual"}:
    print(f"RESIDUAL_MAX_DELTA = {RESIDUAL_MAX_DELTA:.1f}% opening per step")
if ACTION_MODE == "bc_prior_residual":
    print(f"BC_PRIOR_ACTOR_PATH = {BC_PRIOR_ACTOR_PATH}")
    print(f"PRIOR_OUTPUT_MODE = {PRIOR_OUTPUT_MODE}")
    if PRIOR_OUTPUT_MODE == "delta":
        print(f"PRIOR_MAX_DELTA = {PRIOR_MAX_DELTA:.1f}% opening per step")


# ==========================================
# T(L) 分bin 线性插值（与 RL_env_6m.py / RL_dyna_6m.py / BC.py 完全一致）
# ==========================================
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


# ==========================================
# 1. 影子前瞻与状态增强
# ==========================================
class ShadowAugmentor:
    N_SHADOW_FEATS = 4

    def __init__(self, shadow_model, n_lags=10, horizon=SHADOW_HORIZON, target=606.0):
        self.model   = shadow_model
        self.n_lags  = n_lags
        self.horizon = horizon
        self.target  = target

    def rollout(self, env) -> np.ndarray:
        if self.model is None:
            return np.zeros(self.N_SHADOW_FEATS, dtype=np.float32)
        y_buf, u_buf = list(env.y_hist), list(env.u_hist)
        curr_u, step, df = env.curr_u, env.step_idx, env.df
        max_idx = len(df) - 2
        y_std   = max(env.stats.get(COL_Y + '_std', 1.0), 1e-6)
        preds   = []
        for h in range(self.horizon):
            idx_c = min(step + h,     max_idx)
            idx_n = min(step + h + 1, max_idx)
            h_y = list(reversed(y_buf[-self.n_lags:]))
            h_u = [curr_u] + list(reversed(u_buf[-(self.n_lags - 1):]))
            while len(h_y) < self.n_lags: h_y.append(h_y[-1] if h_y else self.target)
            while len(h_u) < self.n_lags: h_u.append(curr_u)
            feat = (h_y[:self.n_lags] + h_u[:self.n_lags] +
                    [float(df.iloc[idx_c][COL_LOAD]), float(df.iloc[idx_n][COL_LOAD]),
                     float(df.iloc[idx_c][COL_COAL]), float(df.iloc[idx_n][COL_COAL])])
            y_next = float(self.model.predict(
                np.array(feat, dtype=np.float32).reshape(1, -1))[0])
            preds.append(y_next); y_buf.append(y_next); u_buf.append(curr_u)
        preds = np.array(preds, dtype=np.float32)
        errs  = preds - self.target
        feats = np.array([
            np.mean(errs) / y_std, np.max(np.abs(errs)) / y_std,
            errs[-1] / y_std, (preds[-1] - preds[0]) / max(self.horizon, 1) / y_std,
        ], dtype=np.float32)
        return np.clip(feats, -5.0, 5.0)


def augment_state(raw_state, env, augmentor):
    return np.concatenate([raw_state,
                           augmentor.rollout(env)]).astype(np.float32)


# ==========================================
# 2. 动作解码：Actor 输出 = 目标绝对开度（Tanh 域 [-1,1] → [0,100]%）
#    死区只对“相对当前开度的变化量”判断，用于滤抖动
# ==========================================
DEADZONE = DEADZONE_FRAC * 100.0
BC_PRIOR_ACTOR = None

def absolute_action_to_opening(action_value):
    return float(np.clip((float(action_value) + 1.0) / 2.0 * 100.0, 0.0, 100.0))


def bc_prior_action_for_state(state):
    if BC_PRIOR_ACTOR is None:
        raise RuntimeError("BC prior actor is required for ACTION_MODE=bc_prior_residual.")
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        a = BC_PRIOR_ACTOR(s).cpu().numpy()[0][0]
    return float(np.clip(a, -1.0, 1.0))


def bc_prior_opening_for_state(state, curr_u):
    prior_action = bc_prior_action_for_state(state)
    if PRIOR_OUTPUT_MODE == "delta":
        target_u = float(curr_u) + prior_action * PRIOR_MAX_DELTA
        return float(np.clip(target_u, 0.0, 100.0))
    return absolute_action_to_opening(prior_action)


def action_to_opening(curr_u, action_value, state=None):
    if ACTION_MODE == "bc_prior_residual":
        if state is None:
            raise RuntimeError("state is required for ACTION_MODE=bc_prior_residual.")
        target_u = (bc_prior_opening_for_state(state, curr_u) +
                    float(action_value) * RESIDUAL_MAX_DELTA)
    elif ACTION_MODE == "residual":
        target_u = curr_u + float(action_value) * RESIDUAL_MAX_DELTA
    else:
        target_u = absolute_action_to_opening(action_value)
    return float(np.clip(target_u, 0.0, 100.0))

def decode_action(curr_u, action_arr, state=None):
    planned, prev_u = [], curr_u
    for k in range(len(action_arr)):
        target_u = action_to_opening(prev_u, float(action_arr[k]), state=state)
        # 相对上一开度的变化量小于死区则保持不动（滤抖动）
        if abs(target_u - prev_u) < DEADZONE:
            next_u = prev_u
        else:
            next_u = float(np.clip(target_u, 0.0, 100.0))
        planned.append(next_u)
        prev_u = next_u
    return np.array(planned, dtype=np.float32)


# ==========================================
# 3. 网络结构
# ==========================================
class Actor(nn.Module):
    def __init__(self, state_dim, ctrl_horizon=CTRL_HORIZON):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, ctrl_horizon), nn.Tanh()
        )
    def forward(self, x): return self.net(x)


# ==========================================
# 训练集归一化统计量（与 BC.py / RL_env_6m.py 同源同算法）
# 优先用 params['norm_stats']；没有则用训练 CSV 重算
# ==========================================
def build_train_stats():
    if 'norm_stats' in params:
        print("✓ 使用 pkl 内训练集归一化统计量 norm_stats")
        return params['norm_stats']
    print(f"⚠ pkl 无 norm_stats，从训练集 {TRAIN_DATA_PATH} 重算 ...")
    dt = params.get('dt', 60)
    raw = pd.read_csv(TRAIN_DATA_PATH, encoding='gbk')
    dft = (raw[[COL_U, COL_Y, COL_LOAD, COL_COAL]]
           .groupby(raw.index // dt)
           .agg({COL_U: 'mean', COL_Y: 'last',
                 COL_LOAD: 'mean', COL_COAL: 'mean'})
           .reset_index(drop=True))
    dft['load_delta'] = dft[COL_LOAD].diff().fillna(0)
    stat_cols = [COL_U, COL_Y, COL_LOAD, COL_COAL, 'load_delta']
    stats = {c + '_mean': dft[c].mean() for c in stat_cols}
    stats.update({c + '_std': max(dft[c].std(), 1e-6) for c in stat_cols})
    return stats


# ==========================================
# 4. 评估环境（分bin T(L)，三阶惯性）—— 与训练环境一致
# ==========================================
class BoilerEvalEnv:
    STATE_DIM = 18

    def __init__(self, data_path, md, phys_params, train_stats):
        self.model_d = md

        self.K, self.u_ss = phys_params['K'], phys_params['u_ss']
        self.dt    = phys_params.get('dt', 60)

        self.feat_d, self.n_lags = phys_params['features_d'], phys_params['N_LAGS']
        self.delay = int(phys_params.get('delay', 0))

        raw = pd.read_csv(data_path, encoding='gbk')
        self.df = (raw[[COL_U, COL_Y, COL_LOAD, COL_COAL]]
                   .groupby(raw.index // self.dt)
                   .agg({COL_U: 'mean', COL_Y: 'last',
                         COL_LOAD: 'mean', COL_COAL: 'mean'})
                   .reset_index(drop=True))

        u_arr    = self.df[COL_U].values
        load_arr = self.df[COL_LOAD].values

        x1 = np.zeros(len(self.df))
        x2 = np.zeros(len(self.df))
        x3 = np.zeros(len(self.df))

        for t in range(1, len(self.df)):
            T_t = T_interp(load_arr[t])          # 分bin 插值
            a = np.exp(-self.dt / T_t)
            b = 1.0 - a
            x1[t] = a * x1[t-1] + b * self.K * (u_arr[t-1] - self.u_ss)
            x2[t] = a * x2[t-1] + b * x1[t]
            x3[t] = a * x3[t-1] + b * x2[t]

        self.df['x1'] = x1
        self.df['x2'] = x2
        self.df['x3'] = x3
        self.df['x_ctrl']    = x3
        self.df['y_disturb'] = self.df[COL_Y] - x3

        self.df['load_delta']        = self.df[COL_LOAD].diff().fillna(0)
        self.df['load_delta_mean_5'] = (self.df['load_delta']
                                        .rolling(5, min_periods=1).mean().fillna(0))

        # 归一化统计量：使用训练集统计量（与 BC 训练一致），不再用测试集自身
        self.stats = train_stats

        self.target_sp = 606.0
        self.max_steps = len(self.df) - self.n_lags - 5
        self.x_scale   = max(self.K * 50.0, 1e-4)

        self.y_hist         = deque(maxlen=self.n_lags)
        self.y_disturb_hist = deque(maxlen=self.n_lags)
        self.u_hist         = deque(maxlen=self.n_lags)
        self.u_delay_queue  = deque(maxlen=self.delay + 1)
        self.prev_actual_delta = 0.0

    def normalize(self, val, col_name, is_bias=False):
        std = self.stats[col_name + '_std']
        return val / std if is_bias else (val - self.stats[col_name + '_mean']) / std

    def reset(self):
        self.current_step_count = 0
        self.accumulated_error  = 0.0
        self.prev_actual_delta  = 0.0
        self.step_idx = self.n_lags
        idxs = slice(self.step_idx - self.n_lags, self.step_idx)

        self.y_hist.clear();         self.y_hist.extend(self.df[COL_Y].iloc[idxs].values)
        self.y_disturb_hist.clear(); self.y_disturb_hist.extend(self.df['y_disturb'].iloc[idxs].values)
        self.u_hist.clear();         self.u_hist.extend(self.df[COL_U].iloc[idxs].values)

        self.curr_y = float(self.y_hist[-1])
        self.curr_u = float(self.u_hist[-1])
        self.control_state = [
            float(self.df['x1'].iloc[self.step_idx - 1]),
            float(self.df['x2'].iloc[self.step_idx - 1]),
            float(self.df['x3'].iloc[self.step_idx - 1])
        ]

        self.u_delay_queue.clear()
        self.u_delay_queue.extend(
            self.df[COL_U].iloc[max(0, self.step_idx - self.delay - 1): self.step_idx].values)
        return self._get_state()

    def _get_state(self):
        row      = self.df.iloc[self.step_idx]
        next_row = self.df.iloc[self.step_idx + 1]
        error    = self.normalize(self.target_sp - self.curr_y, COL_Y, is_bias=True)
        y_trend  = [self.normalize(x, COL_Y) for x in list(self.y_hist)[-5:]]
        u_trend  = [self.normalize(x, COL_U) for x in list(self.u_hist)[-5:]]
        load_delta_val = self.normalize(next_row.get('load_delta', 0.0), 'load_delta')
        return np.array(
            [error, np.clip(self.accumulated_error / 50, -2, 2)] +
            y_trend + u_trend +
            [self.normalize(row[COL_LOAD],      COL_LOAD),
             self.normalize(next_row[COL_LOAD],  COL_LOAD),
             self.normalize(row[COL_COAL],       COL_COAL),
             self.normalize(next_row[COL_COAL],  COL_COAL),
             load_delta_val,
             self.control_state[2] / self.x_scale],
            dtype=np.float32)

    def _build_input_d(self):
        row      = self.df.iloc[self.step_idx]
        next_row = self.df.iloc[self.step_idx + 1]
        yd  = list(self.y_disturb_hist)
        dyd = np.array([(yd[-i] - yd[-i - 1]) if len(yd) > i else 0.0
                        for i in range(1, self.n_lags)])
        feat = {
            'y_disturb_curr':     float(yd[-1]) if len(yd) > 0 else 0.0,
            **{f'dy_disturb_{i}': dyd[i - 1] for i in range(5, self.n_lags)},
            'dy_disturb_mean_5':  float(np.mean(dyd[:5]))         if len(dyd) >= 5 else 0.0,
            'dy_disturb_mean_10': float(np.mean(dyd)),
            'dy_disturb_std_5':   float(np.std(dyd[:5], ddof=1)) if len(dyd) >= 5 else 0.0,
            'load_curr':          float(row[COL_LOAD]),
            'load_next':          float(next_row[COL_LOAD]),
            'load_delta':         float(row.get('load_delta', 0.0)),
            'load_delta_mean_5':  float(row.get('load_delta_mean_5', 0.0)),
            **{f'load_lag_{i}': float(self.df.iloc[max(0, self.step_idx - i)][COL_LOAD])
               for i in range(1, self.n_lags)},
            'coal_curr': float(row[COL_COAL]),
            'coal_next': float(next_row[COL_COAL]),
        }
        return np.array([feat.get(f, 0.0) for f in self.feat_d],
                        dtype=np.float32).reshape(1, -1)

    def step(self, next_u_value: float):
        self.current_step_count += 1
        actual_delta = next_u_value - self.curr_u
        next_u       = float(np.clip(next_u_value, 0.0, 100.0))

        self.u_delay_queue.append(next_u)
        u_delayed = (self.u_delay_queue[0]
                     if len(self.u_delay_queue) > self.delay else self.u_ss)

        # 分bin T(L) + 三阶递推（与训练一致）
        current_load = float(self.df.iloc[self.step_idx][COL_LOAD])
        T_t = T_interp(current_load)
        a = np.exp(-self.dt / T_t)
        b = 1.0 - a

        x1_prev, x2_prev, x3_prev = self.control_state
        next_x1 = a * x1_prev + b * self.K * (u_delayed - self.u_ss)
        next_x2 = a * x2_prev + b * next_x1
        next_x3 = a * x3_prev + b * next_x2
        new_ctrl_state = [next_x1, next_x2, next_x3]

        y_dist_next = float(self.model_d.predict(self._build_input_d())[0])
        pred_y      = y_dist_next + next_x3

        diff = self.target_sp - pred_y
        self.accumulated_error = np.clip(self.accumulated_error + diff, -100.0, 100.0)
        self.prev_actual_delta = actual_delta
        self.control_state     = new_ctrl_state
        self.y_hist.append(pred_y)
        self.y_disturb_hist.append(y_dist_next)
        self.u_hist.append(next_u)
        self.curr_y, self.curr_u = pred_y, next_u
        self.step_idx += 1

        done = ((self.current_step_count >= self.max_steps) or
                (self.step_idx >= len(self.df) - 5))
        info = {
            'y_rl': pred_y, 'u_rl': next_u, 'diff': diff,
            'load': float(self.df.iloc[self.step_idx][COL_LOAD]),
            'coal': float(self.df.iloc[self.step_idx][COL_COAL]),
            'y_origin': float(self.df.iloc[self.step_idx][COL_Y]),
            'u_origin': float(self.df.iloc[self.step_idx][COL_U]),
        }
        return self._get_state(), 0, done, info


# ==========================================
# 5. 执行评估
# ==========================================
if __name__ == "__main__":
    train_stats = build_train_stats()
    env       = BoilerEvalEnv(TEST_DATA_PATH, model_d, params, train_stats)
    augmentor = ShadowAugmentor(shadow_model, n_lags=10,
                                horizon=SHADOW_HORIZON, target=env.target_sp)

    AUG_STATE_DIM = env.STATE_DIM + ShadowAugmentor.N_SHADOW_FEATS

    actor = Actor(AUG_STATE_DIM, ctrl_horizon=CTRL_HORIZON).to(device)
    try:
        actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
        print(f"✓ 成功加载模型权重: {ACTOR_PATH}")
    except Exception as e:
        print(f"✗ 无法加载模型权重: {e}")
        exit()

    actor.eval()
    if ACTION_MODE == "bc_prior_residual":
        try:
            BC_PRIOR_ACTOR = Actor(AUG_STATE_DIM, ctrl_horizon=CTRL_HORIZON).to(device)
            bc_prior_state = torch.load(BC_PRIOR_ACTOR_PATH, map_location=device)
            BC_PRIOR_ACTOR.load_state_dict(bc_prior_state)
            BC_PRIOR_ACTOR.eval()
            print(f"[BC prior residual] loaded prior actor: {BC_PRIOR_ACTOR_PATH}")
        except Exception as e:
            print(f"无法加载 BC prior actor: {e}")
            exit()
    s_raw = env.reset()
    s     = augment_state(s_raw, env, augmentor)

    history = {k: [] for k in ['y_rl', 'u_rl', 'diff', 'y_origin', 'u_origin', 'load', 'coal']}

    print(f"\n开始在数据集 [{TEST_DATA_PATH}] 上进行测试...")
    print(f"预计测试步数: {env.max_steps} 步 (约 {env.max_steps * env.dt / 3600:.1f} 小时)")

    with torch.no_grad():
        for i in range(env.max_steps):
            s_tensor  = torch.FloatTensor(s).unsqueeze(0).to(device)
            a_raw     = actor(s_tensor).cpu().numpy()[0]
            planned_u = decode_action(env.curr_u, a_raw, state=s)

            ns_raw, _, d, info = env.step(planned_u[0])
            s = augment_state(ns_raw, env, augmentor)

            for k in history: history[k].append(info[k])
            if i % 1000 == 0 and i > 0:
                print(f"  已运行 {i} 步...")
            if d: break

    diffs = np.array(history['diff'])
    mae   = np.mean(np.abs(diffs))
    rmse  = np.sqrt(np.mean(diffs ** 2))
    in_2c = np.mean(np.abs(diffs) <= 2.0) * 100
    in_5c = np.mean(np.abs(diffs) <= 5.0) * 100

    print(f"\n{'='*40}")
    print(f" 测试完成！数据集: {TEST_DATA_PATH}")
    print(f"{'='*40}")
    print(f" MAE:     {mae:.2f} °C")
    print(f" RMSE:    {rmse:.2f} °C")
    print(f" ±2°C:    {in_2c:.1f} %")
    print(f" ±5°C:    {in_5c:.1f} %")
    print(f"{'='*40}\n")

    os.makedirs('figure', exist_ok=True)
    n    = len(history['y_rl'])
    t_ax = np.arange(n) * env.dt / 3600

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.axhline(606, color='green', ls='--', lw=1.5, label='目标 606°C')
    ax1.plot(t_ax, history['y_origin'], 'gray', alpha=0.3, lw=1, label='历史手操')
    ax1.plot(t_ax, history['y_rl'], 'crimson', lw=1.5,
             label=f'RL 控制 (MAE={mae:.2f}°C)')
    ax1.set_ylabel("再热汽温 (°C)"); ax1.legend(loc='upper right'); ax1.grid(alpha=0.3)
    ax1.set_title(f"评估测试 (总耗时: {n*env.dt/3600:.1f}h)", fontsize=14, fontweight='bold')

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t_ax, history['u_origin'], 'gray', alpha=0.3, lw=1, label='手操开度')
    ax2.plot(t_ax, history['u_rl'], 'dodgerblue', lw=1.5, label='RL 开度')
    ax2.set_ylabel("挡板开度 (%)"); ax2.legend(loc='upper right'); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t_ax, history['load'], 'darkgreen', lw=1, label='负荷 (MW)')
    ax3.set_ylabel("负荷 (MW)"); ax3.legend(); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.fill_between(t_ax, -2, 2, color='green', alpha=0.10, label='±2°C')
    ax4.fill_between(t_ax, -5, 5, color='orange', alpha=0.05, label='±5°C')
    ax4.plot(t_ax, history['diff'], 'crimson', alpha=0.8, lw=1, label='偏差')
    ax4.axhline(0, color='black', ls='--', lw=1)
    ax4.set_ylabel("偏差 (°C)"); ax4.set_xlabel("时间 (小时)")
    ax4.legend(loc='upper right'); ax4.grid(alpha=0.3)

    report_path = ROOT / "results/figures" / f"evaluation_report_{EVAL_TAG}.png"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved figure: {report_path}")

    # ════════════════════════════════════════════════════════
    # 追加：手操 baseline 指标 + 时序导出
    # ════════════════════════════════════════════════════════
    SETPOINT = 606.0

    y_rl_arr = np.array(history['y_rl'])
    y_origin_arr = np.array(history['y_origin'])
    diffs_rl = y_rl_arr - SETPOINT
    diffs_h = y_origin_arr - SETPOINT

    def _stats(d):
        return (np.mean(np.abs(d)),
                np.sqrt(np.mean(d ** 2)),
                np.mean(np.abs(d) <= 2.0) * 100,
                np.mean(np.abs(d) <= 5.0) * 100)

    mae_r, rmse_r, p2_r, p5_r = _stats(diffs_rl)
    mae_h, rmse_h, p2_h, p5_h = _stats(diffs_h)

    print("\n" + "═" * 56)
    print(f"{'方法':<10}{'MAE/°C':>10}{'RMSE/°C':>11}{'±2°C/%':>11}{'±5°C/%':>11}")
    print("─" * 56)
    print(f"{'历史手操':<10}{mae_h:>10.3f}{rmse_h:>11.3f}{p2_h:>11.2f}{p5_h:>11.2f}")
    print(f"{'RL控制器':<10}{mae_r:>10.3f}{rmse_r:>11.3f}{p2_r:>11.2f}{p5_r:>11.2f}")
    print("═" * 56)
    print(f"MAE 改善：{(mae_h - mae_r) / mae_h * 100:+.1f}%   "
          f"±2°C 提升：{p2_r - p2_h:+.2f} pp   "
          f"±5°C 提升：{p5_r - p5_h:+.2f} pp\n")

    (ROOT / "results/evaluation").mkdir(parents=True, exist_ok=True)
    timeseries_path = ROOT / "results/evaluation" / f"full_controller_timeseries_{EVAL_TAG}.csv"
    pd.DataFrame({
        't_step': np.arange(len(y_rl_arr)),
        'y_rl': y_rl_arr,
        'y_origin': y_origin_arr,
        'u_rl': history['u_rl'],
        'u_origin': history['u_origin'],
        'load': history['load'],
        'diff_rl': diffs_rl,
        'diff_origin': diffs_h,
    }).to_csv(timeseries_path,
              index=False, encoding='utf-8-sig')
    print(f"Saved timeseries: {timeseries_path}")
