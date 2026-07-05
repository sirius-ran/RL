"""
RL_dyna_mpc_multistep.py  (分bin T(L) 线性插值版本)
================================================================
在 RL_dyna_5m.py 基础上，将 T(L) 替换为 1111_528 闭环数据分负荷区间 median
线性插值。

【核心改动 vs RL_dyna_5m.py】

  T(L) = 负荷区间 median T 线性插值 (25MW bins, 61段有效拟合)
  替代原有的两参考点(400MW/580MW)分段线性插值

  其余代码与 RL_dyna_5m.py 完全一致。
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joblib
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ══════════════════════════════════════════
# 超参数
# ══════════════════════════════════════════
CTRL_HORIZON = 1      # 控制时域 M：Actor 每步输出 1 步开度（单步实测最优）
MPC_HORIZON  = 10     # 预测时域 H：MPC 奖励评估未来 H 步
MPC_LAMBDA_Y = 0.3
MPC_LAMBDA_U = 0.05
SHADOW_HORIZON = 10
DEADZONE_FRAC  = env_float("DEADZONE_FRAC", 0.05)

# DDPG + BC actor regularization.  This keeps BC active during RL updates
# instead of using it only as an optional warm start.
USE_BC_REG = env_bool("USE_BC_REG", True)
BC_REG_Q_FILTER = env_bool("BC_REG_Q_FILTER", True)
BC_Q_FILTER_MARGIN = env_float("BC_Q_FILTER_MARGIN", 0.0)
BC_REG_ADV_WEIGHT = env_bool("BC_REG_ADV_WEIGHT", False)
BC_ADV_START_EPISODE = env_int("BC_ADV_START_EPISODE", 20)
BC_ADV_ETA = env_float("BC_ADV_ETA", 1.0)
BC_ADV_WEIGHT_MIN = env_float("BC_ADV_WEIGHT_MIN", 0.25)
BC_ADV_WEIGHT_MAX = env_float("BC_ADV_WEIGHT_MAX", 3.0)
BC_ADV_USE_TARGET = env_bool("BC_ADV_USE_TARGET", True)
BC_REG_LAMBDA_START = env_float("BC_REG_LAMBDA_START", 1.0)
BC_REG_LAMBDA_END = env_float("BC_REG_LAMBDA_END", 0.05)
BC_REG_DECAY_EPISODES = env_int("BC_REG_DECAY_EPISODES", 120)
BC_REG_WARM_EPISODES = env_int("BC_REG_WARM_EPISODES", 0)
BC_DEMO_BATCH_SIZE = env_int("BC_DEMO_BATCH_SIZE", 256)
BC_QUALITY_MAE_THRESHOLD = env_float("BC_QUALITY_MAE_THRESHOLD", 50.0)
BC_QUALITY_WINDOW = env_int("BC_QUALITY_WINDOW", 3)
BC_DEMO_FILTER_MODE = os.environ.get("BC_DEMO_FILTER_MODE", "all").strip().lower()
BC_FILTER_U_MIN = env_float("BC_FILTER_U_MIN", 5.0)
BC_FILTER_U_MAX = env_float("BC_FILTER_U_MAX", 95.0)
BC_FILTER_MAX_DELTA_U = env_float("BC_FILTER_MAX_DELTA_U", 12.0)
BC_FILTER_MAX_LOAD_DELTA = env_float("BC_FILTER_MAX_LOAD_DELTA", 25.0)
BC_FILTER_IMPROVE_HORIZON = env_int("BC_FILTER_IMPROVE_HORIZON", 5)
BC_FILTER_IMPROVE_TOL = env_float("BC_FILTER_IMPROVE_TOL", 0.0)
BC_FILTER_IMPROVE_MIN_ERROR = env_float("BC_FILTER_IMPROVE_MIN_ERROR", 2.0)
BC_FILTER_CHECK_DIRECTION = env_bool("BC_FILTER_CHECK_DIRECTION", True)
BC_FILTER_DEADZONE_U = env_float("BC_FILTER_DEADZONE_U", DEADZONE_FRAC * 100.0)
BC_SOFT_WEIGHT_MIN = env_float("BC_SOFT_WEIGHT_MIN", 0.2)
BC_SOFT_WEIGHT_SATURATION = env_float("BC_SOFT_WEIGHT_SATURATION", 0.7)
BC_SOFT_WEIGHT_ACTION_JUMP = env_float("BC_SOFT_WEIGHT_ACTION_JUMP", 0.6)
BC_SOFT_WEIGHT_LOAD_JUMP = env_float("BC_SOFT_WEIGHT_LOAD_JUMP", 0.8)
BC_SOFT_WEIGHT_WRONG_DIRECTION = env_float("BC_SOFT_WEIGHT_WRONG_DIRECTION", 0.35)
BC_SOFT_WEIGHT_NO_IMPROVEMENT = env_float("BC_SOFT_WEIGHT_NO_IMPROVEMENT", 0.7)
BC_SOFT_SNAP_DEADZONE = env_bool("BC_SOFT_SNAP_DEADZONE", False)
BC_REG_MODE = os.environ.get("BC_REG_MODE", "action").strip().lower()
BC_DELTA_SCALE = env_float("BC_DELTA_SCALE", 10.0)
BC_DELTA_DEADZONE = env_float("BC_DELTA_DEADZONE", DEADZONE_FRAC * 100.0)
BC_DELTA_MAG_WEIGHT = env_float("BC_DELTA_MAG_WEIGHT", 1.0)
BC_DELTA_DIR_WEIGHT = env_float("BC_DELTA_DIR_WEIGHT", 0.25)
BC_DELTA_STAY_WEIGHT = env_float("BC_DELTA_STAY_WEIGHT", 0.5)
ACTION_MODE = os.environ.get("ACTION_MODE", "absolute").strip().lower()
RESIDUAL_MAX_DELTA = env_float("RESIDUAL_MAX_DELTA", 10.0)
PRIOR_OUTPUT_MODE = os.environ.get("PRIOR_OUTPUT_MODE", "absolute").strip().lower()
PRIOR_MAX_DELTA = env_float("PRIOR_MAX_DELTA", 10.0)
BC_PRIOR_ACTOR_PATH = Path(os.environ.get(
    "BC_PRIOR_ACTOR_PATH", ROOT / "models/actors/bc_actor.pth"))
RUN_TAG = os.environ.get("RUN_TAG", "").strip()
TRAIN_DATA_PATH = Path(os.environ.get("TRAIN_DATA_PATH", ROOT / "data/raw/RHT0121quan.csv"))
_VAL_DATA_PATH = os.environ.get("VAL_DATA_PATH", "").strip()
VAL_DATA_PATH = Path(_VAL_DATA_PATH) if _VAL_DATA_PATH else None
USE_VAL_SELECTION = env_bool("USE_VAL_SELECTION", False)
VAL_EVERY_EPISODES = env_int("VAL_EVERY_EPISODES", 10)
VAL_MAX_STEPS = env_int("VAL_MAX_STEPS", 500)
VAL_NUM_STARTS = env_int("VAL_NUM_STARTS", 3)
VAL_SELECTION_OBJECTIVE = os.environ.get("VAL_SELECTION_OBJECTIVE", "mae").strip().lower()
VAL_ACTION_DEADZONE = env_float("VAL_ACTION_DEADZONE", 0.5)
VAL_SAT_LOW = env_float("VAL_SAT_LOW", 5.0)
VAL_SAT_HIGH = env_float("VAL_SAT_HIGH", 95.0)
VAL_SCORE_DELTA_WEIGHT = env_float("VAL_SCORE_DELTA_WEIGHT", 0.01)
VAL_SCORE_SAT_WEIGHT = env_float("VAL_SCORE_SAT_WEIGHT", 0.01)
VAL_SCORE_FLIP_WEIGHT = env_float("VAL_SCORE_FLIP_WEIGHT", 0.005)
MAX_EPISODES = env_int("MAX_EPISODES", 200)
WARMUP_STEPS = env_int("WARMUP_STEPS", 5000)

if ACTION_MODE not in {"absolute", "residual", "bc_prior_residual"}:
    raise ValueError(
        f"Unsupported ACTION_MODE={ACTION_MODE!r}; "
        "use 'absolute', 'residual', or 'bc_prior_residual'.")
if PRIOR_OUTPUT_MODE not in {"absolute", "delta"}:
    raise ValueError(
        f"Unsupported PRIOR_OUTPUT_MODE={PRIOR_OUTPUT_MODE!r}; "
        "use 'absolute' or 'delta'.")
if BC_DEMO_FILTER_MODE not in {"all", "filtered", "soft"}:
    raise ValueError(
        f"Unsupported BC_DEMO_FILTER_MODE={BC_DEMO_FILTER_MODE!r}; "
        "use 'all', 'filtered', or 'soft'.")
if VAL_SELECTION_OBJECTIVE not in {"mae", "quality_score"}:
    raise ValueError(
        f"Unsupported VAL_SELECTION_OBJECTIVE={VAL_SELECTION_OBJECTIVE!r}; "
        "use 'mae' or 'quality_score'.")

# ══════════════════════════════════════════
# 加载模型
# ══════════════════════════════════════════
try:
    model_d = joblib.load(ROOT / "models/env/boiler_model_d.pkl")
    params  = joblib.load(ROOT / "models/env/boiler_physics_params.pkl")
except FileNotFoundError:
    print("错误：缺少 boiler_model_d.pkl / boiler_physics_params.pkl")
    exit()

try:
    shadow_model = joblib.load(ROOT / "models/env/boiler_env_model.pkl")
    print("✓ 已加载影子模型")
except FileNotFoundError:
    shadow_model = None
    print("⚠ 未找到 boiler_env_model.pkl")

K, u_ss = params['K'], params['u_ss']
features_d, N_LAGS = params['features_d'], params['N_LAGS']
COL_U, COL_Y = params['col_u'], params['col_y']
COL_LOAD, COL_COAL = params['col_load'], params['col_coal']

# 【分 bin T(L) median 线性插值】
# 优先使用 params 中的 bin 数据，否则回退到旧版
if 'T_bin_centers' in params:
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
    print(f"✓ 使用分bin T(L) 线性插值 ({len(T_bin_centers)} bins, "
          f"{T_bin_centers[0]:.0f}~{T_bin_centers[-1]:.0f} MW)")
elif 'T_A' in params:
    # 回退到二次拟合（兼容旧 params）
    T_A   = params['T_A']
    T_B   = params['T_B']
    T_C   = params['T_C']
    T_min = params.get('T_min', 30.0)
    T_max = params.get('T_max', 1200.0)
    def T_interp(load):
        if isinstance(load, np.ndarray):
            T = T_A * load**2 + T_B * load + T_C
            return np.clip(T, T_min, T_max)
        else:
            T = T_A * load**2 + T_B * load + T_C
            return max(T_min, min(T_max, T))
    print(f"✓ 使用闭环辨识 T(L) = {T_A:.6f}*L^2 + {T_B:.4f}*L + {T_C:.2f}")
else:
    # 回退到旧版分段线性插值（兼容性）
    T_400 = params.get('T_400', 270.0)
    T_580 = params.get('T_580', 80.0)
    L_400 = params.get('L_400', 400.0)
    L_580 = params.get('L_580', 580.0)
    slope_inner = params.get('slope_inner', (T_580 - T_400) / (L_580 - L_400))
    slope_outer = params.get('slope_outer', slope_inner * 0.5)
    T_min = params.get('T_min', 50.0)
    def T_interp(load):
        if isinstance(load, np.ndarray):
            T = np.where(load <= L_400, T_400,
                         np.where(load <= L_580,
                                  T_400 + slope_inner * (load - L_400),
                                  T_580 + slope_outer * (load - L_580)))
            return np.clip(T, T_min, T_400)
        else:
            if load <= L_400:
                return T_400
            elif load <= L_580:
                return T_400 + slope_inner * (load - L_400)
            else:
                return max(T_min, T_580 + slope_outer * (load - L_580))
    print("⚠ 未检测到闭环辨识参数，回退到旧版分段线性插值")

# ══════════════════════════════════════════
# 影子状态增强（与原版相同）
# ══════════════════════════════════════════
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
        y_buf   = list(env.y_hist)
        u_buf   = list(env.u_hist)
        curr_u  = env.curr_u
        step    = env.step_idx
        df      = env.df
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
            errs[-1] / y_std, (preds[-1] - preds[0]) / max(self.horizon, 1) / y_std
        ], dtype=np.float32)
        return np.clip(feats, -5.0, 5.0)


def augment_state(raw_state, env, shadow_aug):
    return np.concatenate([raw_state,
                           shadow_aug.rollout(env)]).astype(np.float32)


# ══════════════════════════════════════════
# MPC 前瞻奖励
# ══════════════════════════════════════════
class MPCRewardAugmentor:
    def __init__(self, shadow_model, n_lags=10, horizon=MPC_HORIZON,
                 ctrl_horizon=CTRL_HORIZON, target=606.0,
                 lambda_y=MPC_LAMBDA_Y, lambda_u=MPC_LAMBDA_U):
        self.model         = shadow_model
        self.n_lags        = n_lags
        self.horizon       = horizon
        self.ctrl_horizon  = ctrl_horizon
        self.target        = target
        self.lambda_y      = lambda_y
        self.lambda_u      = lambda_u

    def compute(self, env, planned_u_seq: np.ndarray) -> float:
        if self.model is None:
            return 0.0

        y_buf   = list(env.y_hist)
        u_buf   = list(env.u_hist)
        step    = env.step_idx
        df      = env.df
        max_idx = len(df) - 2
        y_std   = max(env.stats.get(COL_Y + '_std', 1.0), 1e-6)
        u_std   = max(env.stats.get(COL_U + '_std', 10.0), 1e-6)

        preds = []
        for h in range(self.horizon):
            if h < self.ctrl_horizon:
                hold_u = float(planned_u_seq[h])
            else:
                hold_u = float(planned_u_seq[-1])

            idx_c = min(step + h,     max_idx)
            idx_n = min(step + h + 1, max_idx)
            h_y = list(reversed(y_buf[-self.n_lags:]))
            h_u = [hold_u] + list(reversed(u_buf[-(self.n_lags - 1):]))
            while len(h_y) < self.n_lags: h_y.append(h_y[-1] if h_y else self.target)
            while len(h_u) < self.n_lags: h_u.append(hold_u)
            feat = (h_y[:self.n_lags] + h_u[:self.n_lags] +
                    [float(df.iloc[idx_c][COL_LOAD]), float(df.iloc[idx_n][COL_LOAD]),
                     float(df.iloc[idx_c][COL_COAL]), float(df.iloc[idx_n][COL_COAL])])
            y_next = float(self.model.predict(
                np.array(feat, dtype=np.float32).reshape(1, -1))[0])
            preds.append(y_next); y_buf.append(y_next); u_buf.append(hold_u)

        if not preds:
            return 0.0
        preds = np.array(preds, dtype=np.float32)
        errs  = preds - self.target
        J_track = float(np.mean(errs ** 2)) / (y_std ** 2)
        u_diffs = np.diff([env.curr_u] + planned_u_seq.tolist())
        J_ctrl = float(np.mean(u_diffs ** 2)) / (u_std ** 2)
        r_mpc   = -(self.lambda_y * J_track + self.lambda_u * J_ctrl)
        return float(np.clip(r_mpc, -10.0, 10.0))


# ══════════════════════════════════════════
# 动作解码（M 步绝对开度）
# ══════════════════════════════════════════
DEADZONE = DEADZONE_FRAC * 100.0
BC_PRIOR_ACTOR = None


def absolute_action_to_opening(action_value: float) -> float:
    return float(np.clip((float(action_value) + 1.0) / 2.0 * 100.0, 0.0, 100.0))


def bc_prior_action_for_state(state) -> float:
    if BC_PRIOR_ACTOR is None:
        raise RuntimeError("BC prior actor is required for ACTION_MODE=bc_prior_residual.")
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        a = BC_PRIOR_ACTOR(s).cpu().numpy()[0][0]
    return float(np.clip(a, -1.0, 1.0))


def bc_prior_opening_for_state(state, curr_u: float) -> float:
    prior_action = bc_prior_action_for_state(state)
    if PRIOR_OUTPUT_MODE == "delta":
        target_u = float(curr_u) + prior_action * PRIOR_MAX_DELTA
        return float(np.clip(target_u, 0.0, 100.0))
    return absolute_action_to_opening(prior_action)

def action_to_opening(curr_u: float, action_value: float, state=None) -> float:
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


def encode_actor_action(curr_u: float, target_u: float, state=None) -> float:
    if ACTION_MODE == "bc_prior_residual":
        if state is None:
            raise RuntimeError("state is required for ACTION_MODE=bc_prior_residual.")
        prior_u = bc_prior_opening_for_state(state, curr_u)
        return float(np.clip((float(target_u) - prior_u) / RESIDUAL_MAX_DELTA,
                             -1.0, 1.0))
    if ACTION_MODE == "residual":
        return float(np.clip((float(target_u) - float(curr_u)) / RESIDUAL_MAX_DELTA,
                             -1.0, 1.0))
    return float(np.clip(float(target_u) / 100.0 * 2.0 - 1.0, -1.0, 1.0))


def current_actor_reference(curr_u: float, state=None) -> float:
    if ACTION_MODE in {"residual", "bc_prior_residual"}:
        return 0.0
    return encode_actor_action(curr_u, curr_u)

def decode_action(curr_u: float, action_arr: np.ndarray, state=None) -> np.ndarray:
    # ACTION_MODE controls whether actor output is absolute opening or residual delta.
    planned = []
    prev_u  = curr_u
    for k in range(len(action_arr)):
        target_u = action_to_opening(prev_u, float(action_arr[k]), state=state)
        if abs(target_u - prev_u) < DEADZONE:
            next_u = prev_u
        else:
            next_u = float(np.clip(target_u, 0.0, 100.0))
        planned.append(next_u)
        prev_u = next_u
    return np.array(planned, dtype=np.float32)


# ══════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s,  dtype=np.float32), np.array(a, dtype=np.float32),
                np.array(r,  dtype=np.float32), np.array(ns, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════
# 环境（适配三阶惯性 + 闭环辨识 T(L)）
# ══════════════════════════════════════════
class DemoBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, curr_a, weight=1.0):
        self.buffer.append((s, a, curr_a, float(weight)))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        s, a, curr_a, weight = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.float32),
                np.array(curr_a, dtype=np.float32),
                np.array(weight, dtype=np.float32).reshape(-1, 1))

    def __len__(self):
        return len(self.buffer)


class BoilerGymEnvPhysics:
    STATE_DIM = 18

    def __init__(self, data_path, md, phys_params, shadow_predictor=None, noise_std=0.05):
        self.model_d          = md
        self.shadow_predictor = shadow_predictor
        self.n_lags_shadow    = 10
        self.noise_std        = noise_std

        self.K, self.u_ss = phys_params['K'], phys_params['u_ss']
        self.dt = phys_params.get('dt', 60)

        # 【分 bin T(L) 参数】
        if 'T_bin_centers' in phys_params:
            self.T_bin_centers = phys_params['T_bin_centers']
            self.T_bin_values  = phys_params['T_bin_values']
            self.T_A = None  # 标记使用 bin 模式
        else:
            self.T_bin_centers = None
            self.T_bin_values  = None
            self.T_A   = phys_params.get('T_A', None)
            self.T_B   = phys_params.get('T_B', None)
            self.T_C   = phys_params.get('T_C', None)
        self.T_min = phys_params.get('T_min', 30.0)
        self.T_max = phys_params.get('T_max', 1200.0)

        # 回退兼容旧版线性插值参数
        self.T_400 = phys_params.get('T_400', 270.0)
        self.T_580 = phys_params.get('T_580', 80.0)
        self.L_400 = phys_params.get('L_400', 400.0)
        self.L_580 = phys_params.get('L_580', 580.0)
        self.slope_inner = phys_params.get('slope_inner',
                                           (self.T_580 - self.T_400) / (self.L_580 - self.L_400))
        self.slope_outer = phys_params.get('slope_outer', self.slope_inner * 0.5)

        self.feat_d = phys_params['features_d']
        self.n_lags = phys_params['N_LAGS']
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
            T_t = T_interp(load_arr[t])
            a = np.exp(-self.dt / T_t)
            b = 1.0 - a
            x1[t] = a * x1[t - 1] + b * self.K * (u_arr[t - 1] - self.u_ss)
            x2[t] = a * x2[t - 1] + b * x1[t]
            x3[t] = a * x3[t - 1] + b * x2[t]

        self.df['x1'] = x1
        self.df['x2'] = x2
        self.df['x3'] = x3
        self.df['x_ctrl']    = x3
        self.df['y_disturb'] = self.df[COL_Y] - x3

        self.df['load_delta']        = self.df[COL_LOAD].diff().fillna(0)
        self.df['load_delta_mean_5'] = (self.df['load_delta']
                                        .rolling(5, min_periods=1).mean().fillna(0))

        stat_cols = [COL_U, COL_Y, COL_LOAD, COL_COAL, 'load_delta']
        self.stats = {c + '_mean': self.df[c].mean() for c in stat_cols}
        self.stats.update({c + '_std': max(self.df[c].std(), 1e-6) for c in stat_cols})

        self.target_sp = 606.0
        self.max_steps = 1000
        self.x_scale   = max(self.K * 50.0, 1e-4)

        self.y_hist         = deque(maxlen=self.n_lags)
        self.y_disturb_hist = deque(maxlen=self.n_lags)
        self.u_hist         = deque(maxlen=self.n_lags)
        self.u_delay_queue  = deque(maxlen=self.delay + 1)
        self.prev_actual_delta = 0.0

    def _T_load(self, load_val):
        """统一的 T(L) 计算，优先用分bin线性插值"""
        if self.T_bin_centers is not None:
            T = np.interp(load_val, self.T_bin_centers, self.T_bin_values,
                          left=self.T_bin_values[0], right=self.T_bin_values[-1])
            return max(self.T_min, min(self.T_max, T))
        elif self.T_A is not None:
            T = self.T_A * load_val**2 + self.T_B * load_val + self.T_C
            return max(self.T_min, min(self.T_max, T))
        else:
            if load_val <= self.L_400:
                return self.T_400
            elif load_val <= self.L_580:
                return self.T_400 + self.slope_inner * (load_val - self.L_400)
            else:
                return max(self.T_min, self.T_580 + self.slope_outer * (load_val - self.L_580))

    def normalize(self, val, col_name, is_bias=False):
        std = self.stats[col_name + '_std']
        return val / std if is_bias else (val - self.stats[col_name + '_mean']) / std

    def reset_at(self, step_idx):
        self.current_step_count = 0
        self.accumulated_error  = 0.0
        self.prev_actual_delta  = 0.0
        max_start = max(self.n_lags, len(self.df) - self.max_steps - 5)
        self.step_idx = int(np.clip(step_idx, self.n_lags, max_start))
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

    def reset(self):
        high = max(self.n_lags + 1, len(self.df) - self.max_steps - 5)
        return self.reset_at(np.random.randint(self.n_lags, high))

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
        u_delayed      = (self.u_delay_queue[0]
                          if len(self.u_delay_queue) > self.delay else self.u_ss)

        current_load = float(self.df.iloc[self.step_idx][COL_LOAD])
        T_t = self._T_load(current_load)
        a = np.exp(-self.dt / T_t)
        b = 1.0 - a

        x1_prev, x2_prev, x3_prev = self.control_state
        next_x1 = a * x1_prev + b * self.K * (u_delayed - self.u_ss)
        next_x2 = a * x2_prev + b * next_x1
        next_x3 = a * x3_prev + b * next_x2
        new_ctrl_state = [next_x1, next_x2, next_x3]

        y_dist_next = float(self.model_d.predict(self._build_input_d())[0])
        pred_y = y_dist_next + next_x3 + np.random.normal(0, self.noise_std)
        y_shadow = pred_y
        if self.shadow_predictor is not None:
            h_y = list(self.y_hist)[-self.n_lags_shadow:][::-1]
            h_u = list(self.u_hist)[-self.n_lags_shadow + 1:][::-1]
            h_u.insert(0, next_u)
            while len(h_y) < self.n_lags_shadow: h_y.append(self.curr_y)
            while len(h_u) < self.n_lags_shadow: h_u.append(self.curr_u)
            r_, nr_ = self.df.iloc[self.step_idx], self.df.iloc[self.step_idx + 1]
            s_feat = {f'y_lag_{i}': h_y[i] for i in range(self.n_lags_shadow)}
            s_feat.update({f'u_lag_{i}': h_u[i] for i in range(self.n_lags_shadow)})
            s_feat.update({'load_curr': float(r_[COL_LOAD]), 'load_next': float(nr_[COL_LOAD]),
                           'coal_curr': float(r_[COL_COAL]), 'coal_next': float(nr_[COL_COAL])})
            f_names = ([f'y_lag_{i}' for i in range(self.n_lags_shadow)] +
                       [f'u_lag_{i}' for i in range(self.n_lags_shadow)] +
                       ['load_curr', 'load_next', 'coal_curr', 'coal_next'])
            y_shadow = float(self.shadow_predictor.predict(
                np.array([s_feat[f] for f in f_names]).reshape(1, -1))[0])

        diff     = self.target_sp - pred_y
        abs_diff = abs(diff)
        r_track  = np.exp(-0.5 * (diff / 1.5) ** 2) - 1.0
        r_cliff = 0.0
        if abs_diff > 3.5:
            r_cliff = -0.1 * min((abs_diff - 3.5) ** 2, 25.0)

        r_logic = 0.0
        if abs(actual_delta) > DEADZONE:
            if abs_diff < 1.0:
                r_logic -= 1.5
            elif abs_diff < 3.0:
                r_logic -= 0.2 * (abs(actual_delta) / 20.0)
            else:
                if abs_diff < 1.0:
                    r_logic += 0.3

            if self.prev_actual_delta * actual_delta < 0:
                r_logic -= 1.0 * (abs(actual_delta) / 20.0)

        self.accumulated_error = np.clip(self.accumulated_error + diff, -100.0, 100.0)
        self.prev_actual_delta = actual_delta
        self.control_state     = new_ctrl_state
        self.y_hist.append(pred_y)
        self.y_disturb_hist.append(y_dist_next)
        self.u_hist.append(next_u)
        self.curr_y = pred_y
        self.curr_u = next_u
        self.step_idx += 1

        done = (self.current_step_count >= self.max_steps) or \
               (self.step_idx >= len(self.df) - 5)
        r_env = r_track + r_cliff + r_logic
        info  = {
            'y_rl': pred_y, 'y_shadow': y_shadow, 'u_rl': next_u,
            'load': float(self.df.iloc[self.step_idx][COL_LOAD]),
            'coal': float(self.df.iloc[self.step_idx][COL_COAL]),
            'diff': diff, 'y_origin': float(self.df.iloc[self.step_idx][COL_Y]),
            'u_origin': float(self.df.iloc[self.step_idx][COL_U]),
            'r_track': r_track, 'r_cliff': r_cliff, 'r_logic': r_logic,
        }
        return self._get_state(), r_env, done, info


# ══════════════════════════════════════════
# DDPG（Actor 输出 M 维，Critic 接受 M 维动作）
# ══════════════════════════════════════════
def bc_lambda_for_episode(ep: int) -> float:
    if BC_REG_WARM_EPISODES > 0 and ep >= BC_REG_WARM_EPISODES:
        return 0.0
    frac = min(max(ep, 0) / max(BC_REG_DECAY_EPISODES, 1), 1.0)
    return BC_REG_LAMBDA_START + frac * (BC_REG_LAMBDA_END - BC_REG_LAMBDA_START)


def build_demo_state(env, shadow_aug, t: int, target_sp=606.0):
    row = env.df.iloc[t]
    next_row = env.df.iloc[t + 1]
    curr_y = float(env.df[COL_Y].iloc[t])
    error = env.normalize(target_sp - curr_y, COL_Y, is_bias=True)

    y_hist = env.df[COL_Y].iloc[max(0, t - env.n_lags + 1):t + 1].values
    u_hist = env.df[COL_U].iloc[max(0, t - env.n_lags + 1):t + 1].values
    y_trend = [env.normalize(x, COL_Y) for x in y_hist[-5:]]
    u_trend = [env.normalize(x, COL_U) for x in u_hist[-5:]]
    while len(y_trend) < 5:
        y_trend.insert(0, y_trend[0] if y_trend else 0.0)
    while len(u_trend) < 5:
        u_trend.insert(0, u_trend[0] if u_trend else 0.0)

    load_delta_val = env.normalize(next_row.get('load_delta', 0.0), 'load_delta')
    base = np.array(
        [error, 0.0] +
        y_trend + u_trend +
        [env.normalize(row[COL_LOAD], COL_LOAD),
         env.normalize(next_row[COL_LOAD], COL_LOAD),
         env.normalize(row[COL_COAL], COL_COAL),
         env.normalize(next_row[COL_COAL], COL_COAL),
         load_delta_val,
         env.df['x3'].iloc[t] / env.x_scale],
        dtype=np.float32)

    class Snapshot:
        pass

    snap = Snapshot()
    snap.df = env.df
    snap.stats = env.stats
    snap.y_hist = deque(y_hist[-env.n_lags:], maxlen=env.n_lags)
    snap.u_hist = deque(u_hist[-env.n_lags:], maxlen=env.n_lags)
    snap.curr_u = float(env.df[COL_U].iloc[t])
    snap.step_idx = t
    shadow = shadow_aug.rollout(snap)
    return np.concatenate([base, shadow]).astype(np.float32)


def bc_filter_reasons(env, t: int, target_sp=606.0):
    if BC_DEMO_FILTER_MODE == "all":
        return []

    reasons = []
    curr_u = float(env.df[COL_U].iloc[t])
    next_u = float(env.df[COL_U].iloc[t + 1])
    delta_u = next_u - curr_u

    if (curr_u < BC_FILTER_U_MIN or curr_u > BC_FILTER_U_MAX or
            next_u < BC_FILTER_U_MIN or next_u > BC_FILTER_U_MAX):
        reasons.append("saturation")
    if abs(delta_u) > BC_FILTER_MAX_DELTA_U:
        reasons.append("action_jump")

    curr_load = float(env.df[COL_LOAD].iloc[t])
    next_load = float(env.df[COL_LOAD].iloc[t + 1])
    load_step = max(abs(next_load - curr_load),
                    abs(float(env.df['load_delta'].iloc[t + 1])))
    if load_step > BC_FILTER_MAX_LOAD_DELTA:
        reasons.append("load_jump")

    signed_error = float(target_sp - env.df[COL_Y].iloc[t])
    curr_error = abs(signed_error)
    if (BC_FILTER_CHECK_DIRECTION and curr_error >= BC_FILTER_IMPROVE_MIN_ERROR and
            abs(delta_u) >= BC_FILTER_DEADZONE_U):
        expected_sign = np.sign(signed_error) * np.sign(K)
        if expected_sign != 0 and np.sign(delta_u) != expected_sign:
            reasons.append("wrong_direction")

    h = max(0, BC_FILTER_IMPROVE_HORIZON)
    if h > 0 and curr_error >= BC_FILTER_IMPROVE_MIN_ERROR and t + h < len(env.df):
        future_y = env.df[COL_Y].iloc[t + 1:t + h + 1].values
        best_future_error = float(np.min(np.abs(future_y - target_sp)))
        if best_future_error > curr_error - BC_FILTER_IMPROVE_TOL:
            reasons.append("no_improvement")

    return reasons


def bc_filter_reason(env, t: int, target_sp=606.0):
    reasons = bc_filter_reasons(env, t, target_sp=target_sp)
    return reasons[0] if reasons else None


def bc_demo_weight(reasons):
    if BC_DEMO_FILTER_MODE != "soft" or not reasons:
        return 1.0
    penalties = {
        "saturation": BC_SOFT_WEIGHT_SATURATION,
        "action_jump": BC_SOFT_WEIGHT_ACTION_JUMP,
        "load_jump": BC_SOFT_WEIGHT_LOAD_JUMP,
        "wrong_direction": BC_SOFT_WEIGHT_WRONG_DIRECTION,
        "no_improvement": BC_SOFT_WEIGHT_NO_IMPROVEMENT,
    }
    weight = 1.0
    for reason in set(reasons):
        weight *= penalties.get(reason, 1.0)
    return float(max(BC_SOFT_WEIGHT_MIN, min(1.0, weight)))


def build_demo_buffer(env, shadow_aug, target_sp=606.0):
    demo = DemoBuffer()
    skipped_quality = 0
    skipped_filter = {
        "saturation": 0,
        "action_jump": 0,
        "load_jump": 0,
        "wrong_direction": 0,
        "no_improvement": 0,
    }
    deadzone_snapped = 0
    t_start = env.n_lags
    t_end = len(env.df) - max(CTRL_HORIZON + 1, SHADOW_HORIZON + 1) - 1
    for t in range(t_start, t_end):
        window = env.df[COL_Y].iloc[max(0, t - BC_QUALITY_WINDOW):
                                    t + BC_QUALITY_WINDOW + 1].values
        local_mae = np.mean(np.abs(window - target_sp))
        if local_mae > BC_QUALITY_MAE_THRESHOLD:
            skipped_quality += 1
            continue

        reasons = bc_filter_reasons(env, t, target_sp=target_sp)
        for reason in reasons:
            skipped_filter[reason] += 1
        if BC_DEMO_FILTER_MODE == "filtered" and reasons:
            continue
        demo_weight = bc_demo_weight(reasons)

        s = build_demo_state(env, shadow_aug, t, target_sp=target_sp)
        a = []
        prev_u = float(env.df[COL_U].iloc[t])
        for k in range(CTRL_HORIZON):
            target_u = float(env.df[COL_U].iloc[t + k + 1])
            if (BC_DEMO_FILTER_MODE == "filtered" or
                    (BC_DEMO_FILTER_MODE == "soft" and BC_SOFT_SNAP_DEADZONE)):
                if abs(target_u - prev_u) < BC_FILTER_DEADZONE_U:
                    target_u = prev_u
                    deadzone_snapped += 1
            a.append(encode_actor_action(prev_u, target_u, state=s))
            prev_u = action_to_opening(prev_u, a[-1], state=s)
        curr_a = current_actor_reference(float(env.df[COL_U].iloc[t]), state=s)
        demo.push(s, np.array(a, dtype=np.float32),
                  np.array([curr_a], dtype=np.float32),
                  weight=demo_weight)

    total = max(t_end - t_start, 1)
    skipped_filtered_total = sum(skipped_filter.values())
    weights = [x[3] for x in demo.buffer]
    mean_weight = float(np.mean(weights)) if weights else 0.0
    print(f"[BC regularization] demo mode: {BC_DEMO_FILTER_MODE}")
    print(f"[BC regularization] demo samples: {len(demo):,}, "
          f"quality skipped: {skipped_quality:,} ({skipped_quality / total * 100:.1f}%), "
          f"flagged: {skipped_filtered_total:,} ({skipped_filtered_total / total * 100:.1f}%), "
          f"mean_weight: {mean_weight:.3f}")
    if BC_DEMO_FILTER_MODE in {"filtered", "soft"}:
        detail = ", ".join(f"{k}={v:,}" for k, v in skipped_filter.items())
        print(f"[BC {BC_DEMO_FILTER_MODE}] {detail}; deadzone snapped={deadzone_snapped:,}")
        print(f"[BC {BC_DEMO_FILTER_MODE}] u=[{BC_FILTER_U_MIN:.1f}, {BC_FILTER_U_MAX:.1f}], "
              f"max_delta_u={BC_FILTER_MAX_DELTA_U:.1f}, "
              f"max_load_delta={BC_FILTER_MAX_LOAD_DELTA:.1f}, "
              f"improve_h={BC_FILTER_IMPROVE_HORIZON}, "
              f"min_error={BC_FILTER_IMPROVE_MIN_ERROR:.1f}")
        if BC_DEMO_FILTER_MODE == "soft":
            print(f"[BC soft] min_weight={BC_SOFT_WEIGHT_MIN:.2f}, "
                  f"snap_deadzone={BC_SOFT_SNAP_DEADZONE}")
    return demo


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


class Critic(nn.Module):
    def __init__(self, state_dim, ctrl_horizon=CTRL_HORIZON):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + ctrl_horizon, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),                      nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),                       nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))


def weighted_tensor_mean(values, weights=None):
    values = values.reshape(-1)
    if weights is None:
        return values.mean()
    weights = weights.reshape(-1).to(values.device)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def weighted_action_mse(pred, target, weights=None):
    losses = F.mse_loss(pred, target, reduction="none").mean(dim=1)
    return weighted_tensor_mean(losses, weights)


def advantage_bc_weights(critic, states, demo_actions, policy_actions):
    eta = max(BC_ADV_ETA, 1e-6)
    with torch.no_grad():
        q_demo = critic(states, demo_actions)
        q_policy = critic(states, policy_actions.detach())
        adv = q_demo - q_policy
        log_min = float(np.log(max(BC_ADV_WEIGHT_MIN, 1e-6)))
        log_max = float(np.log(max(BC_ADV_WEIGHT_MAX, BC_ADV_WEIGHT_MIN)))
        adv_log = torch.clamp(adv / eta, log_min, log_max)
        weights = torch.exp(adv_log)
    return weights


class DDPGAgent:
    def __init__(self, state_dim, ctrl_horizon=CTRL_HORIZON):
        self.ctrl_horizon  = ctrl_horizon
        self.actor         = Actor(state_dim, ctrl_horizon).to(device)
        self.actor_target  = Actor(state_dim, ctrl_horizon).to(device)
        self.critic        = Critic(state_dim, ctrl_horizon).to(device)
        self.critic_target = Critic(state_dim, ctrl_horizon).to(device)
        if ACTION_MODE == "bc_prior_residual":
            nn.init.zeros_(self.actor.net[-2].weight)
            nn.init.zeros_(self.actor.net[-2].bias)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay     = ReplayBuffer()
        self.batch_size = 256
        self.gamma      = 0.99
        self.tau_soft   = 0.005

    def select_action(self, state, noise=0.0):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            a = self.actor(s).cpu().numpy()[0]
        if noise > 0:
            a += np.random.normal(0, noise, size=a.shape)
        return np.clip(a, -1.0, 1.0)

    def update(self, demo_buffer=None, bc_lambda=0.0,
               demo_batch_size=BC_DEMO_BATCH_SIZE, use_adv_weight=False):
        if len(self.replay) < self.batch_size:
            return None
        s, a, r, ns, d = self.replay.sample(self.batch_size)
        S, A, R, NS, D = [torch.FloatTensor(x).to(device) for x in [s, a, r, ns, d]]
        with torch.no_grad():
            target_q = (R.unsqueeze(1) +
                        (1 - D.unsqueeze(1)) * self.gamma *
                        self.critic_target(NS, self.actor_target(NS)))
        c_loss = nn.MSELoss()(self.critic(S, A), target_q)
        self.critic_opt.zero_grad(); c_loss.backward(); self.critic_opt.step()

        policy_actions = self.actor(S)
        q_loss = -self.critic(S, policy_actions).mean()
        bc_loss = torch.zeros((), device=device)
        bc_active_frac = 0.0
        bc_adv_weight_mean = 1.0
        if demo_buffer is not None and bc_lambda > 0.0 and len(demo_buffer) > 0:
            ds, da, curr_a, demo_w = demo_buffer.sample(demo_batch_size)
            DS = torch.FloatTensor(ds).to(device)
            DA = torch.FloatTensor(da).to(device)
            CA = torch.FloatTensor(curr_a).to(device)
            DW = torch.FloatTensor(demo_w).to(device)
            pred_demo_actions = self.actor(DS)
            if BC_REG_ADV_WEIGHT and use_adv_weight:
                adv_critic = self.critic_target if BC_ADV_USE_TARGET else self.critic
                adv_w = advantage_bc_weights(adv_critic, DS, DA, pred_demo_actions)
                bc_adv_weight_mean = float(adv_w.mean().item())
                DW = DW * adv_w
            if BC_REG_Q_FILTER:
                with torch.no_grad():
                    q_demo = self.critic(DS, DA)
                    q_policy = self.critic(DS, pred_demo_actions)
                    mask = (q_demo > q_policy + BC_Q_FILTER_MARGIN).squeeze(1)
                bc_active_frac = float(mask.float().mean().item())
                if mask.any():
                    if BC_REG_MODE == "delta":
                        DWA = DW[mask]
                        pred_delta = (pred_demo_actions[mask] - CA[mask]) * 50.0
                        demo_delta = (DA[mask] - CA[mask]) * 50.0
                        move_mask = torch.abs(demo_delta) >= BC_DELTA_DEADZONE
                        if move_mask.any():
                            move_weights = DWA.expand_as(demo_delta)[move_mask]
                            mag_vals = F.smooth_l1_loss(
                                pred_delta[move_mask] / BC_DELTA_SCALE,
                                demo_delta[move_mask] / BC_DELTA_SCALE,
                                reduction="none")
                            mag_loss = weighted_tensor_mean(mag_vals, move_weights)
                            dir_vals = F.softplus(
                                -torch.sign(demo_delta[move_mask]) *
                                pred_delta[move_mask] / BC_DELTA_SCALE)
                            dir_loss = weighted_tensor_mean(dir_vals, move_weights)
                        else:
                            mag_loss = torch.zeros((), device=device)
                            dir_loss = torch.zeros((), device=device)
                        if (~move_mask).any():
                            stay_weights = DWA.expand_as(demo_delta)[~move_mask]
                            stay_vals = F.smooth_l1_loss(
                                pred_delta[~move_mask] / BC_DELTA_SCALE,
                                torch.zeros_like(pred_delta[~move_mask]),
                                reduction="none")
                            stay_loss = weighted_tensor_mean(stay_vals, stay_weights)
                        else:
                            stay_loss = torch.zeros((), device=device)
                        bc_loss = (BC_DELTA_MAG_WEIGHT * mag_loss +
                                   BC_DELTA_DIR_WEIGHT * dir_loss +
                                   BC_DELTA_STAY_WEIGHT * stay_loss)
                    else:
                        bc_loss = weighted_action_mse(pred_demo_actions[mask], DA[mask], DW[mask])
            else:
                bc_active_frac = float(DW.mean().item())
                if BC_REG_MODE == "delta":
                    pred_delta = (pred_demo_actions - CA) * 50.0
                    demo_delta = (DA - CA) * 50.0
                    move_mask = torch.abs(demo_delta) >= BC_DELTA_DEADZONE
                    if move_mask.any():
                        move_weights = DW.expand_as(demo_delta)[move_mask]
                        mag_vals = F.smooth_l1_loss(
                            pred_delta[move_mask] / BC_DELTA_SCALE,
                            demo_delta[move_mask] / BC_DELTA_SCALE,
                            reduction="none")
                        mag_loss = weighted_tensor_mean(mag_vals, move_weights)
                        dir_vals = F.softplus(
                            -torch.sign(demo_delta[move_mask]) *
                            pred_delta[move_mask] / BC_DELTA_SCALE)
                        dir_loss = weighted_tensor_mean(dir_vals, move_weights)
                    else:
                        mag_loss = torch.zeros((), device=device)
                        dir_loss = torch.zeros((), device=device)
                    if (~move_mask).any():
                        stay_weights = DW.expand_as(demo_delta)[~move_mask]
                        stay_vals = F.smooth_l1_loss(
                            pred_delta[~move_mask] / BC_DELTA_SCALE,
                            torch.zeros_like(pred_delta[~move_mask]),
                            reduction="none")
                        stay_loss = weighted_tensor_mean(stay_vals, stay_weights)
                    else:
                        stay_loss = torch.zeros((), device=device)
                    bc_loss = (BC_DELTA_MAG_WEIGHT * mag_loss +
                               BC_DELTA_DIR_WEIGHT * dir_loss +
                               BC_DELTA_STAY_WEIGHT * stay_loss)
                else:
                    bc_loss = weighted_action_mse(pred_demo_actions, DA, DW)
        a_loss = q_loss + float(bc_lambda) * bc_loss
        self.actor_opt.zero_grad(); a_loss.backward(); self.actor_opt.step()
        for p, tp in zip(self.actor.parameters(),  self.actor_target.parameters()):
            tp.data.copy_(self.tau_soft * p.data + (1 - self.tau_soft) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau_soft * p.data + (1 - self.tau_soft) * tp.data)
        return {
            'critic_loss': float(c_loss.item()),
            'actor_loss': float(a_loss.item()),
            'q_loss': float(q_loss.item()),
            'bc_loss': float(bc_loss.item()),
            'bc_lambda': float(bc_lambda),
            'bc_active_frac': bc_active_frac,
            'bc_adv_weight_mean': bc_adv_weight_mean,
        }


# ══════════════════════════════════════════
# 训练主循环
# ══════════════════════════════════════════
def validation_start_indices(eval_env, num_starts):
    max_start = len(eval_env.df) - eval_env.max_steps - 5
    if max_start <= eval_env.n_lags:
        return [eval_env.n_lags]
    num_starts = max(1, int(num_starts))
    if num_starts == 1:
        return [int((eval_env.n_lags + max_start) // 2)]
    return [int(x) for x in np.linspace(eval_env.n_lags, max_start, num_starts)]


def validation_action_metrics(us):
    values = np.array(us, dtype=np.float32)
    if len(values) <= 1:
        du = np.array([], dtype=np.float32)
    else:
        du = np.diff(values)
    abs_du = np.abs(du)

    move_mask = abs_du >= VAL_ACTION_DEADZONE
    move_sign = np.sign(du[move_mask])
    if len(move_sign) <= 1:
        flip_rate = 0.0
    else:
        flip_rate = float(np.mean(move_sign[1:] * move_sign[:-1] < 0) * 100.0)

    return {
        'u_mean': float(np.mean(values)) if len(values) else 0.0,
        'u_min': float(np.min(values)) if len(values) else 0.0,
        'u_max': float(np.max(values)) if len(values) else 0.0,
        'mean_abs_delta_u': float(np.mean(abs_du)) if len(abs_du) else 0.0,
        'p95_abs_delta_u': float(np.percentile(abs_du, 95)) if len(abs_du) else 0.0,
        'direction_flip_rate_pct': flip_rate,
        'saturation_rate_pct': float(np.mean((values <= VAL_SAT_LOW) | (values >= VAL_SAT_HIGH)) * 100.0) if len(values) else 0.0,
    }


def validation_selection_score(stats):
    if VAL_SELECTION_OBJECTIVE == "mae":
        return stats['mae']
    return (
        stats['mae'] +
        VAL_SCORE_DELTA_WEIGHT * stats['p95_abs_delta_u'] +
        VAL_SCORE_SAT_WEIGHT * stats['saturation_rate_pct'] +
        VAL_SCORE_FLIP_WEIGHT * stats['direction_flip_rate_pct']
    )


def evaluate_actor_on_env(actor, eval_env, eval_shadow_aug, num_starts=3):
    was_training = actor.training
    actor.eval()
    diffs = []
    us = []
    with torch.no_grad():
        for start_idx in validation_start_indices(eval_env, num_starts):
            s_raw = eval_env.reset_at(start_idx)
            s = augment_state(s_raw, eval_env, eval_shadow_aug)
            for _ in range(eval_env.max_steps):
                s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
                a_raw = actor(s_tensor).cpu().numpy()[0]
                planned_u = decode_action(eval_env.curr_u, a_raw, state=s)
                ns_raw, _, done, info = eval_env.step(planned_u[0])
                diffs.append(float(info['diff']))
                us.append(float(info['u_rl']))
                s = augment_state(ns_raw, eval_env, eval_shadow_aug)
                if done:
                    break
    if was_training:
        actor.train()
    diffs = np.array(diffs, dtype=np.float32)
    abs_diffs = np.abs(diffs)
    stats = {
        'mae': float(np.mean(abs_diffs)),
        'rmse': float(np.sqrt(np.mean(diffs ** 2))),
        'in2': float(np.mean(abs_diffs <= 2.0) * 100.0),
        'in5': float(np.mean(abs_diffs <= 5.0) * 100.0),
        'n': int(len(diffs)),
    }
    stats.update(validation_action_metrics(us))
    stats['selection_score'] = float(validation_selection_score(stats))
    return stats

if __name__ == "__main__":
    # ── 随机种子：两组消融实验设同一种子，消除起始点/初始化/采样的随机差异 ──
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"随机种子 SEED = {SEED}")

    print(f"TRAIN_DATA_PATH = {TRAIN_DATA_PATH}")
    env = BoilerGymEnvPhysics(TRAIN_DATA_PATH, model_d, params,
                              shadow_predictor=shadow_model)

    shadow_aug = ShadowAugmentor(shadow_model, n_lags=10,
                                 horizon=SHADOW_HORIZON, target=env.target_sp)
    use_val_selection = USE_VAL_SELECTION
    val_env = None
    val_shadow_aug = None
    if use_val_selection:
        if VAL_DATA_PATH is None or not VAL_DATA_PATH.exists():
            print("[Validation] VAL_DATA_PATH missing; fall back to train-reward selection.")
            use_val_selection = False
        else:
            val_env = BoilerGymEnvPhysics(VAL_DATA_PATH, model_d, params,
                                          shadow_predictor=shadow_model,
                                          noise_std=0.0)
            val_env.max_steps = max(1, min(VAL_MAX_STEPS,
                                           len(val_env.df) - val_env.n_lags - 5))
            val_shadow_aug = ShadowAugmentor(shadow_model, n_lags=10,
                                             horizon=SHADOW_HORIZON,
                                             target=val_env.target_sp)
            print(f"[Validation] path={VAL_DATA_PATH}, every={VAL_EVERY_EPISODES} eps, "
                  f"starts={VAL_NUM_STARTS}, steps/start={val_env.max_steps}")
    mpc_aug    = MPCRewardAugmentor(shadow_model, n_lags=10,
                                    horizon=MPC_HORIZON, ctrl_horizon=CTRL_HORIZON,
                                    target=env.target_sp,
                                    lambda_y=MPC_LAMBDA_Y, lambda_u=MPC_LAMBDA_U)

    AUG_STATE_DIM = env.STATE_DIM + ShadowAugmentor.N_SHADOW_FEATS
    print(f"\nSTATE_DIM = {env.STATE_DIM} + 4 (影子) = {AUG_STATE_DIM}")
    print(f"ACTION_DIM = {CTRL_HORIZON}（控制时域 M={CTRL_HORIZON} 步，单步）")
    print(f"死区 {DEADZONE_FRAC*100:.0f}%\n")

    print(f"ACTION_MODE = {ACTION_MODE}")
    if ACTION_MODE in {"residual", "bc_prior_residual"}:
        print(f"RESIDUAL_MAX_DELTA = {RESIDUAL_MAX_DELTA:.1f}% opening per step")
    if ACTION_MODE == "bc_prior_residual":
        print(f"BC_PRIOR_ACTOR_PATH = {BC_PRIOR_ACTOR_PATH}")
        print(f"PRIOR_OUTPUT_MODE = {PRIOR_OUTPUT_MODE}")
        if PRIOR_OUTPUT_MODE == "delta":
            print(f"PRIOR_MAX_DELTA = {PRIOR_MAX_DELTA:.1f}% opening per step")
    print()

    agent = DDPGAgent(AUG_STATE_DIM, CTRL_HORIZON)
    if ACTION_MODE == "bc_prior_residual":
        try:
            BC_PRIOR_ACTOR = Actor(AUG_STATE_DIM, CTRL_HORIZON).to(device)
            bc_prior_state = torch.load(BC_PRIOR_ACTOR_PATH, map_location=device)
            BC_PRIOR_ACTOR.load_state_dict(bc_prior_state)
            BC_PRIOR_ACTOR.eval()
            for p in BC_PRIOR_ACTOR.parameters():
                p.requires_grad_(False)
            print(f"[BC prior residual] loaded prior actor: {BC_PRIOR_ACTOR_PATH}")
        except Exception as e:
            raise RuntimeError(f"Failed to load BC prior actor from {BC_PRIOR_ACTOR_PATH}: {e}")
    demo_buffer = build_demo_buffer(env, shadow_aug, target_sp=env.target_sp) if USE_BC_REG else None
    if USE_BC_REG:
        print(f"[BC regularization] lambda: {BC_REG_LAMBDA_START:.3f} -> "
              f"{BC_REG_LAMBDA_END:.3f} over {BC_REG_DECAY_EPISODES} episodes")
        print(f"[BC regularization] q_filter={BC_REG_Q_FILTER}, "
              f"margin={BC_Q_FILTER_MARGIN:.3f}")
        print(f"[BC advantage] enabled={BC_REG_ADV_WEIGHT}, "
              f"start_ep={BC_ADV_START_EPISODE}, eta={BC_ADV_ETA:.3f}, "
              f"clip=[{BC_ADV_WEIGHT_MIN:.2f}, {BC_ADV_WEIGHT_MAX:.2f}], "
              f"target_critic={BC_ADV_USE_TARGET}")
        print(f"[BC regularization] mode={BC_REG_MODE}")
        print(f"[BC regularization] demo_filter={BC_DEMO_FILTER_MODE}")
        if BC_DEMO_FILTER_MODE == "soft":
            print(f"[BC soft] weights: min={BC_SOFT_WEIGHT_MIN:.2f}, "
                  f"sat={BC_SOFT_WEIGHT_SATURATION:.2f}, "
                  f"jump={BC_SOFT_WEIGHT_ACTION_JUMP:.2f}, "
                  f"wrong_dir={BC_SOFT_WEIGHT_WRONG_DIRECTION:.2f}, "
                  f"no_improve={BC_SOFT_WEIGHT_NO_IMPROVEMENT:.2f}")
        if BC_REG_MODE == "delta":
            print(f"[BC delta] scale={BC_DELTA_SCALE:.2f}, "
                  f"deadzone={BC_DELTA_DEADZONE:.2f}, "
                  f"weights=({BC_DELTA_MAG_WEIGHT:.2f}, "
                  f"{BC_DELTA_DIR_WEIGHT:.2f}, {BC_DELTA_STAY_WEIGHT:.2f})")
        if BC_REG_WARM_EPISODES > 0:
            print(f"[BC regularization] warm-only episodes: {BC_REG_WARM_EPISODES}")

    # ══════════════════════════════════════════
    # 消融开关：是否用 BC 权重热启动 Actor
    #   USE_BC = True  → 加载 bc_actor.pth 作为 Actor 起点（BC + RL）
    #   USE_BC = False → Actor 随机初始化（纯 RL 对照组）
    # 两组实验除此之外完全一致，保证对比公平
    # ══════════════════════════════════════════
    USE_BC        = False
    BC_ACTOR_PATH = ROOT / "models/actors/bc_actor.pth"
    if USE_BC:
        try:
            bc_state = torch.load(BC_ACTOR_PATH, map_location=device)
            agent.actor.load_state_dict(bc_state)
            agent.actor_target.load_state_dict(bc_state)   # target 同步
            print(f"✓ [BC热启动] 已加载 {BC_ACTOR_PATH} 作为 Actor 起点")
        except Exception as e:
            print(f"✗ BC 权重加载失败，回退随机初始化: {e}")
            USE_BC = False
    else:
        print("▶ [纯RL对照] Actor 随机初始化")
    # 产物标签：两组实验产物自动分开，避免互相覆盖
    TAG = "withBC" if USE_BC else "pureRL"
    if USE_BC and USE_BC_REG and BC_REG_Q_FILTER:
        TAG = "withBCQReg"
    elif USE_BC and USE_BC_REG:
        TAG = "withBCReg"
    elif USE_BC:
        TAG = "withBC"
    elif USE_BC_REG and BC_REG_Q_FILTER:
        TAG = "bcQReg"
    elif USE_BC_REG:
        TAG = "bcReg"
    else:
        TAG = "pureRL"
    if USE_BC_REG and BC_DEMO_FILTER_MODE == "filtered" and not RUN_TAG:
        TAG = f"{TAG}Filtered"
    if USE_BC_REG and BC_DEMO_FILTER_MODE == "soft" and not RUN_TAG:
        TAG = f"{TAG}SoftFilter"
    if USE_BC_REG and BC_REG_ADV_WEIGHT and not RUN_TAG:
        TAG = f"{TAG}Adv"
    if RUN_TAG:
        TAG = RUN_TAG
    ACTOR_OUT = ROOT / "models/actors" / f"best_actor_{TAG}.pth"
    PKL_OUT   = ROOT / "results/checkpoints" / f"multistep_best_{TAG}.pkl"
    FIG_OUT   = ROOT / "results/figures" / f"final_report_{TAG}.png"
    TRAIN_REWARD_ACTOR_OUT = ROOT / "models/actors" / f"best_actor_{TAG}_trainReward.pth"
    TRAIN_REWARD_PKL_OUT = ROOT / "results/checkpoints" / f"multistep_best_{TAG}_trainReward.pkl"
    VAL_HISTORY_OUT = ROOT / "results/evaluation" / f"validation_history_{TAG}.csv"
    if use_val_selection:
        print(f"[Validation] actor selected by {VAL_SELECTION_OBJECTIVE} -> {ACTOR_OUT}")
        if VAL_SELECTION_OBJECTIVE == "quality_score":
            print(f"[Validation] score = MAE + {VAL_SCORE_DELTA_WEIGHT:.4f}*p95|du| + "
                  f"{VAL_SCORE_SAT_WEIGHT:.4f}*sat% + "
                  f"{VAL_SCORE_FLIP_WEIGHT:.4f}*flip%")
        print(f"[Validation] train-reward actor backup -> {TRAIN_REWARD_ACTOR_OUT}")
    print(f"产物标签 TAG = {TAG}  (actor→{ACTOR_OUT})")
    print()

    print("[Warm-up] 填充经验池...")
    s_raw      = env.reset()
    s          = augment_state(s_raw, env, shadow_aug)

    for _ in range(WARMUP_STEPS):
        a_raw       = np.random.uniform(-1, 1, (CTRL_HORIZON,))
        planned_u   = decode_action(env.curr_u, a_raw, state=s)
        r_mpc       = mpc_aug.compute(env, planned_u)
        ns_raw, r_env, d, _ = env.step(planned_u[0])
        r_total     = r_env + r_mpc
        ns          = augment_state(ns_raw, env, shadow_aug)
        agent.replay.push(s, a_raw, r_total, ns, d)
        if d:
            s_raw = env.reset()
            s = augment_state(s_raw, env, shadow_aug)
        else:
            s = ns

    print(f"预热完成，共 {len(agent.replay)} 条经验\n")

    best_reward, best_history = -float('inf'), None
    best_val_score = float('inf')
    val_history = []
    no_improve_count = 0
    frozen           = False
    FREEZE_THRESHOLD = -350.0
    EARLY_STOP_PAT   = 30

    for ep in range(MAX_EPISODES):
        s_raw     = env.reset()
        s         = augment_state(s_raw, env, shadow_aug)
        ep_reward = 0.0
        ep_data   = {k: [] for k in ['y_rl', 'y_shadow', 'u_rl', 'load', 'coal',
                                      'diff', 'y_origin', 'u_origin',
                                      'r_track', 'r_cliff', 'r_logic', 'r_mpc']}
        noise     = max(0.05, 0.3 * (0.97 ** ep))
        bc_lambda = bc_lambda_for_episode(ep) if USE_BC_REG else 0.0
        update_logs = []

        for t in range(env.max_steps):
            a_raw     = agent.select_action(s, noise)
            planned_u = decode_action(env.curr_u, a_raw, state=s)

            r_mpc = mpc_aug.compute(env, planned_u)
            ns_raw, r_env, d, info = env.step(planned_u[0])
            ns    = augment_state(ns_raw, env, shadow_aug)

            r_total = r_env + r_mpc
            agent.replay.push(s, a_raw, r_total, ns, d)
            if not frozen and (t % 2 == 0):
                update_info = agent.update(
                    demo_buffer=demo_buffer,
                    bc_lambda=bc_lambda,
                    demo_batch_size=BC_DEMO_BATCH_SIZE,
                    use_adv_weight=(BC_REG_ADV_WEIGHT and ep >= BC_ADV_START_EPISODE))
                if update_info is not None:
                    update_logs.append(update_info)

            s, ep_reward = ns, ep_reward + r_total
            info['r_mpc'] = r_mpc
            for k in ep_data:
                ep_data[k].append(info.get(k, 0))
            if d:
                break

        mae         = np.mean(np.abs(ep_data['diff']))
        r_mpc_mean  = np.mean(ep_data['r_mpc'])
        bc_loss_mean = (np.mean([x['bc_loss'] for x in update_logs])
                        if update_logs else 0.0)
        bc_active_mean = (np.mean([x['bc_active_frac'] for x in update_logs])
                          if update_logs else 0.0)
        status      = "[冻结]" if frozen else ""
        print(f"Ep {ep+1:3d} | Reward: {ep_reward:8.1f} | MAE: {mae:.2f}°C | "
              f"r_mpc: {r_mpc_mean:.3f} | BC lambda: {bc_lambda:.3f} | "
              f"BC loss: {bc_loss_mean:.4f} | BC active: {bc_active_mean:.2f} | "
              f"Noise: {noise:.3f} {status}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            if not use_val_selection:
                best_history = ep_data
            no_improve_count = 0
            (ROOT / "results/checkpoints").mkdir(parents=True, exist_ok=True)
            (ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
            if use_val_selection:
                torch.save(agent.actor.state_dict(), TRAIN_REWARD_ACTOR_OUT)
                joblib.dump(ep_data, TRAIN_REWARD_PKL_OUT)
            else:
                torch.save(agent.actor.state_dict(), ACTOR_OUT)
                joblib.dump(ep_data, PKL_OUT)
            print(f"  ★ 保存最优 (Reward={best_reward:.1f})")
            if ep_reward > FREEZE_THRESHOLD and not frozen:
                frozen = True
                print(f"  ▶ 网络已冻结")
        else:
            no_improve_count += 1
            if frozen and no_improve_count >= EARLY_STOP_PAT:
                print(f"  ■ 提前停止 (Ep {ep+1})")
                break

    # ── 可视化 ──
        if (use_val_selection and val_env is not None and
                ((ep + 1) == 1 or (ep + 1) % VAL_EVERY_EPISODES == 0)):
            val_stats = evaluate_actor_on_env(agent.actor, val_env, val_shadow_aug,
                                              num_starts=VAL_NUM_STARTS)
            val_row = {
                'episode': ep + 1,
                'train_reward': float(ep_reward),
                'train_mae': float(mae),
                'val_mae': val_stats['mae'],
                'val_rmse': val_stats['rmse'],
                'val_in2': val_stats['in2'],
                'val_in5': val_stats['in5'],
                'val_n': val_stats['n'],
                'val_u_mean': val_stats['u_mean'],
                'val_u_min': val_stats['u_min'],
                'val_u_max': val_stats['u_max'],
                'val_mean_abs_delta_u': val_stats['mean_abs_delta_u'],
                'val_p95_abs_delta_u': val_stats['p95_abs_delta_u'],
                'val_saturation_rate_pct': val_stats['saturation_rate_pct'],
                'val_direction_flip_rate_pct': val_stats['direction_flip_rate_pct'],
                'val_selection_score': val_stats['selection_score'],
                'val_selection_objective': VAL_SELECTION_OBJECTIVE,
            }
            val_history.append(val_row)
            print(f"  [Validation] MAE={val_stats['mae']:.3f} | "
                  f"RMSE={val_stats['rmse']:.3f} | "
                  f"+/-2={val_stats['in2']:.2f}% | +/-5={val_stats['in5']:.2f}% | "
                  f"p95|du|={val_stats['p95_abs_delta_u']:.2f} | "
                  f"sat={val_stats['saturation_rate_pct']:.2f}% | "
                  f"flip={val_stats['direction_flip_rate_pct']:.2f}% | "
                  f"score={val_stats['selection_score']:.3f}")
            if val_stats['selection_score'] < best_val_score:
                best_val_score = val_stats['selection_score']
                best_history = ep_data
                (ROOT / "models/actors").mkdir(parents=True, exist_ok=True)
                (ROOT / "results/checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save(agent.actor.state_dict(), ACTOR_OUT)
                joblib.dump(ep_data, PKL_OUT)
                print(f"  [Validation best] saved actor "
                      f"(score={best_val_score:.3f}, Val MAE={val_stats['mae']:.3f})")
    if val_history:
        VAL_HISTORY_OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(val_history).to_csv(VAL_HISTORY_OUT, index=False, encoding='utf-8-sig')
        print(f"[Validation] history saved: {VAL_HISTORY_OUT}")

    if best_history is None:
        raise RuntimeError("No best actor was selected; check validation/training configuration.")

    t_ax = np.arange(len(best_history['y_rl']))
    fig  = plt.figure(figsize=(15, 18))
    gs   = gridspec.GridSpec(5, 1, height_ratios=[2, 1, 1, 1, 1.2], hspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    ax1.axhline(606, color='green', linestyle='--', linewidth=1.5, label='目标 606°C')
    ax1.plot(t_ax, best_history['y_origin'], 'gray', alpha=0.4, linewidth=1, label='历史手操')
    ax1.plot(t_ax, best_history['y_rl'], 'crimson', linewidth=1.5, label='RL+MPC+多步动作')
    ax1.set_ylabel("汽温 (°C)"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title(f"[{TAG}] DDPG + MPC前瞻 + 控制时域M={CTRL_HORIZON}  (分bin T插值)  Best Reward={best_reward:.1f}",
                  fontsize=13, fontweight='bold')

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t_ax, best_history['u_origin'], 'gray', alpha=0.4, linewidth=1, label='手操开度')
    ax2.plot(t_ax, best_history['u_rl'],     'dodgerblue', linewidth=1.5, label='RL开度')
    ax2.set_ylabel("开度 (%)"); ax2.legend(); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t_ax, best_history['load'], 'darkgreen', linewidth=1, label='负荷 (MW)')
    ax3.set_ylabel("负荷 (MW)"); ax3.legend(); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(t_ax, best_history['diff'], 'crimson', alpha=0.7, linewidth=1)
    ax4.fill_between(t_ax, -2, 2, color='green', alpha=0.1, label='±2°C 目标带')
    ax4.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax4.set_ylabel("偏差 (°C)"); ax4.legend(); ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.stackplot(t_ax,
                  np.array(best_history['r_track']),
                  np.array(best_history['r_cliff']),
                  np.array(best_history['r_logic']),
                  np.array(best_history['r_mpc']),
                  labels=['r_track', 'r_cliff', 'r_logic', 'r_mpc（MPC前瞻）'],
                  colors=['#4CAF50', '#F44336', '#FF9800', '#2196F3'], alpha=0.75)
    ax5.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax5.set_ylabel("奖励分量"); ax5.set_xlabel("时间步")
    ax5.legend(loc='lower right', fontsize=8, ncol=2); ax5.grid(alpha=0.3)

    plt.suptitle(f"DDPG + MPC前瞻奖励 + 控制时域 M={CTRL_HORIZON} —— 分bin T插值版本",
                 fontsize=14, fontweight='bold')
    plt.savefig(FIG_OUT, dpi=150, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    print(f"\n图表已保存: {FIG_OUT}")
