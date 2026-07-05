import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据加载与降采样
# ==========================================
print("=" * 60)
print("  三阶惯性(分bin T插值) + 残差解耦 灰盒环境模型训练")
print("=" * 60)

df = pd.read_csv(ROOT / "data/raw/RHT0121quan.csv", encoding="gbk")

COL_U    = "AI_RHT_GP_CO: 烟气挡板控制指令AI[159]*100"
COL_Y    = "AI_RHT_GP_PV: 烟气挡板再热汽温选择值AI[158]*50"
try:
    COL_LOAD = [c for c in df.columns if "MW" in c or "负荷" in c][0]
except:
    COL_LOAD = "AI_CCS_MW_SP"
COL_COAL = [c for c in df.columns if "煤" in c or "FDR_ALL" in c][0]

DT = 60
agg_dict = {
    COL_U:    'mean',
    COL_Y:    'last',
    COL_LOAD: 'mean',
    COL_COAL: 'mean',
}
df = (df[[COL_U, COL_Y, COL_LOAD, COL_COAL]]
      .groupby(df.index // DT)
      .agg(agg_dict)
      .reset_index(drop=True))

print(f"  降采样后长度: {len(df):,} 步 ({DT}秒/步)")


# ==========================================
# 2. 三阶惯性机理模型参数 + 闭环辨识 T(L)
# ==========================================
print("\n" + "=" * 60)
print("  Step 1: 电厂机理模型参数（分bin T(L) median 线性插值）")
print("=" * 60)

K = 0.74
N_ORDER = 3
u_ss = float(df[COL_U].mean())

# T(L) = 负荷区间 median T 的线性插值
# 来自 1111_528 闭环数据 61段有效拟合，按 25MW bins 分组取 median
T_bin_centers = np.array(
    [100, 125, 300, 325, 350, 375, 400, 425, 450, 475,
     500, 525, 550, 575, 600, 625, 650, 675, 700, 725,
     750, 775, 800, 825, 850, 900, 925, 950, 975, 1000, 1025],
    dtype=np.float64)
T_bin_values = np.array(
    [24.7, 158.9, 880.5, 352.0, 558.4, 780.4, 283.2, 564.0, 349.8, 387.6,
     376.4, 241.4, 258.9, 262.4, 349.5, 231.1, 433.1, 200.0, 277.3, 347.1,
     201.7, 171.6, 166.3, 334.1, 113.0, 126.5, 114.7, 81.4, 76.3, 116.5, 63.9],
    dtype=np.float64)
T_MIN = 30.0
T_MAX = 1200.0

def T_interp(load):
    """
    T(L) = 负荷区间 median T 的线性插值
    基于 1111_528 数据 61段有效拟合，25MW bins
    边界外 clamp 到最近 bin 的值，再经 [T_MIN, T_MAX] 硬限幅
    """
    if isinstance(load, np.ndarray):
        T = np.interp(load, T_bin_centers, T_bin_values,
                      left=T_bin_values[0], right=T_bin_values[-1])
        return np.clip(T, T_MIN, T_MAX)
    else:
        T = np.interp(load, T_bin_centers, T_bin_values,
                      left=T_bin_values[0], right=T_bin_values[-1])
        return max(T_MIN, min(T_MAX, T))


print(f"  K       = {K:.2f} °C/%")
print(f"  n       = {N_ORDER}")
print(f"  T(L)    = 负荷区间 median T 线性插值 ({len(T_bin_centers)} bins, "
      f"{T_bin_centers[0]:.0f}~{T_bin_centers[-1]:.0f} MW)")
print(f"  T_min   = {T_MIN:.0f} s")
print(f"  T_max   = {T_MAX:.0f} s")
print(f"  u_ss    = {u_ss:.2f} %")

load_min, load_max = df[COL_LOAD].min(), df[COL_LOAD].max()
print(f"\n  数据负荷范围: [{load_min:.0f}, {load_max:.0f}] MW")
print(f"  T({load_min:.0f}MW) = {T_interp(load_min):.1f} s")
print(f"  T({load_max:.0f}MW) = {T_interp(load_max):.1f} s")

# 检查几个关键点
for L_check in [300, 400, 500, 580, 700, 900, 1000]:
    T_check = T_interp(float(L_check))
    alpha_check = np.exp(-DT / T_check)
    print(f"  T({L_check}MW) = {T_check:.1f}s  α={alpha_check:.4f}")


# ==========================================
# 2.5 三阶惯性递推 + 信号解耦
# ==========================================
print("\n" + "=" * 60)
print("  Step 1.5: 三阶惯性递推 + 信号解耦")
print("=" * 60)

u_arr = df[COL_U].values
load_arr = df[COL_LOAD].values

x1 = np.zeros(len(df))
x2 = np.zeros(len(df))
x3 = np.zeros(len(df))

for t in range(1, len(df)):
    T_t = T_interp(load_arr[t])
    alpha_t = np.exp(-DT / T_t)
    beta_t = 1.0 - alpha_t

    x1[t] = alpha_t * x1[t-1] + beta_t * K * (u_arr[t-1] - u_ss)
    x2[t] = alpha_t * x2[t-1] + beta_t * x1[t]
    x3[t] = alpha_t * x3[t-1] + beta_t * x2[t]

df['x_ctrl'] = x3
df['y_disturb'] = df[COL_Y] - df['x_ctrl']

print(f"  原始温度波动区间: [{df[COL_Y].min():.1f}, {df[COL_Y].max():.1f}]")
print(f"  控制分量区间:     [{df['x_ctrl'].min():.3f}, {df['x_ctrl'].max():.3f}]")
print(f"  纯扰动温度区间:   [{df['y_disturb'].min():.1f}, {df['y_disturb'].max():.1f}]")

ctrl_range = df['x_ctrl'].max() - df['x_ctrl'].min()
expected_range = K * 100
print(f"\n  控制分量波动范围: {ctrl_range:.1f}°C（理论全程: {expected_range:.0f}°C）")
if ctrl_range > expected_range * 2:
    print("  ⚠ 控制分量波动异常偏大")
else:
    print("  ✓ 控制分量范围合理")


# ==========================================
# 3. 扰动模型 Model_d
# ==========================================
print("\n" + "=" * 60)
print("  Step 2: 训练纯扰动模型 Model_d")
print("=" * 60)

N_LAGS = 10
data = pd.DataFrame()
data['y']         = df[COL_Y]
data['y_disturb'] = df['y_disturb']
data['x_ctrl']    = df['x_ctrl']
data['load']      = df[COL_LOAD]
data['coal']      = df[COL_COAL]

for i in range(1, N_LAGS):
    data[f'dy_disturb_{i}'] = data['y_disturb'].shift(i - 1) - data['y_disturb'].shift(i)

data['dy_disturb_mean_5']  = data['y_disturb'].diff(1).rolling(5, min_periods=1).mean()
data['dy_disturb_mean_10'] = data['y_disturb'].diff(1).rolling(N_LAGS - 1, min_periods=1).mean()
data['dy_disturb_std_5']   = data['y_disturb'].diff(1).rolling(5, min_periods=1).std().fillna(0)
data['y_disturb_curr']     = data['y_disturb']

data['load_curr']  = data['load']
data['load_next']  = data['load'].shift(-1)
data['load_delta'] = data['load'].diff()
for i in range(1, N_LAGS):
    data[f'load_lag_{i}'] = data['load'].shift(i)
data['load_delta_mean_5'] = data['load_delta'].rolling(5, min_periods=1).mean()
data['coal_curr']  = data['coal']
data['coal_next']  = data['coal'].shift(-1)

data['target_y_disturb_next'] = data['y_disturb'].shift(-1)
data['target_y_next']         = data['y'].shift(-1)
data['x_ctrl_next']           = data['x_ctrl'].shift(-1)

data = data.dropna().reset_index(drop=True)

features_d = (
    ['y_disturb_curr'] +
    [f'dy_disturb_{i}' for i in range(5, N_LAGS)] +
    ['dy_disturb_mean_5', 'dy_disturb_mean_10', 'dy_disturb_std_5'] +
    ['load_curr', 'load_next', 'load_delta', 'load_delta_mean_5'] +
    [f'load_lag_{i}' for i in range(1, N_LAGS)] +
    ['coal_curr', 'coal_next']
)

print(f"  特征数: {len(features_d)}  |  样本数: {len(data):,}")

train_size = int(len(data) * 0.85)
train = data.iloc[:train_size]
test  = data.iloc[train_size:]

model_d = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)
model_d.fit(
    train[features_d], train['target_y_disturb_next'],
    eval_set=[(test[features_d], test['target_y_disturb_next'])],
    eval_metric='l1',
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

pred_y_disturb_test  = model_d.predict(test[features_d])
pred_y_test_combined = pred_y_disturb_test + test['x_ctrl_next'].values

mae_combined  = mean_absolute_error(test['target_y_next'], pred_y_test_combined)
rmse_combined = np.sqrt(mean_squared_error(test['target_y_next'], pred_y_test_combined))

print(f"\n  综合测试集精度: MAE={mae_combined:.4f}°C  RMSE={rmse_combined:.4f}°C")


# ==========================================
# 4. 保存
# ==========================================
physics_params = {
    'K':          K,
    'n_order':    N_ORDER,
    'T_bin_centers': T_bin_centers,
    'T_bin_values':  T_bin_values,
    'T_min':      T_MIN,
    'T_max':      T_MAX,
    'dt':         DT,
    'u_ss':       u_ss,
    'features_d': features_d,
    'mae_d':      mae_combined,
    'col_u':      COL_U,
    'col_y':      COL_Y,
    'col_load':   COL_LOAD,
    'col_coal':   COL_COAL,
    'N_LAGS':     N_LAGS,
}
joblib.dump(model_d,        ROOT / "models/env/boiler_model_d.pkl")
joblib.dump(physics_params, ROOT / "models/env/boiler_physics_params.pkl")
print(f"\n✓ boiler_model_d.pkl")
print(f"✓ boiler_physics_params.pkl")


# ==========================================
# 5. 可视化
# ==========================================
n_show = 1000
t_ax   = np.arange(n_show)
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

y_true_show = test['target_y_next'].iloc[:n_show].values
axes[0].plot(t_ax, y_true_show, 'k-', linewidth=1.2, label='真实 y[t+1]')
axes[0].plot(t_ax, pred_y_test_combined[:n_show], 'b--', linewidth=1.2,
             label=f'三阶惯性解耦预测（MAE={mae_combined:.4f}°C）')
axes[0].set_ylabel("温度 (°C)"); axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)
axes[0].set_title(f"三阶惯性解耦模型  K={K}  n={N_ORDER}  T={T_MIN:.0f}~{T_MAX:.0f}s(分bin插值)",
                  fontsize=12, fontweight='bold')

err = y_true_show - pred_y_test_combined[:n_show]
axes[1].fill_between(t_ax, err, 0, alpha=0.5, color='steelblue', label='预测误差')
axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[1].set_ylabel("误差 (°C)"); axes[1].set_xlabel("时间步")
axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ROOT / "results/training/closed_loop_3order_T.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n=== 完成 ===")
