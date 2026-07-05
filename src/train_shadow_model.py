import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error , mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from pathlib import Path  # 用于保存模型


ROOT = Path(__file__).resolve().parents[1]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid" , font='SimHei' , font_scale=1.1)

# ==========================================
# 1. 数据加载与变量筛选
# ==========================================
df = pd.read_csv(ROOT / "data/raw/RHT0121quan.csv", encoding="gbk")


COL_U = "AI_RHT_GP_CO: 烟气挡板控制指令AI[159]*100"
COL_Y = "AI_RHT_GP_PV: 烟气挡板再热汽温选择值AI[158]*50"
try:
    COL_LOAD = [c for c in df.columns if "MW" in c or "负荷" in c][0]
except:
    COL_LOAD = "AI_CCS_MW_SP"

COL_COAL = [c for c in df.columns if "煤" in c or "FDR_ALL" in c][0]


print(f"【RL环境模型构建】变量映射:")
print(f"- 挡板 (u): {COL_U}")
print(f"- 气温 (y): {COL_Y}")
print(f"- 负荷 (L): {COL_LOAD}")
print(f"- 煤量 (C): {COL_COAL}")

data = pd.DataFrame()
data['u'] = df[COL_U]
data['y'] = df[COL_Y]
data['load'] = df[COL_LOAD]
data['coal'] = df[COL_COAL]

# ==========================================
# 2. 特征工程 (构建状态空间 State Space)
# ==========================================

N_LAGS = 10  # 历史回顾窗口 (过去100秒)

# 2.1 历史状态 (Autoregressive features)
for i in range(0 , N_LAGS):
    # i=0 代表当前时刻 t
    data[f'y_lag_{i}'] = data['y'].shift(i)
    data[f'u_lag_{i}'] = data['u'].shift(i)

# 2.2 能量输入 (扰动)
# 环境演变取决于：当前时刻的负荷/煤 + 下一时刻的负荷/煤 (积分效应)
# 使用 shift(-1) 获取下一时刻的值作为输入特征
data['load_curr'] = data['load']
data['load_next'] = data['load'].shift(-1)  # 未来时刻的值作为输入
data['coal_curr'] = data['coal']
data['coal_next'] = data['coal'].shift(-1)  # 未来时刻的值作为输入

# 2.3 预测目标
data['target_y_next'] = data['y'].shift(-1)

# 删除因 lag 和 shift 产生的空值
data = data.dropna().reset_index(drop=True)

print(f"特征工程完成。当前用于预测下一时刻温度的特征数量: {len(data.columns) - 1}")

# ==========================================
# 3. 训练环境模型 (Transition Model)
# ==========================================
train_size = int(len(data) * 0.85)
train = data.iloc[:train_size]
test = data.iloc[train_size:]

# 输入特征列表：
# 1. 历史温度 (y_t, y_t-1...)
# 2. 历史挡板 (u_t, u_t-1...)
# 3. 负荷/煤 (当前+未来)
features = [f'y_lag_{i}' for i in range(N_LAGS)] + \
           [f'u_lag_{i}' for i in range(N_LAGS)] + \
           ['load_curr' , 'load_next' , 'coal_curr' , 'coal_next']

print(f"\n正在训练环境模拟器 (LightGBM)...")
model_env = lgb.LGBMRegressor(
    n_estimators=5000 ,
    learning_rate=0.01 ,
    num_leaves=63 ,
    random_state=42 ,
    n_jobs=-1
)

model_env.fit(
    train[features] ,
    train['target_y_next'] ,
    eval_set=[(test[features] , test['target_y_next'])] ,
    eval_metric='l1' ,
    callbacks=[lgb.early_stopping(100) , lgb.log_evaluation(500)]
)

# ==========================================
# 4. 验证与保存 (作为 RL 的 Env)
# ==========================================
# 预测下一时刻
pred_y_next = model_env.predict(test[features])
mae = mean_absolute_error(test['target_y_next'] , pred_y_next)
rmse = np.sqrt(mean_squared_error(test['target_y_next'] , pred_y_next))

print(f"\n环境模型单步预测精度:")
print(f"MAE:  {mae:.4f} ℃")
print(f"RMSE: {rmse:.4f} ℃")

joblib.dump(model_env, ROOT / "models/env/boiler_env_model.pkl")
print("模型已保存为 'boiler_env_model.pkl'")

# ==========================================
# 5. 可视化
# ==========================================
start_idx = 0
end_idx = 1000  # 看前1000个步长
time_axis = range(start_idx , end_idx)

fig = plt.figure(figsize=(16 , 10))
gs = gridspec.GridSpec(2 , 1 , height_ratios=[1 , 1])

# --- Plot 1: 温度预测 ---
ax1 = fig.add_subplot(gs[0])
ax1.plot(time_axis , test['target_y_next'].iloc[start_idx:end_idx] , 'k-' , label='真实下一刻气温 (Real y_t+1)' ,
         linewidth=1.5)
ax1.plot(time_axis , pred_y_next[start_idx:end_idx] , 'r--' , label='环境预测气温 (Simulated y_t+1)' , linewidth=1.5)

# 画出误差分布
error = test['target_y_next'].iloc[start_idx:end_idx] - pred_y_next[start_idx:end_idx]
ax1_twin = ax1.twinx()
ax1_twin.fill_between(time_axis , error , 0 , color='blue' , alpha=0.1 , label='预测误差')
ax1_twin.set_ylim(-1 , 1)  # 误差通常很小，固定刻度方便看
ax1_twin.set_ylabel("误差 (℃)")

ax1.set_title(f"【环境模型验证】单步状态转移预测 (MAE: {mae:.3f})" , fontsize=14 , fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True , linestyle='--' , alpha=0.5)

# --- Plot 2: 输入条件 ---
ax2 = fig.add_subplot(gs[1] , sharex=ax1)
ax2.plot(time_axis , test['u_lag_0'].iloc[start_idx:end_idx] , color='#1f77b4' , label='挡板动作 u(t)')
ax2_twin = ax2.twinx()
ax2_twin.plot(time_axis , test['load_curr'].iloc[start_idx:end_idx] , color='#8c564b' , label='负荷 Load(t)' ,
              linestyle='-')
ax2_twin.plot(time_axis , test['coal_curr'].iloc[start_idx:end_idx] , color='black' , label='煤量 Coal(t)' ,
              linestyle=':' , alpha=0.6)

ax2.set_title("【环境输入】动作与扰动" , fontsize=14 , fontweight='bold')
ax2.set_ylabel("挡板 (%)")
ax2_twin.set_ylabel("负荷/煤")
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')

save_path = '基线预测.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"可视化图表已保存至: {save_path}")

plt.tight_layout()
plt.show()
