print("🚨 檢查：train.py 已經開始執行", flush=True)
import pymc as pm
print("✅ 匯入 pymc 完成", flush=True)
import arviz as az
from utils import load_data
print("📥 載入訓練資料中...", flush=True)
from model import build_model


print("Hello world")
# 載入資料
print("📥 載入訓練資料中...")
train_obs, _, _, _ = load_data()
obs = train_obs["CSObs"].values
true = train_obs["TrueValue"].values

# 建立模型
print("🔧 建立模型結構...")
model = build_model(obs, true)

# 開始訓練
print("🏋️ 開始訓練模型（這可能需要幾分鐘）...")
with model:
    trace = pm.sample(1000, tune=1000, target_accept=0.9)

# 儲存並顯示結果
az.to_netcdf(trace, "trace.nc")
print("✅ 模型訓練完成，結果已儲存至 trace.nc")

# 印出參數後驗摘要
summary = az.summary(trace, var_names=["alpha", "beta", "sigma"])
print("📊 模型參數摘要：")
print(summary)