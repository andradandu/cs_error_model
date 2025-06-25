import arviz as az

# 載入訓練後的模型 trace
idata = az.from_netcdf("trace.nc")

# 顯示後驗樣本統計
print(az.summary(idata, var_names=["alpha", "beta", "sigma"]))

# 顯示錯誤型別分佈（可選）
print("Posterior samples of error type assignments:")
print(idata.posterior["eps"].values)