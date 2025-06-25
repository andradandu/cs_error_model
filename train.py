import pymc as pm
import arviz as az
from utils import load_data
from model import build_model

train_obs, _, _, _ = load_data()
obs = train_obs["CSObs"].values
true = train_obs["TrueValue"].values

model = build_model(obs, true)

with model:
    trace = pm.sample(1000, tune=1000, target_accept=0.9)
    az.to_netcdf(trace, "trace.nc")
    print(az.summary(trace, var_names=["alpha", "beta", "sigma"]))