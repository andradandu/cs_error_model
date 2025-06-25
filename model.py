import pymc as pm
import numpy as np

def build_model(obs, true_values, n_error_types=5):
    n = len(obs)
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=1.0, sigma=2.0, shape=n_error_types)
        beta = pm.Normal("beta", mu=0.0, sigma=2.0, shape=n_error_types)
        sigma = pm.Exponential("sigma", lam=1.0, shape=n_error_types)

        eps = pm.Categorical("eps", p=np.ones(n_error_types)/n_error_types, shape=n)
        theta = pm.MutableData("theta", true_values)

        mu = alpha[eps] * theta + beta[eps]
        pm.Normal("obs", mu=mu, sigma=sigma[eps], observed=obs)

    return model
