import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pymc as pm
import arviz as az

data = pd.read_csv("clean_data.csv", index_col=0, parse_dates=True)

# 1 - we split the data into training and test data
Y = data["target"].values
features_col_names = [c for c in data.columns if c != 'target']

X = data[features_col_names].values
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
dates_train = data.index[:split]
dates_test = data.index[split:]
print(f"Training set is from {dates_train[0].date()} to {dates_train[-1].date()}. Total days of the training set: {len(X_train)}")
print(f"Test set is from {dates_test[0].date()} to {dates_test[-1].date()}. Total days of the test set: {len(X_test)}")




scaler = StandardScaler()
X_train_s  = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)



# FREQUENTIST APPROACH - OLS Linear Regression
ols = LinearRegression()
ols.fit(X_train_s, Y_train)
y_pred_ols = ols.predict(X_test_s)
mae_ols = mean_absolute_error(Y_test, y_pred_ols)
rmse_ols = np.sqrt(mean_squared_error(Y_test, y_pred_ols))
print(f"Mean absolute error OLS: {mae_ols:.6f}")
print(f"Mean squared error OLS: {rmse_ols:.6f}")

#err = abs(y_pred_ols-Y_test)
#plt.plot(dates_test, err)
#plt.show()









# Bayesian linear regression with PyMC
print(f"First we must sample the posterior")



with pm.Model() as bayesian_model:
    X_data = pm.Data("X", X_train_s)
    Y_data = pm.Data("Y", Y_train)
    prior_intercept = pm.Normal("intercept", 0, 1)
    prior_coefficients = pm.Normal("coefficients", 0, 1, shape=X_train_s.shape[1])
    prior_sigma = pm.HalfNormal("sigma", 1)
    prior_mu = prior_intercept + pm.math.dot(X_data, prior_coefficients)
    likelihood = pm.Normal("obs", mu=prior_mu, sigma=prior_sigma, observed = Y_data)
    trace = pm.sample()
    
az.plot_trace(trace)
#plt.show()




with bayesian_model:
    pm.set_data({"X": X_test_s, "Y":np.zeros(len(X_test_s))})
    posterior_pred = pm.sample_posterior_predictive(trace, predictions = True)
#print(posterior_pred.predictions)
#print(posterior_pred.predictions["obs"].shape)

predicted_values = posterior_pred.predictions["obs"].values
predicted_values = predicted_values.reshape(-1, len(X_test_s))

y_pred_bayes_mean = predicted_values.mean(axis=0)
y_pred_bayes_lower = np.percentile(predicted_values, 5, axis=0)
y_pred_bayes_upper = np.percentile(predicted_values, 95, axis=0)

# print(f"Bayes predicted mean: {y_pred_bayes_mean}")
# print(f"Bayes predicted 5th percentile: {y_pred_bayes_lower}")
# print(f"Bayes predicted 95th percentile: {y_pred_bayes_upper}")



mae_bayes = mean_absolute_error(y_pred_bayes_mean, Y_test)
rmse_bayes = np.sqrt(mean_squared_error(y_pred_bayes_mean, Y_test))

print(f"Mean absolute error Bayes: {mae_bayes:.6f}")
print(f"Mean squared error Bayes: {rmse_bayes:.6f}")



fig, axes = plt.subplots(2, 2, figsize=(14,10))


axes[0,0].plot(dates_test, Y_test, label="Actual")
axes[0,0].plot(dates_test, y_pred_ols, label="OLS")
axes[0,0].plot(dates_test, y_pred_bayes_mean, label="Bayes")
axes[0,0].set_title("Predictions vs Actual")
axes[0,0].legend()




axes[0,1].plot(dates_test, Y_test, label="Actual")
axes[0,1].plot(dates_test, y_pred_bayes_mean, label="Bayes", color="orange")
axes[0,1].fill_between(dates_test, y_pred_bayes_lower, y_pred_bayes_upper, alpha=0.3, color="orange", label="90% credible interval")
axes[0,1].set_title("Bayesian Prediction with Uncertainty")
axes[0,1].legend()





coeff_means = trace.posterior["coefficients"].mean(dim=["chain", "draw"]).values
coeff_stds = trace.posterior["coefficients"].std(dim=["chain", "draw"]).values
axes[1,0].barh(features_col_names, coeff_means, xerr=coeff_stds, alpha=0.7)
axes[1,0].set_title("Bayesian Coefficient Estimates (+- 1 std)")
axes[1,0].axvline(x=0, color="black", linestyle="--", linewidth=0.5)
axes[1, 0].tick_params(axis="y", labelsize=7)
#axes[1,0].legend()






# Backtesting
signal = y_pred_bayes_mean > 0
strategy_returns = np.where(signal, Y_test, 0)
cumulative_strategy = np.cumsum(strategy_returns)
cumulative_buyhold = np.cumsum(Y_test)
axes[1,1].plot(dates_test, cumulative_strategy, label="Returns using Bayesian trading strategy")
axes[1,1].plot(dates_test, cumulative_buyhold, label="Buy and Hold")
axes[1,1].legend()
axes[1,1].set_title("Bayesian vs Buy and Hold")


plt.tight_layout()
plt.savefig("02_model_results.png", dpi=150)
plt.show()