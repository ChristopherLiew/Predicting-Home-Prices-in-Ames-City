import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.regressor import AlphaSelection
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

## Load & Prep our Training & Testing data ##
train_set = pd.read_csv("train_clean.csv")
test_set = pd.read_csv("test_clean.csv")
X = train_set.drop(columns=["Saleprice"])
y = train_set[["Saleprice"]]

# Scale our Data with Robust Scaler to minimise outlier influence (Approx 4% of the data are significant outliers as measured by Cook's Distance)
rb_scaler = RobustScaler()
X_scaled = pd.DataFrame(rb_scaler.fit_transform(X), columns=X.columns)

std_scaler = StandardScaler()
X_standard = pd.DataFrame(std_scaler.fit_transform(X), columns=X.columns)

## Define CV Root Mean Square Error ##
def cv_rmse(estimator, X, y, cv=5):
    rmse = np.mean(np.sqrt(-cross_val_score(estimator, X, y, cv=cv, scoring="neg_mean_squared_error")))
    return rmse

## Regression Models ##
# Elastic Net Regressor
elastic_reg = ElasticNetCV(cv=5, max_iter=15000)
# Lasso Model for Comparison
lasso_reg = LassoCV(cv=5, alphas=[0.011], max_iter=15000) # Previously Optimised

## Model Evaluation & Hyperparameter Tuning ##
# CV Root Mean Squared Error on Training Set (Robust Scaled)
cv_rmse(lasso_reg, X_scaled, np.ravel(y)) # LASSO: 0.319
cv_rmse(elastic_reg, X_scaled, np.ravel(y)) # Elastic Net (ratio = 0.5): 0.317

# CV Root Mean Squared Error on Training Set (Standardised)
cv_rmse(lasso_reg, X_standard, np.ravel(y)) # LASSO: 0.2992
cv_rmse(elastic_reg, X_standard, np.ravel(y)) # Elastic Net (ratio = 0.5): 0.3012


# Alpha Selection
alphas = np.logspace(-10, 1, 400)
visualizer = AlphaSelection(elastic_reg)
visualizer.fit(X_scaled, y)
visualizer.show() # Optimal Alpha = 0.020

alphas = np.logspace(-10, 1, 400)
visualizer = AlphaSelection(elastic_reg)
visualizer.fit(X_standard, y)
visualizer.show() # Optimal Alpha = 0.020

# Search Algorithms to Further tune our Hyperparameters
# RandomizedSearchCV to narrow search space
rnd_params = {"l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              "alphas": [[0.1], [0.2], [0.3], [0.4], [0.5]],
              "max_iter": [15000],
              "normalize": [False]}
rnd_src = RandomizedSearchCV(elastic_reg, param_distributions=rnd_params, n_iter=100, scoring="neg_mean_squared_error", n_jobs=-1)
rnd_src.fit(X_scaled, np.ravel(y))
rnd_src.best_params_# {'normalize': False, 'max_iter': 15000, 'l1_ratio': 0.1, 'alphas': [0.1]}
rnd_src.best_score_ # -0.1117 OR

rnd_src.fit(X_standard, np.ravel(y))
rnd_src.best_params_# {'normalize': False, 'max_iter': 15000, 'l1_ratio': 0.1, 'alphas': [0.1]}
rnd_src.best_score_# -0.09379

# GridSearch
grd_params = {"l1_ratio": [0.1, 0.2, 0.3],
              "alphas": [[0.1], [0.2], [0.3]],
              "max_iter": [30000],
              "normalize": [False]}
grd_src = GridSearchCV(elastic_reg, param_grid=grd_params, scoring="neg_mean_squared_error", n_jobs=-1)
grd_src.fit(X_scaled, np.ravel(y))
grd_src.best_params_# {'alphas': [0.1], 'l1_ratio': 0.1, 'max_iter': 30000, 'normalize': False}
grd_src.best_score_ # -0.1117

grd_src.fit(X_standard, np.ravel(y))
grd_src.best_params_# {'alphas': [0.1], 'l1_ratio': 0, 'max_iter': 30000, 'normalize': False}
grd_src.best_score_# -0.09379

## Train & Evaluate Tuned Model ##
# Test out Alpha Selection from Yellowbrick Module
elastic_reg_yb = ElasticNetCV(alphas=[0.020], max_iter=15000)
cv_rmse(elastic_reg_yb, X_scaled, np.ravel(y)) # 0.3191 (Worse) - Scaled
cv_rmse(elastic_reg_yb, X_standard, np.ravel(y)) # 0.2990 (Better) - Standardised

# Test out GridSearchCV optimal hyperparameters
elastic_reg_tune = ElasticNetCV(alphas=[0.1], l1_ratio=0, max_iter=30000)
cv_rmse(elastic_reg_tune, X_scaled, y) # 0.3155 (Slight Improvement, but failed to fully converge) - Scaled
cv_rmse(elastic_reg_tune, X_standard, y) # 0.3189 - Standardised

## Ridge Regression ##
# Since l1_ratio = 0, test out Ridge Regression
ridge_reg = RidgeCV(scoring="neg_mean_squared_error") # From above
cv_rmse(ridge_reg, X_scaled, y) # 0.2995 (Better than LASSO & Elastic Net) - Scaled
cv_rmse(ridge_reg, X_standard, y) # 0.3298 (Worse than LASSO & Elastic Net) - Standardised

# Tune Ridge Model
grd_params_ridge = {'alphas': [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]]}
grd_src_ridge = GridSearchCV(ridge_reg, grd_params_ridge, scoring="neg_mean_squared_error")
grd_src_ridge.fit(X_scaled, y) # RunTimeError: Cannot clone object RidgeCV ... (Sklearn bug)
grd_src_ridge.best_params_# {'alphas': [0.9]}
grd_src_ridge.best_score_ # -0.09855

grd_src_ridge.fit(X_standard, y) # RunTimeError: Cannot clone object RidgeCV ... (Sklearn bug)
grd_src_ridge.best_params_# {'alphas': [0.9]}
grd_src_ridge.best_score_ # -0.1143

# Test our Tuned Ridge Model
ridge_reg_tune = RidgeCV(alphas=[0.9])
cv_rmse(ridge_reg_tune, X_scaled, y) # 0.3123 (Worse after tuning, but better than Lasso & Elastic Net)
cv_rmse(ridge_reg_tune, X_standard, y) # 0.3353 (Worse after tuning, but better than Lasso & Elastic Net)

## Best scores came from an untuned RidgeCV modeL, ridge_reg:  RMSE = 0.2995  AND LASSO: RMSE = 0.2992##
# LASSO performed better on Standardised Data since our numerical features are already Gaussian via powertransform, but
# RIDGE performed better on RobustScaled data, since RIDGE is more sensitive to outlier given L2 norm.

## Further Error Analysis (Learning Curves) ##
def plot_learning_curves(estimator, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_error, val_error = [], []
    for m in range(1, len(X_train)):
        estimator.fit(X_train[:m], y_train[:m])
        train_pred = estimator.predict(X_train[:m])
        val_pred = estimator.predict(X_val)
        train_pred_error = np.sqrt(mean_squared_error(y_train[:m], train_pred))
        val_pred_error = np.sqrt(mean_squared_error(y_val, val_pred))
        train_error.append(train_pred_error)
        val_error.append(val_pred_error)
    plt.plot(train_error, "b-", linewidth=3, label="Train")
    plt.plot(val_error, "r-", linewidth=3, label="Validation")
    plt.legend()
    plt.xlabel("No. of Training Examples")
    plt.ylabel("RMSE")
    plt.show()

plot_learning_curves(ridge_reg_tune, X_scaled, y)

# Relatively low RMSE, and narrow gap between Training & Validation RMSE, thus indicating possible issue of Underfitting (High Var.)
# Remedies:
# 1) Increase Training Data
# 2) Reduce Alpha Constraints
# 3) Non-Linear Methods (E.g. KernelRidge)
# 4) Increase useful features (Feature Engineering)

