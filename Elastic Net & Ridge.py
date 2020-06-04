import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.regressor import AlphaSelection
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
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
# CV Root Mean Squared Error on Training Set
cv_rmse(elastic_reg, X, np.ravel(y)) # Elastic Net (ratio = 0.5): 0.4488
cv_rmse(lasso_reg, X, np.ravel(y)) # LASSO: 0.3193

# Alpha Selection
alphas = np.logspace(-10, 1, 400)
visualizer = AlphaSelection(elastic_reg)
visualizer.fit(X, y)
visualizer.show() # Optimal Alpha = 0.322

# Search Algorithms to Further tune our Hyperparameters
elastic_reg_tune = ElasticNetCV()

# RandomizedSearchCV to narrow search space
rnd_params = {"l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              "alphas": [[0.1], [0.2], [0.3], [0.4], [0.5]],
              "max_iter": [15000],
              "normalize": [False]}
rnd_src = RandomizedSearchCV(elastic_reg_tune, param_distributions=rnd_params, n_iter=100, scoring="neg_mean_squared_error", n_jobs=-1)
rnd_src.fit(X, np.ravel(y)) # Convergence Issue
rnd_src.best_params_# {'normalize': False, 'max_iter': 15000, 'l1_ratio': 0.1, 'alphas': [0.1]}
rnd_src.best_score_ # -0.1093 OR MSE = 0.1093

# GridSearch
grd_params = {"l1_ratio": [0.1, 0.2, 0.3],
              "alphas": [[0.1], [0.2], [0.3]],
              "max_iter": [30000],
              "normalize": [False]}
grd_src = GridSearchCV(elastic_reg_tune, param_grid=grd_params, scoring="neg_mean_squared_error", n_jobs=-1)
grd_src.fit(X, np.ravel(y))
grd_src.best_params_# {'alphas': [0.1], 'l1_ratio': 0, 'max_iter': 30000, 'normalize': False}
grd_src.best_score_ # -0.0985 OR MSE = 0.0985

## Train & Evaluate Tuned Model ##
# Test out Alpha Selection from Yellowbrick Module
elastic_reg = ElasticNetCV(alphas=[0.322], max_iter=15000)
cv_rmse(elastic_reg, X, np.ravel(y)) # 0.4488, No improvement in RMSE

# Test out GridSearchCV optimal hyperparameters
elastic_reg_tune = ElasticNetCV(alphas=[0.1], l1_ratio=0, max_iter=15000)
cv_rmse(elastic_reg_tune, X, y) # Essentially Ridge Regression (l1_ratio = 0): 0.313 (Slight improvement over LASSO)

## Ridge Regression ##
# Since l1_ratio = 0, test out Ridge Regression
ridge_reg = RidgeCV(alphas=[0.1], scoring="neg_mean_squared_error") # From above
cv_rmse(ridge_reg, X, y) # 0.329 (worse than LASSO)
# Ridge due to its Cost Function avoids bouncing around, thus no ConvergenceError vs Lasso/ E-Net

# Tune Ridge Model
ridge_reg_tune = RidgeCV()
grd_params_ridge = {'alphas': [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]]}
grd_src_ridge = GridSearchCV(ridge_reg_tune, grd_params_ridge, scoring="neg_mean_squared_error")
grd_src_ridge.fit(X, y) # RunTimeError: Cannot clone object RidgeCV ... (Sklearn bug)
grd_src_ridge.best_params_# {'alphas': [0.9]}
grd_src_ridge.best_score_ # -0.09862

# Test our Tuned Ridge Model
ridge_reg_tune = RidgeCV(alphas=[0.9])
cv_rmse(ridge_reg_tune, X, y) # 0.3124 (Best score amongst the bunch)

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

plot_learning_curves(ridge_reg_tune, X, y)

# Relatively low RMSE, and narrow gap between Training & Validation RMSE, thus indicating possible issue of Underfitting (High Var.)
# Remedies:
# 1) Increase Training Data
# 2) Reduce Alpha Constraints
# 3) Non-Linear Methods (E.g. KernelRidge)
# 4) Increase useful features (Feature Engineering)

