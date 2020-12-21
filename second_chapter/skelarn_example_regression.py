import pandas as pd
from sklearn import datasets
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
  diabetes = datasets.load_diabetes()
  y = diabetes.target
  X = diabetes.data

  feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

  X_train = X[:310]
  y_train = y[:310]

  X_test = X[310:]
  y_test = y[310:]

  lasso = Lasso(random_state=0)
  alphas = np.logspace(-4, -.5, 30)

  estimator = GridSearchCV(lasso, dict(alpha=alphas))
  estimator.fit(X_train, y_train)

  prediction = estimator.predict(X_test)
  