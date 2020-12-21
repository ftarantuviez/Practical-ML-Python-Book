import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics

if __name__ == "__main__":
  cancer = load_breast_cancer()

  X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=.3, random_state=42)

  model = Sequential()
  model.add(Dense(15, input_dim=30, activation="relu"))
  model.add(Dense(15, activation="relu"))
  model.add(Dense(15, activation="relu"))
  model.add(Dense(1, activation="sigmoid"))

  model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

  model.fit(X_train, y_train, epochs=20, batch_size=50)

  predictions = model.predict_classes(X_test)
  print("Accuracy: ", metrics.accuracy_score(y_true=y_test, y_pred=predictions))
  print(metrics.classification_report(y_true=y_test, y_pred=predictions))