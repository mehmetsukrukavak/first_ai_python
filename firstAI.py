import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score

df = pd.read_csv("glass.csv")

df.head()

print(df.shape)

print(df["Type"].value_counts())

X = df.drop("Type", axis=1)
y = df["Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

model = RandomForestClassifier(random_state=9)
model.fit(X_train, y_train)

cam_pred = model.predict(X_test)

print(accuracy_score(y_test, cam_pred))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

print(classification_report(y_test, cam_pred))