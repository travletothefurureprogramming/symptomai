from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import joblib
import os

os.makedirs("models", exist_ok=True)

dataset = pd.read_csv("symptoms_dataset.csv")

X = dataset.drop("label", axis=1)
y = dataset["label"]

model = DecisionTreeClassifier(max_depth=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train,y_train)

y_train_pred = model.predict(X_train)

train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"Training Accuracy: {train_accuracy}")


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Testing Accuracy: {accuracy}')


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.2f}")

print(f"Mean Accuracy: {scores.mean():.2f}")

joblib.dump(model, "models/model.pkl")