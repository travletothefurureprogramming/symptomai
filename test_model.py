
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)



dataset = pd.read_csv("symptoms_dataset.csv")

X = dataset.drop("label", axis=1)
y = dataset["label"]




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)




cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

depths = [3, 5, 7, 9]

results = {}

print("\nCross Validation Results\n")

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    mean_score = scores.mean()

    results[depth] = mean_score

    print(f"Max Depth: {depth}")

    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: {score:.2%}")

    print(f"Mean Accuracy: {mean_score:.2%}\n")




best_depth = max(
    results,
    key=lambda depth: (results[depth], -depth)
)

print(f"Best Max Depth: {best_depth}")
print(f"Best CV Accuracy: {results[best_depth]:.2%}")




model = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=42
)

model.fit(X_train, y_train)




y_train_pred = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

print(f"\nTraining Accuracy: {train_accuracy:.2%}")



y_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Testing Accuracy: {test_accuracy:.2%}")




print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)




cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
