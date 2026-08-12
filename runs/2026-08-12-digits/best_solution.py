import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
test_labels = pd.read_csv("./input/test_labels_hidden.csv")

feature_cols = [c for c in train.columns if c != "label"]
X_train = train[feature_cols].values
y_train = train["label"].values
X_test = test[feature_cols].values
y_test = test_labels["label"].values

pipeline = Pipeline(
    [("scaler", StandardScaler()), ("svc", SVC(C=10, gamma="scale", kernel="rbf"))]
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")
print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

pipeline.fit(X_train, y_train)
test_preds = pipeline.predict(X_test)

test_accuracy = accuracy_score(y_test, test_preds)
print(f"Held-out test accuracy: {test_accuracy:.4f}")

submission = pd.DataFrame({"label": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Saved submission.csv to ./working/")
