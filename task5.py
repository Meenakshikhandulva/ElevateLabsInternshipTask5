# task5_decision_tree_random_forest.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import graphviz
from sklearn import tree

# 1. Load dataset
df = pd.read_csv("heart.csv")

print("First 5 rows:")
print(df.head())

# Separate features & target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train Decision Tree
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

# Predict
y_pred_dt = dt_clf.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_clf, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.show()

# 3. Analyze Overfitting - Control Tree Depth
dt_clf_depth = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_clf_depth.fit(X_train, y_train)
y_pred_depth = dt_clf_depth.predict(X_test)

print("\nDecision Tree (max_depth=4) Accuracy:", accuracy_score(y_test, y_pred_depth))

# 4. Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# 5. Feature Importance
importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh", figsize=(10, 6))
plt.title("Feature Importances - Random Forest")
plt.show()

# 6. Cross-validation
cv_scores_dt = cross_val_score(dt_clf, X, y, cv=5)
cv_scores_rf = cross_val_score(rf_clf, X, y, cv=5)

print("\nCross-validation (Decision Tree):", cv_scores_dt.mean())
print("Cross-validation (Random Forest):", cv_scores_rf.mean())
