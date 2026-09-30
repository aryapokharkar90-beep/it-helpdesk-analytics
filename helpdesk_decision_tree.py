import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("helpdesk_tickets.csv")

# Handle missing values
df["Response_Time"] = df["Response_Time"].fillna(
    df["Response_Time"].mean()
)

df["Customer_Satisfaction"] = df["Customer_Satisfaction"].fillna(
    df["Customer_Satisfaction"].mean()
)

features = [
    "Department",
    "Issue_Type",
    "Device_Type",
    "Technician",
    "Response_Time",
    "Resolution_Time",
    "Status",
    "Reopened",
    "Customer_Satisfaction",
    "Number_of_Interactions"
]

X = df[features]
y = df["Priority"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("DECISION TREE CLASSIFICATION")
print("=" * 40)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9
)

plt.title("IT Helpdesk Priority - Decision Tree")
plt.tight_layout()
plt.show()
