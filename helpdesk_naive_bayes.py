import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("NAIVE BAYES CLASSIFICATION")
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

print("\nNaive Bayes classification completed successfully!")
