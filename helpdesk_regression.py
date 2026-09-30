import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
    "Priority",
    "Response_Time",
    "Status",
    "Reopened",
    "Customer_Satisfaction",
    "Number_of_Interactions"
]

X = df[features]
y = df["Resolution_Time"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("LINEAR REGRESSION - HELP DESK")
print("=" * 40)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Mean Squared Error:",
    round(mse, 2)
)

print(
    "R2 Score:",
    round(r2, 4)
)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Resolution Time")
plt.ylabel("Predicted Resolution Time")

plt.title(
    "Actual vs Predicted Resolution Time"
)

plt.show()

print("\nRegression completed successfully!")
