import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("helpdesk_tickets.csv")

print("ORIGINAL MISSING VALUES")
print("=" * 30)
print(df.isnull().sum())

# Handle missing values
df["Customer_Satisfaction"] = df["Customer_Satisfaction"].fillna(
    df["Customer_Satisfaction"].mean()
)

df["Response_Time"] = df["Response_Time"].fillna(
    df["Response_Time"].mean()
)

# Encode categorical attributes
categorical_columns = [
    "Department",
    "Issue_Type",
    "Device_Type",
    "Priority",
    "Technician",
    "Status",
    "Reopened"
]

df = pd.get_dummies(df, columns=categorical_columns)

# Standardize numerical attributes
numeric_columns = [
    "Response_Time",
    "Resolution_Time",
    "Customer_Satisfaction",
    "Number_of_Interactions"
]

scaler = StandardScaler()

df[numeric_columns] = scaler.fit_transform(
    df[numeric_columns]
)

df.to_csv(
    "processed_helpdesk_tickets.csv",
    index=False
)

print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")
print("Remaining missing values:",
      df.isnull().sum().sum())

print("Processed dataset saved as:")
print("processed_helpdesk_tickets.csv")

print("\nProcessed dataset shape:")
print(df.shape)
