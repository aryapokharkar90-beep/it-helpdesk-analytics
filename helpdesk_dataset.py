import pandas as pd
import numpy as np

np.random.seed(42)

n = 6000

departments = ["CSE", "IT", "ENTC", "Mechanical", "Admin", "Accounts"]
issues = ["Network", "Software", "Hardware", "Login", "Printer", "Security"]
technicians = ["Tech_A", "Tech_B", "Tech_C", "Tech_D", "Tech_E"]

department = np.random.choice(departments, n)
issue = np.random.choice(issues, n)

# Device depends on issue type
device = []

for i in range(n):
    if issue[i] == "Printer":
        device.append("Printer")
    elif issue[i] == "Network":
        device.append(np.random.choice(["Router", "Server", "Laptop"],
                                       p=[0.50, 0.30, 0.20]))
    elif issue[i] == "Security":
        device.append(np.random.choice(["Server", "Laptop", "Desktop"],
                                       p=[0.50, 0.30, 0.20]))
    elif issue[i] == "Hardware":
        device.append(np.random.choice(["Laptop", "Desktop"],
                                       p=[0.60, 0.40]))
    else:
        device.append(np.random.choice(["Laptop", "Desktop", "Server"]))

# Priority depends strongly on issue type
priority = []

for i in range(n):
    if issue[i] == "Security":
        priority.append(np.random.choice(
            ["Low", "Medium", "High"], p=[0.05, 0.20, 0.75]
        ))
    elif issue[i] == "Network":
        priority.append(np.random.choice(
            ["Low", "Medium", "High"], p=[0.10, 0.70, 0.20]
        ))
    elif issue[i] == "Printer":
        priority.append(np.random.choice(
            ["Low", "Medium", "High"], p=[0.70, 0.28, 0.02]
        ))
    elif issue[i] == "Hardware":
        priority.append(np.random.choice(
            ["Low", "Medium", "High"], p=[0.20, 0.60, 0.20]
        ))
    else:
        priority.append(np.random.choice(
            ["Low", "Medium", "High"], p=[0.20, 0.60, 0.20]
        ))

response_time = np.random.randint(5, 121, n)

reopened = np.random.choice(
    ["Yes", "No"], n, p=[0.15, 0.85]
)

interactions = np.random.randint(1, 11, n)

# Issue-related effect on resolution time
issue_effect = {
    "Network": 30,
    "Software": 20,
    "Hardware": 40,
    "Login": 15,
    "Printer": 10,
    "Security": 55
}

priority_effect = {
    "Low": 0,
    "Medium": 25,
    "High": 60
}

# Resolution time has meaningful relationships with other variables
resolution_time = []

for i in range(n):
    value = (
        25
        + 0.75 * response_time[i]
        + 9 * interactions[i]
        + priority_effect[priority[i]]
        + issue_effect[issue[i]]
        + (25 if reopened[i] == "Yes" else 0)
        + np.random.normal(0, 12)
    )

    resolution_time.append(max(15, round(value)))

resolution_time = np.array(resolution_time)

status = np.random.choice(
    ["Resolved", "Pending", "Escalated"],
    n,
    p=[0.75, 0.15, 0.10]
)

satisfaction = np.random.randint(1, 6, n)

data = {
    "Ticket_ID": [f"T{i:04d}" for i in range(1, n + 1)],
    "Date": pd.date_range("2025-01-01", periods=n, freq="8h"),
    "Department": department,
    "Issue_Type": issue,
    "Device_Type": device,
    "Priority": priority,
    "Technician": np.random.choice(technicians, n),
    "Response_Time": response_time,
    "Resolution_Time": resolution_time,
    "Status": status,
    "Reopened": reopened,
    "Customer_Satisfaction": satisfaction,
    "Number_of_Interactions": interactions
}

df = pd.DataFrame(data)

# Add missing values for preprocessing demonstration
df.loc[
    np.random.choice(n, 60, replace=False),
    "Customer_Satisfaction"
] = np.nan

df.loc[
    np.random.choice(n, 40, replace=False),
    "Response_Time"
] = np.nan

df.to_csv("helpdesk_tickets.csv", index=False)

print("IT Helpdesk dataset created successfully!")
print("Number of records:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nDataset saved as: helpdesk_tickets.csv")
