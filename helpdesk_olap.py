import pandas as pd

df = pd.read_csv("helpdesk_tickets.csv")

print("IT HELPDESK OLAP ANALYSIS")
print("=" * 40)

# Roll-up: Department
print("\n1. Tickets by Department:")
print(
    df.groupby("Department")
    .size()
    .sort_values(ascending=False)
)

# Roll-up: Issue Type
print("\n2. Tickets by Issue Type:")
print(
    df.groupby("Issue_Type")
    .size()
    .sort_values(ascending=False)
)

# Roll-up: Priority
print("\n3. Tickets by Priority:")
print(
    df.groupby("Priority")
    .size()
    .sort_values(ascending=False)
)

# Average resolution time
print("\n4. Average Resolution Time by Department:")
print(
    df.groupby("Department")["Resolution_Time"]
    .mean()
    .round(2)
)

# Department vs Issue Type
print("\n5. Department vs Issue Type:")

pivot = pd.pivot_table(
    df,
    index="Department",
    columns="Issue_Type",
    values="Ticket_ID",
    aggfunc="count",
    fill_value=0
)

print(pivot)

pivot.to_csv("helpdesk_olap_summary.csv")

print("\nOLAP analysis completed successfully!")
print("Summary saved as: helpdesk_olap_summary.csv")
