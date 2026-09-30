import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("helpdesk_tickets.csv")

# Select categorical attributes
data = df[
    ["Department", "Issue_Type", "Device_Type", "Priority"]
]

# One-hot encoding
basket = pd.get_dummies(data)

# Frequent itemsets
frequent_itemsets = apriori(
    basket,
    min_support=0.05,
    use_colnames=True
)

# Association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.60
)

rules = rules.sort_values(
    by="lift",
    ascending=False
)

print("APRIORI ASSOCIATION RULE MINING")
print("=" * 40)

print("\nFrequent Itemsets:")
print(frequent_itemsets.head(10))

print("\nTop Association Rules:")

for _, row in rules.head(10).iterrows():

    print(
        "Rule:",
        list(row["antecedents"]),
        "->",
        list(row["consequents"])
    )

    print(
        "Support:",
        round(row["support"], 3)
    )

    print(
        "Confidence:",
        round(row["confidence"], 3)
    )

    print(
        "Lift:",
        round(row["lift"], 3)
    )

    print("-" * 40)

rules.to_csv(
    "helpdesk_apriori_rules.csv",
    index=False
)

print("\nApriori analysis completed successfully!")
print("Rules saved as: helpdesk_apriori_rules.csv")
