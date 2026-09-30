import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("helpdesk_tickets.csv")

# Handle missing values
df["Response_Time"] = df["Response_Time"].fillna(
    df["Response_Time"].mean()
)

df["Customer_Satisfaction"] = df["Customer_Satisfaction"].fillna(
    df["Customer_Satisfaction"].mean()
)

features = [
    "Response_Time",
    "Resolution_Time",
    "Customer_Satisfaction",
    "Number_of_Interactions"
]

X = df[features]

# Standardization
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# K-Means clustering
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X_scaled)

# Silhouette score
score = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print("IT HELPDESK K-MEANS CLUSTERING")
print("=" * 40)

print("\nCluster Counts:")
print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print("\nCluster Characteristics:")

print(
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

print(
    "\nSilhouette Score:",
    round(score, 3)
)

# Visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    df["Response_Time"],
    df["Resolution_Time"],
    c=df["Cluster"]
)

plt.xlabel("Response Time")
plt.ylabel("Resolution Time")

plt.title(
    "IT Helpdesk Tickets - K-Means Clustering"
)

plt.show()

print("\nClustering completed successfully!")
