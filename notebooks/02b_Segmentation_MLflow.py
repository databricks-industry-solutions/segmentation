# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Segmentation with MLflow
# MAGIC
# MAGIC This notebook uses unsupervised machine learning with MLflow to segment customers into different groups based on RFM metrics, and uses MLflow for tracking the experiments.
# MAGIC
# MAGIC _Note_: You will need to run 02a_Segmentation_Lakeflow before this notebook, regardless if you want to use Lakeflow or MLflow for segmentation.

# COMMAND ----------

# MAGIC %pip install --upgrade threadpoolctl==3.1.0
# MAGIC %pip install --upgrade mlflow==3.2.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# 1. Imports
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

user = w.current_user.me().user_name

# COMMAND ----------

# Get parameters from job
catalog_name = dbutils.widgets.get("catalog_name") if "catalog_name" in dbutils.widgets.getAll() else "dev_customer_segmentation"
schema_name = dbutils.widgets.get("schema_name") if "schema_name" in dbutils.widgets.getAll() else "segmentation"


# COMMAND ----------

# 2. Load data into pandas DataFrame
sdf = spark.sql(f"SELECT * FROM {catalog_name}.{schema_name}.rfm_analysis")
df = sdf.toPandas()
df.head()

# COMMAND ----------

# Create new DataFrame with required columns, then rename the columns
rfm_df = df[['customer_id', 'frequency_score', 'recency_score', 'monetary_score']]

rfm_df = rfm_df.rename(
  columns={
    'frequency_score': 'frequency',
    'recency_score': 'recency',
    'monetary_score': 'monetary'
  }
)

# COMMAND ----------

# Create a new DataFrame with only the RFM columns
X = rfm_df[['recency', 'frequency', 'monetary']]

# COMMAND ----------

# Calculate inertia (sum of squared distances) for different values of k
with mlflow.start_run():
    inertia = []
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, n_init= 10, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 6),dpi=150)
    plt.plot(range(2, 8), inertia, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve for K-means Clustering')
    plt.grid(True)
    plt.show()

# COMMAND ----------

# Perform K-means clustering with best K
# In this case we can select 5 clusters, but you could pick 3, 4, 6, 7, etc...
best_kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
rfm_df['Cluster'] = best_kmeans.fit_predict(X)

# COMMAND ----------

rfm_df.head()

# COMMAND ----------

# Group by cluster and calculate mean values
cluster_summary = rfm_df.groupby('Cluster').agg({
    'recency': 'median',
    'frequency': 'median',
    'monetary': 'median'
}).reset_index()

# COMMAND ----------

# Let's plot the different clusters we have

colors = ['#3498db', '#2ecc71', '#f39c12','#C9B1BD']

# Plot the average RFM scores for each cluster
plt.figure(figsize=(10, 8),dpi=150)

# Plot Avg Recency
plt.subplot(3, 1, 1)
bars = plt.bar(cluster_summary.index, cluster_summary['recency'], color=colors)
plt.xlabel('Cluster')
plt.ylabel('Avg Recency')
plt.title('Average Recency for Each Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bars, cluster_summary.index, title='Clusters')

# Plot Avg Frequency
plt.subplot(3, 1, 2)
bars = plt.bar(cluster_summary.index, cluster_summary['frequency'], color=colors)
plt.xlabel('Cluster')
plt.ylabel('Avg Frequency')
plt.title('Average Frequency for Each Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bars, cluster_summary.index, title='Clusters')

# Plot Avg Monetary
plt.subplot(3, 1, 3)
bars = plt.bar(cluster_summary.index, cluster_summary['monetary'], color=colors)
plt.xlabel('Cluster')
plt.ylabel('Avg Monetary')
plt.title('Average Monetary Value for Each Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bars, cluster_summary.index, title='Clusters')

plt.tight_layout()
plt.show()

# COMMAND ----------

# Mapping of integer to category
cluster_map = {
    0: 'Regular', 
    1: 'Champions',
    2: 'New', 
    3: 'Churned',  
    4: 'High Value'
}

# Apply the mapping
rfm_df['rfm_cluster'] = rfm_df['Cluster'].map(cluster_map)

# COMMAND ----------

# Look at the output data
display(rfm_df)

# COMMAND ----------

# Retrieve the experiment ID, then view MLflow metadata

expId = mlflow.get_experiment_by_name(f"/Users/{user}/.bundle/customer-segmentation/lg/files/notebooks/02b_Segmentation_MLflow").experiment_id
experiment = mlflow.get_experiment(expId)

print(f"Experiment_id: {expId}")
print(f"Name: {experiment.name}")
print(f"Artifact Location: {experiment.artifact_location}")
print(f"Tags: {experiment.tags}")
print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
print(f"Creation timestamp: {experiment.creation_time}")

# COMMAND ----------

# Build a DataFrame for all MLflow runs
df = mlflow.search_runs([expId])
display(df)

# COMMAND ----------

