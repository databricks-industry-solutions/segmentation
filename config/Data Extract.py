# Databricks notebook source
# MAGIC %md The purpose of this notebook is to download and set up the data we will use for the solution accelerator. Before running this notebook, make sure you have entered your own credentials for Kaggle.

# COMMAND ----------

# MAGIC %pip install kaggle

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC Set Kaggle credential configuration values in the block below: You can set up a [secret scope](https://docs.databricks.com/security/secrets/secret-scopes.html) to manage credentials used in notebooks. For the block below, we have manually set up the `solution-accelerator-cicd` secret scope and saved our credentials there for internal testing purposes.

# COMMAND ----------

import os
# os.environ['kaggle_username'] = 'YOUR KAGGLE USERNAME HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_username'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_username")

# os.environ['kaggle_key'] = 'YOUR KAGGLE KEY HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_key'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_key")

# COMMAND ----------

# MAGIC %md Download the data from Kaggle using the credentials set above:

# COMMAND ----------

# MAGIC %sh 
# MAGIC cd /databricks/driver
# MAGIC export KAGGLE_USERNAME=$kaggle_username
# MAGIC export KAGGLE_KEY=$kaggle_key
# MAGIC kaggle datasets download -d frtgnn/dunnhumby-the-complete-journey
# MAGIC unzip -o dunnhumby-the-complete-journey.zip

# COMMAND ----------

# MAGIC %md Move the downloaded data to the folder used throughout the accelerator:

# COMMAND ----------

# MAGIC %run "./Unity Catalog"

# COMMAND ----------

spark.sql(f'USE CATALOG {CATALOG}');
spark.sql(f'USE SCHEMA {SCHEMA}');

# COMMAND ----------

spark.sql(f'CREATE VOLUME IF NOT EXISTS {VOLUME_NAME}');

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/campaign_desc.csv", f"{VOLUME_PATH}/bronze/campaign_desc.csv")
dbutils.fs.mv("file:/databricks/driver/campaign_table.csv", f"{VOLUME_PATH}/bronze/campaign_table.csv")
dbutils.fs.mv("file:/databricks/driver/causal_data.csv", f"{VOLUME_PATH}/bronze/causal_data.csv")
dbutils.fs.mv("file:/databricks/driver/coupon.csv", f"{VOLUME_PATH}/bronze/coupon.csv")
dbutils.fs.mv("file:/databricks/driver/coupon_redempt.csv", f"{VOLUME_PATH}/bronze/coupon_redempt.csv")
dbutils.fs.mv("file:/databricks/driver/hh_demographic.csv", f"{VOLUME_PATH}/bronze/hh_demographic.csv")
dbutils.fs.mv("file:/databricks/driver/product.csv", f"{VOLUME_PATH}/bronze/product.csv")
dbutils.fs.mv("file:/databricks/driver/transaction_data.csv", f"{VOLUME_PATH}/bronze/transaction_data.csv")
