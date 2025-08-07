# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Segmentation Data Setup
# MAGIC 
# MAGIC This notebook generates synthetic customer data for the segmentation demo. It creates raw tables that will be processed by the DLT pipeline.
# MAGIC 
# MAGIC **Output Tables:**
# MAGIC - `raw_customers`: Customer demographic data
# MAGIC - `raw_products`: Product catalog
# MAGIC - `raw_transactions`: Transaction history

# COMMAND ----------

%pip install databricks-sdk --upgrade
%pip install faker

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
from faker import Faker
from pyspark.sql.types import *
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import pipelines

w = WorkspaceClient()

# Get parameters from job
catalog_name = dbutils.widgets.get("catalog_name") if "catalog_name" in dbutils.widgets.getAll() else "dev_customer_segmentation"
schema_name = dbutils.widgets.get("schema_name") if "schema_name" in dbutils.widgets.getAll() else "segmentation"

# Configuration - Start small for testing
NUM_CUSTOMERS = 1000
NUM_TRANSACTIONS = 5000

# Set random seed for reproducible results
fake = Faker()
random.seed(42)
np.random.seed(42)

print(f"Using catalog: {catalog_name}, schema: {schema_name}")
print(f"Generating data for {NUM_CUSTOMERS:,} customers and ~{NUM_TRANSACTIONS:,} transactions")

# Setup catalog and schema
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"USE SCHEMA {schema_name}")
spark.sql("CREATE VOLUME IF NOT EXISTS customer_segmentation")

volume_name = 'customer_segmentation'
volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"

# Create an empty folder in a volume.
w.files.create_directory(f"{volume_path}/transactions")

# COMMAND ----------

# DBTITLE 1,Generate Customer Demographics
# 1. Customer Profile
def generate_customer_profile(n):

    income_brackets = ['Under 25K', '25-34K', '35-49K', '50-74K', '75-99K', '100K+']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'FL', 'OH', 'NC', 'GA']
    channels = ['Web', 'Mobile App', 'In-Store']

    data = []
    
    for i in range(n):

        location = fake.local_latlng(country_code='US', coords_only=True)
        index = random.randint(0, len(cities) - 1)

        data.append({
            "customer_id": f"C{str(i+1).zfill(4)}",
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "age": random.randint(15, 65),
            "gender": (random.choices(['M', 'F', 'X'], weights=[45, 45, 10]))[0],
            "income_bracket": (random.choices(income_brackets))[0],
            "city": cities[index],
            "state": states[index],
            "location": f"{float(location[0])} {float(location[1])}",
            "signup_date": fake.date_between(start_date='-5y', end_date='-1y'),
            "preferred_channel": random.choice(channels),
        })

    customer_profiles_df = pd.DataFrame(data)
    customer_profiles_sdf = spark.createDataFrame(customer_profiles_df)

    # Write to table
    customer_profiles_sdf.write \
        .format('delta') \
        .mode('overwrite') \
        .option('overwriteSchema', 'true') \
        .saveAsTable('raw_customer_profiles')

    print(f"Created raw_customer_profiles table with {len(customer_profiles_df):,} records")
    
    return customer_profiles_df
  
customer_profiles = generate_customer_profile(NUM_CUSTOMERS)

# COMMAND ----------

# DBTITLE 1,Generate Product Catalog
def generate_products():
    """Generate simple product catalog data"""
    
    # Simple product categories
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty', 'Food & Grocery']
    
    products_data = []
    
    # Generate exactly 100 products to keep it simple
    for product_id in range(1, 101):
        category = random.choice(categories)
        
        # Simple price ranges
        if category == 'Electronics':
            price = float(round(random.uniform(50, 500), 2))
        elif category == 'Books' or category == 'Beauty':
            price = float(round(random.uniform(15, 100), 2))
        else:
            price = float(round(random.uniform(5, 200), 2))
        
        cost = float(round(price * 0.6, 2))  # Simple 40% margin
        
        product = {
            'product_id': product_id,
            'product_name': f"{category} Item {product_id}",
            'category': category,
            'price': price,
            'cost': cost,
            'brand': f"Brand {random.randint(1, 10)}",
            'is_seasonal': random.choice([True, False])
        }
        products_data.append(product)
    
    # Define explicit schema for products
    products_schema = StructType([
        StructField("product_id", IntegerType(), True),
        StructField("product_name", StringType(), True),
        StructField("category", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("cost", DoubleType(), True),
        StructField("brand", StringType(), True),
        StructField("is_seasonal", BooleanType(), True)
    ])
    
    products_sdf = spark.createDataFrame(products_data, products_schema)
    
    # Write to table
    products_sdf.write \
        .format('delta') \
        .mode('overwrite') \
        .option('overwriteSchema', 'true') \
        .saveAsTable('raw_products')
    
    print(f"Created raw_products table with {products_sdf.count():,} records")
    return products_sdf

products = generate_products()

# COMMAND ----------

# 2. Transaction Summary
def generate_detailed_transactions(profiles, products):
    transactions = []

    channels = ['Web', 'Mobile App', 'In-Store']
    
    # Get products list in a simpler way
    products_list = []
    for row in products.collect():
        products_list.append((row.product_id, row.category, row.price, row.cost))

    for _, row in profiles.iterrows():
        num_txns = random.randint(1, 15) # Generates between 1 & 15 transactions for each customer_id 
        if num_txns == 0:
            continue
        
        for i in range(num_txns):
          
          number_of_items = random.randint(1, 4)
          product_idx = random.randint(0, len(products_list)-1)
          product_selection = products_list[product_idx]

          product_id = product_selection[0]
          category = product_selection[1]
          quantity = number_of_items
          per_unit_cost = product_selection[3]
          per_unit_msrp = product_selection[2]
          total_cost = product_selection[3] * number_of_items            

          # Simple discount logic
          discount_amount = 0.0
          if random.random() < 0.15:  # 10% chance of discount
              discount_amount = float(round(per_unit_msrp * 0.15, 2))

          pre_discount_amount = float(round(per_unit_msrp * number_of_items, 2))
          total_amount = float(round((per_unit_msrp - discount_amount) * number_of_items, 2))

          txn_date = pd.to_datetime("2025-07-30") - pd.to_timedelta(np.random.randint(15, 365), unit='D')

          transactions.append({
              "transaction_id": np.random.randint(100000000, 500000000),
              "customer_id": row['customer_id'],
              "transaction_date": txn_date,
              "product_id": product_id,
              "quantity": quantity,
              "category": category,
              "per_unit_cost": per_unit_cost,
              "per_unit_msrp": per_unit_msrp,
              "total_cost": total_cost,
              "pre_discount_amount": pre_discount_amount,
              "discount_amount": discount_amount,
              "total_amount": total_amount,
              "channel": random.choice(channels),
          })

    return pd.DataFrame(transactions)

detailed_txns = generate_detailed_transactions(customer_profiles, products)
detailed_txns.to_csv(f"{volume_path}/transactions/raw_transactions.csv", index=False)

# COMMAND ----------

# DBTITLE 1,Verify Data Generation
print("=== Data Generation Summary ===")
print(f"Catalog: {catalog_name}")
print(f"Schema: {schema_name}")
print()

# Check table counts
customers_count = spark.table("raw_customer_profiles").count()
products_count = spark.table("raw_products").count()
transactions_count = len(detailed_txns)

print(f"✅ raw_customers: {customers_count:,} records")
print(f"✅ raw_products: {products_count:,} records") 
print(f"✅ raw_transactions: {transactions_count:,} records")
print()
print("Raw data is ready for DLT pipeline processing!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Generation Complete ✅
# MAGIC 
# MAGIC Raw synthetic data has been successfully generated and saved to Unity Catalog tables and volumes:
# MAGIC 
# MAGIC - **raw_customers**: Customer demographics and profiles
# MAGIC - **raw_products**: Product catalog with pricing and categories
# MAGIC - **raw_transactions**: Realistic transaction history with purchasing patterns
# MAGIC 
# MAGIC The DLT pipeline can now process this raw data to create customer segments.