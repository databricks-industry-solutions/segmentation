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
# def generate_customers():
#     """Generate simple customer demographic data"""
    
#     # Simple options
#     age_brackets = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
#     income_brackets = ['Under 25K', '25-34K', '35-49K', '50-74K', '75-99K', '100K+']
#     channels = ['Online', 'Mobile', 'Store']
#     cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
#     states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'FL', 'OH', 'NC', 'GA']
    
#     customers_data = []
    
#     for customer_id in range(1, NUM_CUSTOMERS + 1):
#         # Simple random selections
#         age_bracket = random.choice(age_brackets)
#         income_bracket = random.choice(income_brackets)
#         household_size = random.randint(1, 5)
#         city = random.choice(cities)
#         state = random.choice(states)
#         preferred_channel = random.choice(channels)
        
#         # Simple date generation
#         days_ago = random.randint(180, 1095)  # 6 months to 3 years ago
#         signup_date = (datetime.now() - timedelta(days=days_ago)).date()
        
#         customer = {
#             'customer_id': customer_id,
#             'age_bracket': age_bracket,
#             'income_bracket': income_bracket,
#             'household_size': household_size,
#             'city': city,
#             'state': state,
#             'signup_date': signup_date,
#             'preferred_channel': preferred_channel
#         }
#         customers_data.append(customer)
    
#     # Define explicit schema for customers
#     customers_schema = StructType([
#         StructField("customer_id", IntegerType(), True),
#         StructField("age_bracket", StringType(), True),
#         StructField("income_bracket", StringType(), True),
#         StructField("household_size", IntegerType(), True),
#         StructField("city", StringType(), True),
#         StructField("state", StringType(), True),
#         StructField("signup_date", DateType(), True),
#         StructField("preferred_channel", StringType(), True)
#     ])
    
#     # Convert to Spark DataFrame with explicit schema
#     customers_df = spark.createDataFrame(customers_data, customers_schema)
    
#     # Write to table
#     customers_df.write \
#         .format('delta') \
#         .mode('overwrite') \
#         .option('overwriteSchema', 'true') \
#         .saveAsTable('raw_customers')
    
#     print(f"Created raw_customers table with {customers_df.count():,} records")
#     return customers_df

# customers_df = generate_customers()

# COMMAND ----------

# DBTITLE 1,Generate Customer Demographics
# 1. Customer Profile
def generate_customer_profile(n):

    income_brackets = ['Under 25K', '25-34K', '35-49K', '50-74K', '75-99K', '100K+']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'FL', 'OH', 'NC', 'GA']

    data = []
    
    for i in range(n):

        location = fake.local_latlng(country_code='US', coords_only=True)
        index = random.randint(0, len(cities) - 1)

        data.append({
            "CustomerID": f"C{str(i+1).zfill(4)}",
            "FirstName": fake.first_name(),
            "LastName": fake.last_name(),
            "Age": random.randint(15, 65),
            "Gender": (random.choices(['M', 'F', 'X'], weights=[40, 40, 10]))[0],
            "IncomeBracket": (random.choices(income_brackets))[0],
            "City": cities[index],
            "State": states[index],
            "Location": f"{float(location[0])} {float(location[1])}",
            "SignupDate": fake.date_between(start_date='-5y', end_date='-1y')
        })

    customer_profiles_df = pd.DataFrame(data)
    customer_profiles_sdf = spark.createDataFrame(customer_profiles_df)

    # Write to table
    customer_profiles_sdf.write \
        .format('delta') \
        .mode('overwrite') \
        .option('overwriteSchema', 'true') \
        .saveAsTable('raw_customer_profiles')

    #print(f"Created raw_customer_profiles table with {customer_profiles_df.count():,} records")
    
    return customer_profiles_df
  
customer_profiles_df = generate_customer_profile(NUM_CUSTOMERS)

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
        elif category == 'Clothing':
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
    
    products_df = spark.createDataFrame(products_data, products_schema)
    
    # Write to table
    products_df.write \
        .format('delta') \
        .mode('overwrite') \
        .option('overwriteSchema', 'true') \
        .saveAsTable('raw_products')
    
    print(f"Created raw_products table with {products_df.count():,} records")
    return products_df

products_df = generate_products()

# COMMAND ----------

# DBTITLE 1,Generate Transaction Data
def generate_transactions(products_df):
    """Generate simple, reliable transaction data"""
    
    # Get products list in a simpler way
    products_list = []
    for row in products_df.collect():
        products_list.append((row.product_id, row.category, row.price))
    
    transactions_data = []
    transaction_id = 1
    
    # Simple transaction generation - every customer gets 3-7 transactions
    for customer_id in range(1, NUM_CUSTOMERS + 1):
        num_transactions = random.randint(3, 7)
        
        for _ in range(num_transactions):
            # Simple random date in last 2 years
            days_ago = random.randint(1, 730)
            transaction_date = datetime.now().date() - timedelta(days=days_ago)
            
            # Simple product selection - just pick random products
            num_items = random.randint(1, 3)
            selected_products = random.sample(products_list, min(num_items, len(products_list)))
            
            for prod_id, category, price in selected_products:
                quantity = random.randint(1, 2)
                
                # Simple discount logic
                discount_amount = 0.0
                if random.random() < 0.1:  # 10% chance of discount
                    discount_amount = float(round(price * 0.1, 2))
                
                total_amount = float(round((price - discount_amount) * quantity, 2))
                
                transaction = {
                    'transaction_id': transaction_id,
                    'customer_id': customer_id,
                    'product_id': prod_id,
                    'transaction_date': transaction_date,
                    'quantity': quantity,
                    'unit_price': price,
                    'discount_amount': discount_amount * quantity,
                    'total_amount': total_amount,
                    'category': category
                }
                transactions_data.append(transaction)
                transaction_id += 1
    
    # Define explicit schema for transactions
    transactions_schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("customer_id", IntegerType(), True),
        StructField("product_id", IntegerType(), True),
        StructField("transaction_date", DateType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("unit_price", DoubleType(), True),
        StructField("discount_amount", DoubleType(), True),
        StructField("total_amount", DoubleType(), True),
        StructField("category", StringType(), True)
    ])
    
    # Convert to Spark DataFrame
    transactions_df = spark.createDataFrame(transactions_data, transactions_schema)
    
    # Write to table
    transactions_df.write \
        .format('delta') \
        .mode('overwrite') \
        .option('overwriteSchema', 'true') \
        .saveAsTable('raw_transactions')
    
    print(f"Created raw_transactions table with {transactions_df.count():,} records")
    return transactions_df

transactions_df = generate_transactions(products_df)

# COMMAND ----------

# DBTITLE 1,Verify Data Generation
print("=== Data Generation Summary ===")
print(f"Catalog: {catalog_name}")
print(f"Schema: {schema_name}")
print()

# Check table counts
customers_count = spark.table("raw_customers").count()
products_count = spark.table("raw_products").count()
transactions_count = spark.table("raw_transactions").count()

print(f"✅ raw_customers: {customers_count:,} records")
print(f"✅ raw_products: {products_count:,} records") 
print(f"✅ raw_transactions: {transactions_count:,} records")
print()
print("Raw data is ready for DLT pipeline processing!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Generation Complete ✅
# MAGIC 
# MAGIC Raw synthetic data has been successfully generated and saved to Unity Catalog tables:
# MAGIC 
# MAGIC - **raw_customers**: Customer demographics and profiles
# MAGIC - **raw_products**: Product catalog with pricing and categories
# MAGIC - **raw_transactions**: Realistic transaction history with purchasing patterns
# MAGIC 
# MAGIC The DLT pipeline can now process this raw data to create customer segments.