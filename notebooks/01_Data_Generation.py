# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Segmentation Data Generation
# MAGIC 
# MAGIC This notebook generates synthetic customer data for demonstration purposes, replacing the external Kaggle data dependency with realistic e-commerce customer behavior patterns.
# MAGIC 
# MAGIC **Key Features:**
# MAGIC - 10,000 unique customers
# MAGIC - ~250,000 transactions over 2 years
# MAGIC - Realistic purchasing patterns and demographics
# MAGIC - Unity Catalog integration with managed tables

# COMMAND ----------

# MAGIC %pip install Faker

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries and Set Configuration
import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

# Configuration
CATALOG = "solacc_uc"
SCHEMA = "segmentation_modern"
VOLUME_NAME = "rawfiles"
NUM_CUSTOMERS = 10000
NUM_TRANSACTIONS = 250000

# Set random seeds for reproducible results
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

print(f"Generating data for {NUM_CUSTOMERS:,} customers and ~{NUM_TRANSACTIONS:,} transactions")

# COMMAND ----------

# DBTITLE 1,Setup Unity Catalog Structure
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Generate Customer Demographics
@dlt.table(
    name="customers",
    comment="Generated customer demographic data for segmentation analysis"
)
def customers():
    """Generate synthetic customer demographic data"""
    
    # Define realistic distributions
    age_brackets = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    age_weights = [0.12, 0.22, 0.25, 0.20, 0.15, 0.06]
    
    income_brackets = ['Under 25K', '25-34K', '35-49K', '50-74K', '75-99K', '100-124K', '125-149K', '150K+']
    income_weights = [0.15, 0.12, 0.18, 0.22, 0.15, 0.10, 0.05, 0.03]
    
    household_sizes = [1, 2, 3, 4, 5]
    household_weights = [0.25, 0.35, 0.20, 0.15, 0.05]
    
    # Generate customer data
    customers_data = []
    
    for customer_id in range(1, NUM_CUSTOMERS + 1):
        age_bracket = np.random.choice(age_brackets, p=age_weights)
        income_bracket = np.random.choice(income_brackets, p=income_weights)
        household_size = np.random.choice(household_sizes, p=household_weights)
        
        # Create customer record
        customer = {
            'customer_id': customer_id,
            'age_bracket': age_bracket,
            'income_bracket': income_bracket,
            'household_size': household_size,
            'city': fake.city(),
            'state': fake.state_abbr(),
            'signup_date': fake.date_between(start_date='-3y', end_date='-6m'),
            'preferred_channel': np.random.choice(['Online', 'Mobile', 'Store'], p=[0.45, 0.35, 0.20])
        }
        customers_data.append(customer)
    
    # Convert to Spark DataFrame
    customers_df = spark.createDataFrame(customers_data)
    return customers_df

# COMMAND ----------

# DBTITLE 1,Generate Product Catalog
@dlt.table(
    name="products", 
    comment="Product catalog with categories and pricing"
)
def products():
    """Generate product catalog data"""
    
    # Product categories with realistic pricing
    categories = {
        'Electronics': {'price_range': (50, 2000), 'margin': 0.25},
        'Clothing': {'price_range': (15, 300), 'margin': 0.60},
        'Home & Garden': {'price_range': (10, 500), 'margin': 0.40},
        'Books': {'price_range': (5, 50), 'margin': 0.50},
        'Sports': {'price_range': (20, 400), 'margin': 0.35},
        'Beauty': {'price_range': (8, 150), 'margin': 0.70},
        'Food & Grocery': {'price_range': (2, 100), 'margin': 0.30}
    }
    
    products_data = []
    product_id = 1
    
    for category, details in categories.items():
        # Generate 50-150 products per category
        num_products = random.randint(50, 150)
        
        for _ in range(num_products):
            price = round(random.uniform(*details['price_range']), 2)
            cost = round(price * (1 - details['margin']), 2)
            
            product = {
                'product_id': product_id,
                'product_name': f"{fake.catch_phrase()} {category} Item",
                'category': category,
                'price': price,
                'cost': cost,
                'brand': fake.company(),
                'is_seasonal': random.choice([True, False]) if category in ['Clothing', 'Sports'] else False
            }
            products_data.append(product)
            product_id += 1
    
    products_df = spark.createDataFrame(products_data)
    return products_df

# COMMAND ----------

# DBTITLE 1,Generate Transaction Data
@dlt.table(
    name="transactions",
    comment="Customer transaction data with purchasing behavior patterns"
)
def transactions():
    """Generate realistic customer transaction data"""
    
    # Get products for transaction generation
    products_df = spark.table("products").collect()
    products_list = [(row.product_id, row.category, row.price) for row in products_df]
    
    # Transaction generation parameters
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 1, 1)
    total_days = (end_date - start_date).days
    
    transactions_data = []
    transaction_id = 1
    
    # Generate customer purchasing profiles
    for customer_id in range(1, NUM_CUSTOMERS + 1):
        # Determine customer profile
        purchase_frequency = np.random.gamma(2, 2)  # Transactions per month
        avg_basket_size = max(1, int(np.random.gamma(1.5, 2)))  # Items per transaction
        price_sensitivity = random.uniform(0.3, 1.0)  # Lower = more price sensitive
        
        # Category preferences (some customers prefer certain categories)
        category_preferences = {}
        all_categories = list(set([p[1] for p in products_list]))
        for cat in all_categories:
            category_preferences[cat] = random.uniform(0.1, 1.0)
        
        # Generate transactions for this customer
        num_transactions = max(1, int(np.random.poisson(purchase_frequency * 24)))  # 24 months
        
        for _ in range(num_transactions):
            # Random transaction date
            random_days = random.randint(0, total_days)
            transaction_date = start_date + timedelta(days=random_days)
            
            # Seasonal effects
            month = transaction_date.month
            seasonal_multiplier = 1.0
            if month in [11, 12]:  # Holiday season
                seasonal_multiplier = 1.5
            elif month in [6, 7, 8]:  # Summer
                seasonal_multiplier = 1.2
            
            # Determine number of items in this transaction
            basket_size = max(1, int(np.random.poisson(avg_basket_size * seasonal_multiplier)))
            
            # Select products for this transaction
            available_products = products_list.copy()
            
            for item in range(basket_size):
                if not available_products:
                    break
                    
                # Weight product selection by category preference and price sensitivity
                weights = []
                for prod_id, category, price in available_products:
                    weight = category_preferences.get(category, 0.5)
                    # Price sensitivity effect
                    if price > 100:
                        weight *= price_sensitivity
                    weights.append(weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in weights]
                    selected_idx = np.random.choice(len(available_products), p=weights)
                    selected_product = available_products.pop(selected_idx)
                    
                    prod_id, category, price = selected_product
                    quantity = random.choices([1, 2, 3, 4], weights=[0.7, 0.2, 0.08, 0.02])[0]
                    
                    # Apply random discounts occasionally
                    discount_amount = 0
                    if random.random() < 0.15:  # 15% chance of discount
                        discount_amount = round(price * random.uniform(0.05, 0.25), 2)
                    
                    final_price = max(0, price - discount_amount) * quantity
                    
                    transaction = {
                        'transaction_id': transaction_id,
                        'customer_id': customer_id,
                        'product_id': prod_id,
                        'transaction_date': transaction_date,
                        'quantity': quantity,
                        'unit_price': price,
                        'discount_amount': discount_amount * quantity,
                        'total_amount': final_price,
                        'category': category
                    }
                    transactions_data.append(transaction)
                    transaction_id += 1
    
    # Convert to Spark DataFrame and limit to target transaction count
    transactions_df = spark.createDataFrame(transactions_data)
    
    # If we generated too many transactions, sample down to target
    if transactions_df.count() > NUM_TRANSACTIONS:
        fraction = NUM_TRANSACTIONS / transactions_df.count()
        transactions_df = transactions_df.sample(fraction=fraction, seed=42)
    
    return transactions_df

# COMMAND ----------

# DBTITLE 1,Create Customer Summary Table
@dlt.table(
    name="customer_summary",
    comment="Customer-level summary metrics for segmentation analysis"
)
def customer_summary():
    """Create customer-level summary metrics from transaction data"""
    
    return spark.sql("""
        SELECT 
            c.customer_id,
            c.age_bracket,
            c.income_bracket,
            c.household_size,
            c.city,
            c.state,
            c.signup_date,
            c.preferred_channel,
            
            -- Transaction metrics
            COUNT(t.transaction_id) as total_transactions,
            COUNT(DISTINCT t.product_id) as unique_products_purchased,
            COUNT(DISTINCT t.category) as unique_categories_purchased,
            SUM(t.total_amount) as total_spent,
            AVG(t.total_amount) as avg_transaction_value,
            SUM(t.quantity) as total_items_purchased,
            
            -- Recency metrics
            MAX(t.transaction_date) as last_purchase_date,
            DATEDIFF(CURRENT_DATE(), MAX(t.transaction_date)) as days_since_last_purchase,
            
            -- Frequency metrics  
            COUNT(t.transaction_id) / DATEDIFF(MAX(t.transaction_date), MIN(t.transaction_date)) * 30 as avg_monthly_frequency,
            
            -- Monetary metrics
            SUM(t.total_amount) / COUNT(t.transaction_id) as avg_order_value,
            
            -- Discount behavior
            AVG(CASE WHEN t.discount_amount > 0 THEN 1 ELSE 0 END) as discount_usage_rate,
            AVG(t.discount_amount) as avg_discount_amount
            
        FROM customers c
        INNER JOIN transactions t ON c.customer_id = t.customer_id
        GROUP BY c.customer_id, c.age_bracket, c.income_bracket, c.household_size, 
                 c.city, c.state, c.signup_date, c.preferred_channel
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Generation Complete
# MAGIC 
# MAGIC The synthetic dataset has been generated with:
# MAGIC - **Customer demographics** with realistic distributions
# MAGIC - **Product catalog** across multiple categories
# MAGIC - **Transaction history** with behavioral patterns
# MAGIC - **Customer summary metrics** for segmentation analysis
# MAGIC 
# MAGIC All tables are created as Unity Catalog managed tables and ready for the segmentation analysis in the next notebook.