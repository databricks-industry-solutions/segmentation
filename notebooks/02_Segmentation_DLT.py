# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Segmentation DLT Pipeline
# MAGIC 
# MAGIC This DLT pipeline transforms raw customer data into segmented customer insights using SQL-based transformations.
# MAGIC 
# MAGIC **Pipeline Flow:**
# MAGIC 1. Clean and prepare customer data
# MAGIC 2. Calculate RFM metrics
# MAGIC 3. Create customer segments using clustering logic
# MAGIC 4. Generate segment profiles

# COMMAND ----------

import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Get catalog and schema from pipeline configuration
catalog_name = spark.conf.get("catalog") or "dev_customer_segmentation"
schema_name = spark.conf.get("schema") or "segmentation"

# COMMAND ----------

# DBTITLE 1,Clean Customer Data
@dlt.table(
    name="customers",
    comment="Clean customer demographic data"
)
def customers():
    return spark.sql(f"""
        SELECT 
            customer_id,
            age_bracket,
            income_bracket,
            household_size,
            city,
            state,
            signup_date,
            preferred_channel
        FROM {catalog_name}.{schema_name}.raw_customers
        WHERE customer_id IS NOT NULL
    """)

# COMMAND ----------

# DBTITLE 1,Clean Product Data
@dlt.table(
    name="products",
    comment="Clean product catalog data"
)
def products():
    return spark.sql(f"""
        SELECT 
            product_id,
            product_name,
            category,
            price,
            cost,
            brand,
            is_seasonal,
            price - cost as profit_margin
        FROM {catalog_name}.{schema_name}.raw_products
        WHERE product_id IS NOT NULL AND price > 0
    """)

# COMMAND ----------

# DBTITLE 1,Clean Transaction Data
@dlt.table(
    name="transactions",
    comment="Clean transaction data with calculated fields"
)
def transactions():
    return spark.sql(f"""
        SELECT 
            transaction_id,
            customer_id,
            product_id,
            transaction_date,
            quantity,
            unit_price,
            discount_amount,
            total_amount,
            category,
            -- Calculate additional metrics
            DATEDIFF(CURRENT_DATE(), transaction_date) as days_since_transaction,
            YEAR(transaction_date) as transaction_year,
            MONTH(transaction_date) as transaction_month,
            DAYOFWEEK(transaction_date) as day_of_week,
            CASE WHEN DAYOFWEEK(transaction_date) IN (1,7) THEN 1 ELSE 0 END as is_weekend
        FROM {catalog_name}.{schema_name}.raw_transactions
        WHERE customer_id IS NOT NULL 
          AND product_id IS NOT NULL 
          AND total_amount >= 0
    """)

# COMMAND ----------

# DBTITLE 1,Customer Summary Metrics
@dlt.table(
    name="customer_summary",
    comment="Customer-level summary metrics for segmentation"
)
def customer_summary():
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
            COUNT(t.transaction_id) / GREATEST(DATEDIFF(MAX(t.transaction_date), MIN(t.transaction_date)), 1) * 30 as avg_monthly_frequency,
            
            -- Discount behavior
            AVG(CASE WHEN t.discount_amount > 0 THEN 1 ELSE 0 END) as discount_usage_rate,
            AVG(t.discount_amount) as avg_discount_amount,
            
            -- Category preferences
            SUM(CASE WHEN t.category = 'Electronics' THEN t.total_amount ELSE 0 END) / SUM(t.total_amount) as electronics_preference,
            SUM(CASE WHEN t.category = 'Clothing' THEN t.total_amount ELSE 0 END) / SUM(t.total_amount) as clothing_preference,
            SUM(CASE WHEN t.category = 'Food & Grocery' THEN t.total_amount ELSE 0 END) / SUM(t.total_amount) as food_preference,
            SUM(CASE WHEN t.category = 'Home & Garden' THEN t.total_amount ELSE 0 END) / SUM(t.total_amount) as home_preference,
            
            -- Shopping behavior
            AVG(t.is_weekend) as weekend_shopping_rate
            
        FROM live.customers c
        INNER JOIN live.transactions t ON c.customer_id = t.customer_id
        GROUP BY c.customer_id, c.age_bracket, c.income_bracket, c.household_size, 
                 c.city, c.state, c.signup_date, c.preferred_channel
    """)

# COMMAND ----------

# DBTITLE 1,RFM Analysis
@dlt.table(
    name="rfm_analysis",
    comment="RFM (Recency, Frequency, Monetary) analysis for customer segmentation"
)
def rfm_analysis():
    return spark.sql("""
        WITH rfm_metrics AS (
            SELECT 
                customer_id,
                days_since_last_purchase as recency,
                total_transactions as frequency,
                total_spent as monetary,
                avg_transaction_value,
                unique_categories_purchased,
                discount_usage_rate,
                avg_monthly_frequency
            FROM live.customer_summary
            WHERE total_transactions > 0
        ),
        
        rfm_quartiles AS (
            SELECT 
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY recency) as recency_q1,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY recency) as recency_q2,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY recency) as recency_q3,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY frequency) as frequency_q1,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY frequency) as frequency_q2,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY frequency) as frequency_q3,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY monetary) as monetary_q1,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY monetary) as monetary_q2,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY monetary) as monetary_q3
            FROM rfm_metrics
        )
        
        SELECT 
            rm.*,
            -- Recency Score (1-4, higher is better for recency - lower days)
            CASE 
                WHEN rm.recency <= rq.recency_q1 THEN 4
                WHEN rm.recency <= rq.recency_q2 THEN 3
                WHEN rm.recency <= rq.recency_q3 THEN 2
                ELSE 1
            END as recency_score,
            
            -- Frequency Score (1-4, higher is better)
            CASE 
                WHEN rm.frequency >= rq.frequency_q3 THEN 4
                WHEN rm.frequency >= rq.frequency_q2 THEN 3
                WHEN rm.frequency >= rq.frequency_q1 THEN 2
                ELSE 1
            END as frequency_score,
            
            -- Monetary Score (1-4, higher is better)
            CASE 
                WHEN rm.monetary >= rq.monetary_q3 THEN 4
                WHEN rm.monetary >= rq.monetary_q2 THEN 3
                WHEN rm.monetary >= rq.monetary_q1 THEN 2
                ELSE 1
            END as monetary_score
            
        FROM rfm_metrics rm
        CROSS JOIN rfm_quartiles rq
    """)

# COMMAND ----------

# DBTITLE 1,Customer Segments
@dlt.table(
    name="customer_segments",
    comment="Customer segments based on RFM analysis and behavioral patterns"
)
def customer_segments():
    return spark.sql("""
        WITH rfm_segments AS (
            SELECT 
                customer_id,
                recency,
                frequency,
                monetary,
                recency_score,
                frequency_score,
                monetary_score,
                (recency_score + frequency_score + monetary_score) / 3.0 as rfm_score,
                
                -- RFM Segment Classification
                CASE 
                    WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Champions'
                    WHEN recency_score >= 2 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
                    WHEN recency_score >= 3 AND frequency_score >= 2 AND monetary_score >= 2 THEN 'Potential Loyalists'  
                    WHEN recency_score >= 3 AND frequency_score = 1 AND monetary_score >= 1 THEN 'New Customers'
                    WHEN recency_score >= 2 AND frequency_score >= 2 AND monetary_score >= 2 THEN 'Promising'
                    WHEN recency_score >= 2 AND frequency_score >= 3 AND monetary_score <= 2 THEN 'Need Attention'
                    WHEN recency_score >= 2 AND frequency_score >= 2 AND monetary_score <= 2 THEN 'About to Sleep'
                    WHEN recency_score >= 2 AND frequency_score <= 2 THEN 'At Risk'
                    WHEN recency_score = 1 AND frequency_score >= 3 THEN 'Cannot Lose Them'
                    WHEN recency_score = 1 AND frequency_score >= 2 THEN 'Hibernating'
                    ELSE 'Lost'
                END as rfm_segment,
                
                avg_transaction_value,
                unique_categories_purchased,
                discount_usage_rate,
                avg_monthly_frequency
                
            FROM live.rfm_analysis
        ),
        
        behavioral_segments AS (
            SELECT 
                cs.customer_id,
                cs.age_bracket,
                cs.income_bracket,
                cs.household_size,
                cs.preferred_channel,
                cs.weekend_shopping_rate,
                cs.electronics_preference,
                cs.clothing_preference,
                cs.food_preference,
                cs.home_preference,
                rf.recency,
                rf.frequency,
                rf.monetary,
                rf.recency_score,
                rf.frequency_score,
                rf.monetary_score,
                rf.rfm_score,
                rf.rfm_segment,
                rf.avg_transaction_value,
                rf.unique_categories_purchased,
                rf.discount_usage_rate,
                rf.avg_monthly_frequency,
                
                -- Behavioral segment based on shopping patterns
                CASE 
                    WHEN rf.rfm_score >= 3.5 AND cs.avg_transaction_value >= 100 THEN 'High-Value Loyalists'
                    WHEN rf.frequency_score >= 3 AND cs.avg_monthly_frequency >= 2 THEN 'Frequent Shoppers'
                    WHEN cs.discount_usage_rate >= 0.3 THEN 'Discount Hunters'
                    WHEN rf.frequency_score <= 2 AND rf.monetary_score >= 2 THEN 'Occasional Buyers'
                    WHEN rf.recency_score <= 2 OR rf.frequency_score = 1 THEN 'New/Inactive Customers'
                    WHEN cs.unique_categories_purchased <= 2 THEN 'Category Specialists'
                    ELSE 'Regular Customers'
                END as behavioral_segment
                
            FROM live.customer_summary cs
            INNER JOIN rfm_segments rf ON cs.customer_id = rf.customer_id
        )
        
        SELECT 
            customer_id,
            age_bracket,
            income_bracket,
            household_size,
            preferred_channel,
            recency,
            frequency,
            monetary,
            recency_score,
            frequency_score,
            monetary_score,
            rfm_score,
            rfm_segment,
            behavioral_segment,
            -- Use behavioral segment as primary segment
            behavioral_segment as segment_name,
            avg_transaction_value,
            unique_categories_purchased,
            discount_usage_rate,
            avg_monthly_frequency,
            weekend_shopping_rate,
            electronics_preference,
            clothing_preference,
            food_preference,
            home_preference
            
        FROM behavioral_segments
    """)

# COMMAND ----------

# DBTITLE 1,Segment Profiles
@dlt.table(
    name="segment_profiles",
    comment="Aggregate profiles and characteristics of each customer segment"
)
def segment_profiles():
    return spark.sql("""
        SELECT 
            segment_name,
            COUNT(*) as customer_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as segment_percentage,
            
            -- RFM Characteristics
            ROUND(AVG(recency), 0) as avg_days_since_purchase,
            ROUND(AVG(frequency), 1) as avg_transaction_frequency,
            ROUND(AVG(monetary), 0) as avg_total_spent,
            ROUND(AVG(avg_transaction_value), 0) as avg_order_value,
            
            -- Behavioral Characteristics  
            ROUND(AVG(unique_categories_purchased), 1) as avg_category_diversity,
            ROUND(AVG(discount_usage_rate), 2) as avg_discount_usage,
            ROUND(AVG(avg_monthly_frequency), 1) as avg_monthly_frequency,
            
            -- Category Preferences
            ROUND(AVG(electronics_preference), 2) as electronics_preference,
            ROUND(AVG(clothing_preference), 2) as clothing_preference,
            ROUND(AVG(food_preference), 2) as food_preference,
            ROUND(AVG(home_preference), 2) as home_preference,
            
            -- Shopping Patterns
            ROUND(AVG(weekend_shopping_rate), 2) as weekend_shopping_rate,
            
            -- Demographics
            MODE() WITHIN GROUP (ORDER BY age_bracket) as most_common_age,
            MODE() WITHIN GROUP (ORDER BY income_bracket) as most_common_income,
            ROUND(AVG(household_size), 1) as avg_household_size,
            MODE() WITHIN GROUP (ORDER BY preferred_channel) as preferred_channel
            
        FROM live.customer_segments
        GROUP BY segment_name
        ORDER BY customer_count DESC
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DLT Pipeline Complete ✅
# MAGIC 
# MAGIC This pipeline transforms raw customer data into actionable customer segments:
# MAGIC 
# MAGIC 1. **Clean Data Tables**: customers, products, transactions
# MAGIC 2. **Customer Summary**: Aggregated customer metrics
# MAGIC 3. **RFM Analysis**: Recency, Frequency, Monetary scoring
# MAGIC 4. **Customer Segments**: Behavioral segmentation with 6 distinct groups
# MAGIC 5. **Segment Profiles**: Business-ready segment characteristics
# MAGIC 
# MAGIC The segmented data is ready for business insights and visualization!