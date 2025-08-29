# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Segmentation Lakeflow Declarative Pipeline
# MAGIC 
# MAGIC This Lakeflow Declarative Pipeline transforms raw customer data into segmented customer insights using SQL-based transformations.
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
    name="customers_clean",
    comment="Clean customer demographic data"
)
@dlt.expect_or_drop("valid_customer_age", "age >= 18")
def customers_clean():
    return spark.sql(f"""
        SELECT 
            customer_id,
            age,
            CASE
                WHEN age BETWEEN 18 and 24 THEN 'Young Adult'
                WHEN age BETWEEN 25 and 34 THEN 'Emerging Professional'
                WHEN age BETWEEN 35 AND 49 THEN 'Established Professionals'
                WHEN age BETWEEN 50 and 65 THEN 'Mature Adults'
            END AS age_bracket,
            gender,
            income_bracket,
            city,
            state,
            location,
            signup_date,
            preferred_channel
        FROM {catalog_name}.{schema_name}.raw_customer_profiles
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
            transaction_date,
            product_id,
            quantity,
            category,
            per_unit_cost,
            per_unit_msrp,
            total_cost,
            pre_discount_amount,
            discount_amount,
            total_amount,
            channel,
            -- Calculate additional metrics
            DATEDIFF(CURRENT_DATE(), transaction_date) as days_since_transaction,
            YEAR(transaction_date) as transaction_year,
            MONTH(transaction_date) as transaction_month,
            DAYOFWEEK(transaction_date) as day_of_week,
            CASE WHEN DAYOFWEEK(transaction_date) IN (1,7) THEN 1 ELSE 0 END as is_weekend
        FROM STREAM read_files("/Volumes/{catalog_name}/{schema_name}/customer_segmentation/transactions/", format=>'csv')
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
            c.gender,
            c.city,
            c.state,
            c.signup_date,
            c.preferred_channel,
            
            -- Transaction metrics
            COUNT(t.transaction_id) as total_transactions,
            COUNT(DISTINCT t.product_id) as unique_products_purchased,
            COUNT(DISTINCT t.category) as unique_categories_purchased,
            ROUND(SUM(t.total_amount), 2) as total_spent,
            ROUND(AVG(t.total_amount), 2) as avg_transaction_value,
            SUM(t.quantity) as total_items_purchased,
            
            -- Recency metrics
            MAX(t.transaction_date) as last_purchase_date,
            DATEDIFF(CURRENT_DATE(), MAX(t.transaction_date)) as days_since_last_purchase,
            
            -- Frequency metrics  
            ROUND(COUNT(t.transaction_id) / GREATEST(DATEDIFF(MAX(t.transaction_date), MIN(t.transaction_date)), 1) * 30, 2) as avg_monthly_frequency,
            
            -- Discount behavior
            ROUND(AVG(CASE WHEN t.discount_amount > 0 THEN 1 ELSE 0 END), 2) as discount_usage_rate,
            ROUND(AVG(t.discount_amount), 2) as avg_discount_amount,
            
            -- Category preferences
            ROUND(SUM(CASE WHEN t.category = 'Electronics' THEN t.total_amount ELSE 0 END) / SUM(t.total_amount), 2) as electronics_preference,
            ROUND(SUM(CASE WHEN t.category = 'Clothing' THEN t.total_amount ELSE 0 END) / SUM(t.total_amount), 2) as clothing_preference,
            ROUND(SUM(CASE WHEN t.category = 'Food & Grocery' THEN t.total_amount ELSE 0 END) / SUM(t.total_amount), 2) as food_preference,
            ROUND(SUM(CASE WHEN t.category = 'Home & Garden' THEN t.total_amount ELSE 0 END) / SUM(t.total_amount), 2) as home_preference,
            
            -- Shopping behavior
            ROUND(AVG(t.is_weekend), 2) as weekend_shopping_rate
            
        FROM live.customers_clean c
        INNER JOIN live.transactions t ON c.customer_id = t.customer_id
        GROUP BY c.customer_id, c.age_bracket, c.income_bracket, c.gender, c.city, c.state, c.signup_date, c.preferred_channel
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
                    WHEN recency_score = 4 AND frequency_score = 4 AND monetary_score = 4 THEN 'Champions'                
                    WHEN recency_score >= 3 AND frequency_score = 1 AND monetary_score >= 1 THEN 'New Customers'
                    WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal'                    
                    WHEN recency_score >= 2 AND frequency_score <= 2 AND monetary_score <= 2 THEN 'Regular'                    
                    WHEN recency_score <= 2 AND frequency_score >= 2 AND monetary_score >= 2 THEN 'At Risk'               
                    ELSE 'Churned'                    
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
                cs.gender,
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
            gender,
            preferred_channel,
            recency,
            frequency,
            monetary,
            recency_score,
            frequency_score,
            monetary_score,
            rfm_score,
            rfm_segment,
            rfm_segment as segment_name, 
            behavioral_segment,
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
            MODE() WITHIN GROUP (ORDER BY preferred_channel) as preferred_channel
            
        FROM live.customer_segments
        GROUP BY segment_name
        ORDER BY customer_count DESC
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lakeflow Declarative Pipeline Complete ✅
# MAGIC 
# MAGIC This Lakeflow Declarative Pipeline transforms raw customer data into actionable customer segments:
# MAGIC 
# MAGIC 1. **Clean Data Tables**: customers, products, transactions
# MAGIC 2. **Customer Summary**: Aggregated customer metrics
# MAGIC 3. **RFM Analysis**: Recency, Frequency, Monetary scoring
# MAGIC 4. **Customer Segments**: Behavioral segmentation with 6 distinct groups
# MAGIC 5. **Segment Profiles**: Business-ready segment characteristics
# MAGIC 
# MAGIC The segmented data is ready for business insights and visualization!