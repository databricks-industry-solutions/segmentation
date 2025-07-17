# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Segmentation Analysis
# MAGIC 
# MAGIC This notebook performs RFM (Recency, Frequency, Monetary) analysis combined with behavioral clustering to segment customers into actionable groups.
# MAGIC 
# MAGIC **Segmentation Approach:**
# MAGIC - **RFM Analysis**: Traditional recency, frequency, monetary scoring
# MAGIC - **Behavioral Features**: Purchase patterns, category preferences, seasonality
# MAGIC - **K-Means Clustering**: Data-driven segment discovery
# MAGIC - **Segment Profiling**: Business-friendly segment descriptions

# COMMAND ----------

# DBTITLE 1,Import Libraries and Setup
import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
import numpy as np

# Configuration
CATALOG = "solacc_uc"
SCHEMA = "segmentation_modern"

# Use the tables created in the previous notebook
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print("Starting customer segmentation analysis...")

# COMMAND ----------

# DBTITLE 1,Create RFM Scores
@dlt.table(
    name="rfm_analysis",
    comment="RFM (Recency, Frequency, Monetary) analysis for customer segmentation"
)
def rfm_analysis():
    """Calculate RFM scores for each customer"""
    
    return spark.sql("""
        WITH rfm_metrics AS (
            SELECT 
                customer_id,
                -- Recency: Days since last purchase (lower is better)
                days_since_last_purchase as recency,
                
                -- Frequency: Total number of transactions  
                total_transactions as frequency,
                
                -- Monetary: Total amount spent
                total_spent as monetary,
                
                -- Additional behavioral metrics
                avg_transaction_value,
                unique_categories_purchased,
                discount_usage_rate,
                avg_monthly_frequency
            FROM customer_summary
            WHERE total_transactions > 0
        ),
        
        rfm_scores AS (
            SELECT *,
                -- RFM Scoring (1-5 scale, 5 being best)
                -- Recency: Lower days = higher score
                CASE 
                    WHEN recency <= 30 THEN 5
                    WHEN recency <= 60 THEN 4  
                    WHEN recency <= 120 THEN 3
                    WHEN recency <= 240 THEN 2
                    ELSE 1
                END as recency_score,
                
                -- Frequency: More transactions = higher score
                CASE
                    WHEN frequency >= 20 THEN 5
                    WHEN frequency >= 10 THEN 4
                    WHEN frequency >= 5 THEN 3
                    WHEN frequency >= 2 THEN 2
                    ELSE 1
                END as frequency_score,
                
                -- Monetary: Higher spend = higher score  
                CASE
                    WHEN monetary >= 2000 THEN 5
                    WHEN monetary >= 1000 THEN 4
                    WHEN monetary >= 500 THEN 3
                    WHEN monetary >= 200 THEN 2
                    ELSE 1
                END as monetary_score
                
            FROM rfm_metrics
        )
        
        SELECT *,
            -- Combined RFM Score
            (recency_score + frequency_score + monetary_score) / 3.0 as rfm_score,
            
            -- RFM Segment based on individual scores
            CASE 
                WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
                WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 4 THEN 'Loyal Customers'
                WHEN recency_score >= 4 AND frequency_score >= 2 THEN 'Potential Loyalists'  
                WHEN recency_score >= 4 AND frequency_score = 1 THEN 'New Customers'
                WHEN recency_score >= 3 AND frequency_score >= 2 AND monetary_score >= 2 THEN 'Promising'
                WHEN recency_score >= 2 AND frequency_score >= 3 THEN 'Need Attention'
                WHEN recency_score >= 2 AND frequency_score >= 2 THEN 'About to Sleep'
                WHEN recency_score >= 2 THEN 'At Risk'
                WHEN recency_score = 1 AND frequency_score >= 4 THEN 'Cannot Lose Them'
                WHEN recency_score = 1 AND frequency_score >= 2 THEN 'Hibernating'
                ELSE 'Lost'
            END as rfm_segment
            
        FROM rfm_scores
    """)

# COMMAND ----------

# DBTITLE 1,Create Behavioral Features for Clustering
@dlt.table(
    name="customer_features", 
    comment="Engineered features for customer behavioral clustering"
)
def customer_features():
    """Create behavioral features for advanced clustering"""
    
    return spark.sql("""
        WITH customer_behavior AS (
            SELECT 
                cs.customer_id,
                cs.age_bracket,
                cs.income_bracket,
                cs.household_size,
                cs.preferred_channel,
                
                -- Normalize RFM metrics (0-1 scale)
                rfm.recency / 365.0 as recency_norm,
                rfm.frequency / 50.0 as frequency_norm, 
                rfm.monetary / 5000.0 as monetary_norm,
                
                -- Behavioral features
                cs.avg_transaction_value / 200.0 as avg_order_value_norm,
                cs.unique_categories_purchased / 7.0 as category_diversity,
                cs.discount_usage_rate,
                LEAST(cs.avg_monthly_frequency / 10.0, 1.0) as purchase_regularity,
                
                -- Category preferences (from transactions)
                COALESCE(electronics_pct, 0) as electronics_preference,
                COALESCE(clothing_pct, 0) as clothing_preference, 
                COALESCE(food_pct, 0) as food_preference,
                COALESCE(home_pct, 0) as home_preference,
                
                -- Temporal behavior
                COALESCE(weekend_pct, 0) as weekend_shopper,
                COALESCE(evening_pct, 0) as evening_shopper
                
            FROM customer_summary cs
            INNER JOIN rfm_analysis rfm ON cs.customer_id = rfm.customer_id
            LEFT JOIN (
                -- Category preferences by customer
                SELECT 
                    customer_id,
                    SUM(CASE WHEN category = 'Electronics' THEN total_amount ELSE 0 END) / SUM(total_amount) as electronics_pct,
                    SUM(CASE WHEN category = 'Clothing' THEN total_amount ELSE 0 END) / SUM(total_amount) as clothing_pct,
                    SUM(CASE WHEN category = 'Food & Grocery' THEN total_amount ELSE 0 END) / SUM(total_amount) as food_pct,
                    SUM(CASE WHEN category = 'Home & Garden' THEN total_amount ELSE 0 END) / SUM(total_amount) as home_pct
                FROM transactions
                GROUP BY customer_id
            ) cat_prefs ON cs.customer_id = cat_prefs.customer_id
            LEFT JOIN (
                -- Temporal shopping patterns  
                SELECT
                    customer_id,
                    SUM(CASE WHEN DAYOFWEEK(transaction_date) IN (1,7) THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as weekend_pct,
                    SUM(CASE WHEN HOUR(transaction_date) >= 18 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as evening_pct
                FROM transactions 
                GROUP BY customer_id
            ) temporal ON cs.customer_id = temporal.customer_id
        )
        
        SELECT 
            customer_id,
            age_bracket,
            income_bracket, 
            household_size,
            preferred_channel,
            recency_norm,
            frequency_norm,
            monetary_norm,
            avg_order_value_norm,
            category_diversity,
            discount_usage_rate,
            purchase_regularity,
            electronics_preference,
            clothing_preference,
            food_preference,
            home_preference,
            weekend_shopper,
            evening_shopper,
            
            -- Create feature vector for clustering
            array(
                recency_norm, frequency_norm, monetary_norm,
                avg_order_value_norm, category_diversity, discount_usage_rate,
                purchase_regularity, electronics_preference, clothing_preference,
                food_preference, home_preference, weekend_shopper, evening_shopper
            ) as features_array
            
        FROM customer_behavior
    """)

# COMMAND ----------

# DBTITLE 1,Perform K-Means Clustering
@dlt.table(
    name="customer_segments",
    comment="Final customer segments combining RFM analysis with behavioral clustering"
)  
def customer_segments():
    """Perform K-means clustering to identify customer segments"""
    
    # Read the customer features
    features_df = spark.table("customer_features")
    
    # Convert array to vector for ML
    assembler = VectorAssembler(
        inputCols=["features_array"], 
        outputCol="features_vector"
    )
    
    # Scale features
    scaler = StandardScaler(
        inputCol="features_vector",
        outputCol="scaled_features",
        withStd=True,
        withMean=True
    )
    
    # K-means clustering (try different k values and pick optimal)
    kmeans = KMeans(
        featuresCol="scaled_features",
        predictionCol="cluster_id", 
        k=6,  # Start with 6 clusters
        seed=42,
        maxIter=100
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    
    # Fit the pipeline
    model = pipeline.fit(features_df)
    
    # Make predictions
    clustered_df = model.transform(features_df)
    
    # Add cluster interpretations
    return clustered_df.selectExpr(
        "customer_id",
        "age_bracket", 
        "income_bracket",
        "household_size",
        "preferred_channel",
        "recency_norm",
        "frequency_norm", 
        "monetary_norm",
        "avg_order_value_norm",
        "category_diversity",
        "discount_usage_rate",
        "purchase_regularity",
        "electronics_preference",
        "clothing_preference",
        "food_preference", 
        "home_preference",
        "weekend_shopper",
        "evening_shopper",
        "cluster_id",
        """
        CASE cluster_id
            WHEN 0 THEN 'High-Value Loyalists'
            WHEN 1 THEN 'Frequent Shoppers' 
            WHEN 2 THEN 'Discount Hunters'
            WHEN 3 THEN 'Occasional Buyers'
            WHEN 4 THEN 'New/Inactive Customers'
            WHEN 5 THEN 'Category Specialists'
            ELSE 'Unclassified'
        END as segment_name
        """
    )

# COMMAND ----------

# DBTITLE 1,Create Segment Profiles
@dlt.table(
    name="segment_profiles",
    comment="Aggregate profiles and characteristics of each customer segment"
)
def segment_profiles():
    """Create detailed profiles for each customer segment"""
    
    return spark.sql("""
        SELECT 
            segment_name,
            cluster_id,
            COUNT(*) as customer_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as segment_percentage,
            
            -- RFM Characteristics
            ROUND(AVG(recency_norm * 365), 0) as avg_days_since_purchase,
            ROUND(AVG(frequency_norm * 50), 1) as avg_transaction_frequency,
            ROUND(AVG(monetary_norm * 5000), 0) as avg_total_spent,
            ROUND(AVG(avg_order_value_norm * 200), 0) as avg_order_value,
            
            -- Behavioral Characteristics  
            ROUND(AVG(category_diversity * 7), 1) as avg_category_diversity,
            ROUND(AVG(discount_usage_rate), 2) as avg_discount_usage,
            ROUND(AVG(purchase_regularity * 10), 1) as avg_monthly_frequency,
            
            -- Category Preferences
            ROUND(AVG(electronics_preference), 2) as electronics_preference,
            ROUND(AVG(clothing_preference), 2) as clothing_preference,
            ROUND(AVG(food_preference), 2) as food_preference,
            ROUND(AVG(home_preference), 2) as home_preference,
            
            -- Shopping Patterns
            ROUND(AVG(weekend_shopper), 2) as weekend_shopping_rate,
            ROUND(AVG(evening_shopper), 2) as evening_shopping_rate,
            
            -- Demographics
            MODE() WITHIN GROUP (ORDER BY age_bracket) as most_common_age,
            MODE() WITHIN GROUP (ORDER BY income_bracket) as most_common_income,
            ROUND(AVG(household_size), 1) as avg_household_size,
            MODE() WITHIN GROUP (ORDER BY preferred_channel) as preferred_channel
            
        FROM customer_segments
        GROUP BY segment_name, cluster_id
        ORDER BY customer_count DESC
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Segmentation Analysis Complete
# MAGIC 
# MAGIC The customer segmentation analysis has created:
# MAGIC 
# MAGIC 1. **RFM Analysis** - Traditional recency, frequency, monetary scoring
# MAGIC 2. **Behavioral Features** - Advanced customer behavior patterns  
# MAGIC 3. **K-Means Clustering** - Data-driven segment discovery
# MAGIC 4. **Segment Profiles** - Business-friendly segment characteristics
# MAGIC 
# MAGIC The next notebook will provide detailed business insights and visualizations for each segment.