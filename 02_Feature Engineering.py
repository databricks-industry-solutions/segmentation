# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/segmentation.git. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-segmentation.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to generate the features required for our segmentation work using a combination of feature engineering and dimension reduction techniques.

# COMMAND ----------

# DBTITLE 1,Install Required Python Libraries
# MAGIC %pip install dython==0.7.1

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from sklearn.preprocessing import quantile_transform

import dython
import math

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md ## Step 1: Derive Bases Features
# MAGIC 
# MAGIC With a stated goal of segmenting customer households based on their responsiveness to various promotional efforts, we start by calculating the number of purchase dates (*pdates\_*) and the volume of sales (*amount\_list_*) associated with each promotion item, alone and in combination with one another.  The promotional items considered are:
# MAGIC 
# MAGIC * Campaign targeted products (*campaign\_targeted_*)
# MAGIC * Private label products (*private\_label_*)
# MAGIC * InStore-discounted products (*instore\_discount_*)
# MAGIC * Campaign (retailer-generated) coupon redemptions (*campaign\_coupon\_redemption_*)
# MAGIC * Manufacturer-generated coupon redemptions (*manuf\_coupon\_redemption_*)
# MAGIC 
# MAGIC The resulting metrics are by no means exhaustive but provide a useful starting point for our analysis:

# COMMAND ----------

# DBTITLE 1,Derive Relevant Metrics
# MAGIC %sql
# MAGIC USE journey;
# MAGIC 
# MAGIC DROP VIEW IF EXISTS household_metrics;
# MAGIC 
# MAGIC CREATE VIEW household_metrics
# MAGIC AS
# MAGIC   WITH 
# MAGIC     targeted_products_by_household AS (
# MAGIC       SELECT DISTINCT
# MAGIC         b.household_id,
# MAGIC         c.product_id
# MAGIC       FROM campaigns a
# MAGIC       INNER JOIN campaigns_households b
# MAGIC         ON a.campaign_id=b.campaign_id
# MAGIC       INNER JOIN coupons c
# MAGIC         ON a.campaign_id=c.campaign_id
# MAGIC       ),
# MAGIC     product_spend AS (
# MAGIC       SELECT
# MAGIC         a.household_id,
# MAGIC         a.product_id,
# MAGIC         a.day,
# MAGIC         a.basket_id,
# MAGIC         CASE WHEN a.campaign_coupon_discount > 0 THEN 1 ELSE 0 END as campaign_coupon_redemption,
# MAGIC         CASE WHEN a.manuf_coupon_discount > 0 THEN 1 ELSE 0 END as manuf_coupon_redemption,
# MAGIC         CASE WHEN a.instore_discount > 0 THEN 1 ELSE 0 END as instore_discount_applied,
# MAGIC         CASE WHEN b.brand = 'Private' THEN 1 ELSE 0 END as private_label,
# MAGIC         a.amount_list,
# MAGIC         a.campaign_coupon_discount,
# MAGIC         a.manuf_coupon_discount,
# MAGIC         a.total_coupon_discount,
# MAGIC         a.instore_discount,
# MAGIC         a.amount_paid  
# MAGIC       FROM transactions_adj a
# MAGIC       INNER JOIN products b
# MAGIC         ON a.product_id=b.product_id
# MAGIC       )
# MAGIC   SELECT
# MAGIC 
# MAGIC     x.household_id,
# MAGIC 
# MAGIC     -- Purchase Date Level Metrics
# MAGIC     COUNT(DISTINCT x.day) as purchase_dates,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL THEN x.day ELSE NULL END) as pdates_campaign_targeted,
# MAGIC     COUNT(DISTINCT CASE WHEN x.private_label = 1 THEN x.day ELSE NULL END) as pdates_private_label,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL AND x.private_label = 1 THEN x.day ELSE NULL END) as pdates_campaign_targeted_private_label,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemptions,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 AND x.private_label = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC     COUNT(DISTINCT CASE WHEN x.manuf_coupon_redemption = 1 THEN x.day ELSE NULL END) as pdates_manuf_coupon_redemptions,
# MAGIC     COUNT(DISTINCT CASE WHEN x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_targeted_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.private_label = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_private_label_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL AND x.private_label = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 AND x.private_label = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.manuf_coupon_redemption = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC 
# MAGIC     -- List Amount Metrics
# MAGIC     COALESCE(SUM(x.amount_list),0) as amount_list,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.amount_list),0) as amount_list_with_campaign_targeted,
# MAGIC     COALESCE(SUM(x.private_label * x.amount_list),0) as amount_list_with_private_label,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.private_label * x.amount_list),0) as amount_list_with_campaign_targeted_private_label,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.amount_list),0) as amount_list_with_campaign_coupon_redemptions,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.private_label * x.amount_list),0) as amount_list_with_campaign_coupon_redemptions_on_private_labels,
# MAGIC     COALESCE(SUM(x.manuf_coupon_redemption * x.amount_list),0) as amount_list_with_manuf_coupon_redemptions,
# MAGIC     COALESCE(SUM(x.instore_discount_applied * x.amount_list),0) as amount_list_with_instore_discount_applied,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_targeted_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.private_label * x.instore_discount_applied * x.amount_list),0) as amount_list_with_private_label_instore_discount_applied,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.private_label * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_targeted_private_label_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_coupon_redemption_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.private_label * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.manuf_coupon_redemption * x.instore_discount_applied * x.amount_list),0) as amount_list_with_manuf_coupon_redemption_instore_discount_applied
# MAGIC 
# MAGIC   FROM product_spend x
# MAGIC   LEFT OUTER JOIN targeted_products_by_household y
# MAGIC     ON x.household_id=y.household_id AND x.product_id=y.product_id
# MAGIC   GROUP BY 
# MAGIC     x.household_id;
# MAGIC     
# MAGIC SELECT * FROM household_metrics;

# COMMAND ----------

# MAGIC %md It is assumed that the households included in this dataset were selected based on a minimum level of activity spanning the 711 day period over which data is provided.  That said, different households demonstrate different levels of purchase frequency during his period as well as different levels of overall spend.  In order to normalize these values between households, we'll divide each metric by the total purchase dates or total list amount associated with that household over its available purchase history:
# MAGIC 
# MAGIC **NOTE** Normalizing the data based on total purchase dates and spend as we do in this next step may not be appropriate in all analyses. 

# COMMAND ----------

# DBTITLE 1,Convert Metrics to Features
# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS household_features;
# MAGIC 
# MAGIC CREATE VIEW household_features 
# MAGIC AS 
# MAGIC 
# MAGIC SELECT
# MAGIC       household_id,
# MAGIC   
# MAGIC       pdates_campaign_targeted/purchase_dates as pdates_campaign_targeted,
# MAGIC       pdates_private_label/purchase_dates as pdates_private_label,
# MAGIC       pdates_campaign_targeted_private_label/purchase_dates as pdates_campaign_targeted_private_label,
# MAGIC       pdates_campaign_coupon_redemptions/purchase_dates as pdates_campaign_coupon_redemptions,
# MAGIC       pdates_campaign_coupon_redemptions_on_private_labels/purchase_dates as pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC       pdates_manuf_coupon_redemptions/purchase_dates as pdates_manuf_coupon_redemptions,
# MAGIC       pdates_instore_discount_applied/purchase_dates as pdates_instore_discount_applied,
# MAGIC       pdates_campaign_targeted_instore_discount_applied/purchase_dates as pdates_campaign_targeted_instore_discount_applied,
# MAGIC       pdates_private_label_instore_discount_applied/purchase_dates as pdates_private_label_instore_discount_applied,
# MAGIC       pdates_campaign_targeted_private_label_instore_discount_applied/purchase_dates as pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       pdates_campaign_coupon_redemption_instore_discount_applied/purchase_dates as pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       pdates_campaign_coupon_redemption_private_label_instore_discount_applied/purchase_dates as pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       pdates_manuf_coupon_redemption_instore_discount_applied/purchase_dates as pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC       
# MAGIC       amount_list_with_campaign_targeted/amount_list as amount_list_with_campaign_targeted,
# MAGIC       amount_list_with_private_label/amount_list as amount_list_with_private_label,
# MAGIC       amount_list_with_campaign_targeted_private_label/amount_list as amount_list_with_campaign_targeted_private_label,
# MAGIC       amount_list_with_campaign_coupon_redemptions/amount_list as amount_list_with_campaign_coupon_redemptions,
# MAGIC       amount_list_with_campaign_coupon_redemptions_on_private_labels/amount_list as amount_list_with_campaign_coupon_redemptions_on_private_labels,
# MAGIC       amount_list_with_manuf_coupon_redemptions/amount_list as amount_list_with_manuf_coupon_redemptions,
# MAGIC       amount_list_with_instore_discount_applied/amount_list as amount_list_with_instore_discount_applied,
# MAGIC       amount_list_with_campaign_targeted_instore_discount_applied/amount_list as amount_list_with_campaign_targeted_instore_discount_applied,
# MAGIC       amount_list_with_private_label_instore_discount_applied/amount_list as amount_list_with_private_label_instore_discount_applied,
# MAGIC       amount_list_with_campaign_targeted_private_label_instore_discount_applied/amount_list as amount_list_with_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       amount_list_with_campaign_coupon_redemption_instore_discount_applied/amount_list as amount_list_with_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied/amount_list as amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       amount_list_with_manuf_coupon_redemption_instore_discount_applied/amount_list as amount_list_with_manuf_coupon_redemption_instore_discount_applied
# MAGIC 
# MAGIC FROM household_metrics
# MAGIC ORDER BY household_id;
# MAGIC 
# MAGIC SELECT * FROM household_features;

# COMMAND ----------

# MAGIC %md ## Step 2: Examine Distributions
# MAGIC 
# MAGIC Before proceeding, it's a good idea to examine our features closely to understand their compatibility with clustering techniques we might employ. In general, our preference would be to have standardized data with relatively normal distributions though that's not a hard requirement for every clustering algorithm.
# MAGIC 
# MAGIC To help us examine data distributions, we'll pull our data into a pandas Dataframe.  If our data volume were too large for pandas, we might consider taking a random sample (using the [*sample()*](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sample) against the Spark DataFrame) to examine the distributions:

# COMMAND ----------

# DBTITLE 1,Retrieve Features
# retreive as Spark dataframe
household_features = (
  spark
    .table('household_features')
  )

# retrieve as pandas Dataframe
household_features_pd = household_features.toPandas()

# collect some basic info on our features
household_features_pd.info()

# COMMAND ----------

# MAGIC %md Notice that we have elected to retrieve the *household_id* field with this dataset.  Unique identifiers such as this will not be passed into the data transformation and clustering work that follows but may be useful in helping us validate the results of that work. By retrieving this information with our features, we can now separate our features and the unique identifier into two separate pandas dataframes where instances in each can easily be reassociated leveraging a shared index value:

# COMMAND ----------

# DBTITLE 1,Separate Household ID from Features
# get household ids from dataframe
households_pd = household_features_pd[['household_id']]

# remove household ids from dataframe
features_pd = household_features_pd.drop(['household_id'], axis=1)

features_pd

# COMMAND ----------

# MAGIC %md Let's now start examining the structure of our features:

# COMMAND ----------

# DBTITLE 1,Summary Stats on Features
features_pd.describe()

# COMMAND ----------

# MAGIC %md A quick review of the features finds that many have very low means and a large number of zero values (as indicated by the occurrence of zeros at multiple quantile positions).  We should take a closer look at the distribution of our features to make sure we don't have any data distribution problems that will trip us up later:

# COMMAND ----------

# DBTITLE 1,Examine Feature Distributions
feature_names = features_pd.columns
feature_count = len(feature_names)

# determine required rows and columns for visualizations
column_count = 5
row_count = math.ceil(feature_count / column_count)

# configure figure layout
fig, ax = plt.subplots(row_count, column_count, figsize =(column_count * 4.5, row_count * 3))

# render distribution of each feature
for k in range(0,feature_count):
  
  # determine row & col position
  col = k % column_count
  row = int(k / column_count)
  
  # set figure at row & col position
  ax[row][col].hist(features_pd[feature_names[k]], rwidth=0.95, bins=10) # histogram
  ax[row][col].set_xlim(0,1)   # set x scale 0 to 1
  ax[row][col].set_ylim(0,features_pd.shape[0]) # set y scale 0 to 2500 (household count)
  ax[row][col].text(x=0.1, y=features_pd.shape[0]-100, s=feature_names[k].replace('_','\n'), fontsize=9, va='top')      # feature name in chart

# COMMAND ----------

# MAGIC %md A quick visual inspection shows us that we have *zero-inflated distributions* associated with many of our features.  This is a common challenge when a feature attempts to measure the magnitude of an event that occurs with low frequency.  
# MAGIC 
# MAGIC There is a growing body of literature describing various techniques for dealing with zero-inflated distributions and even some zero-inflated models designed to work with them.  For our purposes, we will simply separate features with these distributions into two features, one of which will capture the occurrence of the event as a binary (categorical) feature and the other which will capture the magnitude of the event when it occurs:
# MAGIC 
# MAGIC **NOTE** We will label our binary features with a *has\_* prefix to make them more easily identifiable. We expect that if a household has zero purchase dates associated with an event, we'd expect that household also has no sales amount values for that event. With that in mind, we will create a single binary feature for an event and a secondary feature for each of the associated purchase date and amount list values.

# COMMAND ----------

# DBTITLE 1,Define Features to Address Zero-Inflated Distribution Problem
# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS household_features;
# MAGIC 
# MAGIC CREATE VIEW household_features 
# MAGIC AS 
# MAGIC 
# MAGIC SELECT
# MAGIC 
# MAGIC       household_id,
# MAGIC       
# MAGIC       -- binary features
# MAGIC       CASE WHEN pdates_campaign_targeted > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted,
# MAGIC       -- CASE WHEN pdates_private_label > 0 THEN 1 ELSE 0 END as has_pdates_private_label,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted_private_label,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions_on_private_labels > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemptions > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_manuf_coupon_redemptions,
# MAGIC       -- CASE WHEN pdates_instore_discount_applied > 0 THEN 1 ELSE 0 END as has_pdates_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted_instore_discount_applied,
# MAGIC       -- CASE WHEN pdates_private_label_instore_discount_applied > 0 THEN 1 ELSE 0 END as has_pdates_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemption_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC   
# MAGIC       -- purchase date features
# MAGIC       CASE WHEN pdates_campaign_targeted > 0 THEN pdates_campaign_targeted/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted,
# MAGIC       pdates_private_label/purchase_dates as pdates_private_label,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label > 0 THEN pdates_campaign_targeted_private_label/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted_private_label,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions > 0 THEN pdates_campaign_coupon_redemptions/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions_on_private_labels > 0 THEN pdates_campaign_coupon_redemptions_on_private_labels/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemptions > 0 THEN pdates_manuf_coupon_redemptions/purchase_dates 
# MAGIC         ELSE NULL END as pdates_manuf_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_targeted_instore_discount_applied > 0 THEN pdates_campaign_targeted_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted_instore_discount_applied,
# MAGIC       pdates_private_label_instore_discount_applied/purchase_dates as pdates_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label_instore_discount_applied > 0 THEN pdates_campaign_targeted_private_label_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_instore_discount_applied > 0 THEN pdates_campaign_coupon_redemption_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied > 0 THEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemption_instore_discount_applied > 0 THEN pdates_manuf_coupon_redemption_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC       
# MAGIC       -- list amount features
# MAGIC       CASE WHEN pdates_campaign_targeted > 0 THEN amount_list_with_campaign_targeted/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted,
# MAGIC       amount_list_with_private_label/amount_list as amount_list_with_private_label,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label > 0 THEN amount_list_with_campaign_targeted_private_label/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted_private_label,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions > 0 THEN amount_list_with_campaign_coupon_redemptions/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions_on_private_labels > 0 THEN amount_list_with_campaign_coupon_redemptions_on_private_labels/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemptions_on_private_labels,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemptions > 0 THEN amount_list_with_manuf_coupon_redemptions/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_manuf_coupon_redemptions,
# MAGIC       amount_list_with_instore_discount_applied/amount_list as amount_list_with_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_instore_discount_applied > 0 THEN amount_list_with_campaign_targeted_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted_instore_discount_applied,
# MAGIC       amount_list_with_private_label_instore_discount_applied/amount_list as amount_list_with_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label_instore_discount_applied > 0 THEN amount_list_with_campaign_targeted_private_label_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_instore_discount_applied > 0 THEN amount_list_with_campaign_coupon_redemption_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied > 0 THEN amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemption_instore_discount_applied > 0 THEN amount_list_with_manuf_coupon_redemption_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_manuf_coupon_redemption_instore_discount_applied
# MAGIC 
# MAGIC FROM household_metrics
# MAGIC ORDER BY household_id;

# COMMAND ----------

# DBTITLE 1,Read Features to Pandas
# retreive as Spark dataframe
household_features = (
  spark
    .table('household_features')
  )

# retrieve as pandas Dataframe
household_features_pd = household_features.toPandas()

# get household ids from dataframe
households_pd = household_features_pd[['household_id']]

# remove household ids from dataframe
features_pd = household_features_pd.drop(['household_id'], axis=1)

features_pd

# COMMAND ----------

# MAGIC %md With our features separated, let's look again at our feature distributions.  We'll start by examining our new binary features:

# COMMAND ----------

# DBTITLE 1,Examine Distribution of Binary Features
b_feature_names = list(filter(lambda f:f[0:4]==('has_') , features_pd.columns))
b_feature_count = len(b_feature_names)

# determine required rows and columns
b_column_count = 5
b_row_count = math.ceil(b_feature_count / b_column_count)

# configure figure layout
fig, ax = plt.subplots(b_row_count, b_column_count, figsize =(b_column_count * 3.5, b_row_count * 3.5))

# render distribution of each feature
for k in range(0,b_feature_count):
  
  # determine row & col position
  b_col = k % b_column_count
  b_row = int(k / b_column_count)
  
  # determine feature to be plotted
  f = b_feature_names[k]
  
  value_counts = features_pd[f].value_counts()

  # render pie chart
  ax[b_row][b_col].pie(
    x = value_counts.values,
    labels = value_counts.index,
    explode = None,
    autopct='%1.1f%%',
    labeldistance=None,
    #pctdistance=0.4,
    frame=True,
    radius=0.48,
    center=(0.5, 0.5)
    )
  
  # clear frame of ticks
  ax[b_row][b_col].set_xticks([])
  ax[b_row][b_col].set_yticks([])
  
  # legend & feature name
  ax[b_row][b_col].legend(bbox_to_anchor=(1.04,1.05),loc='upper left', fontsize=8)
  ax[b_row][b_col].text(1.04,0.8, s=b_feature_names[k].replace('_','\n'), fontsize=8, va='top')

# COMMAND ----------

# MAGIC %md From the pie charts, it appears many promotional offers are not acted upon. This is typical for most promotional offers, especially those associated with coupons. Individually, we see low uptake on many promotional offers, but when we examine the uptake of multiple promotional offers in combination with each other, the frequency of uptake drops to levels where we might consider ignoring the offers in combination, instead focusing on them individually. We'll hold off on addressing that to turn our attention to our continuous features, many of which are now corrected for zero-inflation:

# COMMAND ----------

# DBTITLE 1,Examine Distribution of Continuous Features
c_feature_names = list(filter(lambda f:f[0:4]!=('has_') , features_pd.columns))
c_feature_count = len(c_feature_names)

# determine required rows and columns
c_column_count = 5
c_row_count = math.ceil(c_feature_count / c_column_count)

# configure figure layout
fig, ax = plt.subplots(c_row_count, c_column_count, figsize =(c_column_count * 4.5, c_row_count * 3))

# render distribution of each feature
for k in range(0, c_feature_count):
  
  # determine row & col position
  c_col = k % c_column_count
  c_row = int(k / c_column_count)
  
  # determine feature to be plotted
  f = c_feature_names[k]
  
  # set figure at row & col position
  ax[c_row][c_col].hist(features_pd[c_feature_names[k]], rwidth=0.95, bins=10) # histogram
  ax[c_row][c_col].set_xlim(0,1)   # set x scale 0 to 1
  ax[c_row][c_col].set_ylim(0,features_pd.shape[0]) # set y scale 0 to 2500 (household count)
  ax[c_row][c_col].text(x=0.1, y=features_pd.shape[0]-100, s=c_feature_names[k].replace('_','\n'), fontsize=9, va='top')      # feature name in chart

# COMMAND ----------

# MAGIC %md With the zeros removed from many of our problem features, we now have more standard distributions.  That said, may of those distributions are non-normal (not Gaussian), and Gaussian distributions could be really helpful with many clustering techniques.
# MAGIC 
# MAGIC One way to make these distributions more normal is to apply the Box-Cox transformation.  In our application of this transformation to these features (not shown), we found that many of the distributions failed to become much more normal than what is shown here.  So, we'll make use of another transformation which is a bit more assertive, the [quantile transformation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html#sklearn.preprocessing.quantile_transform).
# MAGIC 
# MAGIC The quantile transformation calculates the cumulative probability function associated with the data points for a given feature.  This is a fancy way to say that the data for a feature are sorted and a function for calculating the percent rank of a value within the range of observed values is calculated. That percent ranking function provides the basis of mapping the data to a well-known distribution such as a normal distribution. The [exact math](https://www.sciencedirect.com/science/article/abs/pii/S1385725853500125) behind this transformation doesn't have to be fully understood for the utility of this transformation to be observed.  If this is your first introduction to quantile transformations, just know the technique has been around since the 1950s and is heavily used in many academic disciplines:

# COMMAND ----------

# DBTITLE 1,Apply Quantile Transformation to Continuous Features
# access continous features
c_features_pd = features_pd[c_feature_names]

# apply quantile transform
qc_features_pd = pd.DataFrame(
  quantile_transform(c_features_pd, output_distribution='normal', ignore_implicit_zeros=True),
  columns=c_features_pd.columns,
  copy=True
  )

# show transformed data
qc_features_pd

# COMMAND ----------

# DBTITLE 1,Examine Distribution of Quantile-Transformed Continuous Features
qc_feature_names = qc_features_pd.columns
qc_feature_count = len(qc_feature_names)

# determine required rows and columns
qc_column_count = 5
qc_row_count = math.ceil(qc_feature_count / qc_column_count)

# configure figure layout
fig, ax = plt.subplots(qc_row_count, qc_column_count, figsize =(qc_column_count * 5, qc_row_count * 4))

# render distribution of each feature
for k in range(0,qc_feature_count):
  
  # determine row & col position
  qc_col = k % qc_column_count
  qc_row = int(k / qc_column_count)
  
  # set figure at row & col position
  ax[qc_row][qc_col].hist(qc_features_pd[qc_feature_names[k]], rwidth=0.95, bins=10) # histogram
  #ax[qc_row][qc_col].set_xlim(0,1)   # set x scale 0 to 1
  ax[qc_row][qc_col].set_ylim(0,features_pd.shape[0]) # set y scale 0 to 2500 (household count)
  ax[qc_row][qc_col].text(x=0.1, y=features_pd.shape[0]-100, s=qc_feature_names[k].replace('_','\n'), fontsize=9, va='top')      # feature name in chart

# COMMAND ----------

# MAGIC %md It's important to note that as powerful as the quantile transformation is, it does not magically solve all data problems.  In developing this notebook, we identified several features after transformation where there appeared to be a bimodal distribution to the data.  These features were ones for which we had initially decided not to apply the zero-inflated distribution correction.  Returning to our feature definitions, implementing the correction and rerunning the transform solved the problem for us. That said, we did not correct every transformed distribution where there is a small group of households positioned to the far-left of the distribution.  We decided that we would address only those where about 250+ households fell within that bin.

# COMMAND ----------

# MAGIC %md ## Step 3: Examine Relationships
# MAGIC 
# MAGIC Now that we have our continuous features aligned with a normal distribution, let's examine the relationship between our feature variables, starting with our continuous features.  Using standard correlation, we can see we have a large number of highly related features.  The multicollinearity captured here, if not addressed, will cause our clustering to overemphasize some aspects of promotion response to the diminishment of others:

# COMMAND ----------

# DBTITLE 1,Examine Relationships between Continuous Features
# generate correlations between features
qc_features_corr = qc_features_pd.corr()

# assemble a mask to remove top-half of heatmap
top_mask = np.zeros(qc_features_corr.shape, dtype=bool)
top_mask[np.triu_indices(len(top_mask))] = True

# define size of heatmap (for large number of features)
plt.figure(figsize=(10,8))

# generate heatmap
hmap = sns.heatmap(
  qc_features_corr,
  cmap = 'coolwarm',
  vmin =  1.0, 
  vmax = -1.0,
  mask = top_mask
  )

# COMMAND ----------

# MAGIC %md And what about relationships between our binary features?  Pearson's correlation (used in the heatmap above), doesn't produce valid results when dealing with categorical data. So instead, we'll calculate [Theil's Uncertainty Coefficient](https://en.wikipedia.org/wiki/Uncertainty_coefficient), a metric designed to examine to what degree the value of one binary measure predicts another.  Theil's U falls within a range between 0, where there is no predictive value between the variables, and 1, where there is perfect predictive value. What's really interesting about this metric is that it is **asymmetric** so that the score shows for one binary measure predicts the other but not necessarily the other way around.  This will mean we need to carefully examine the scores in the heatmap below and not assume a symmetry in output around the diagonal:
# MAGIC 
# MAGIC **NOTE** The primary author of the *dython* package from which we are taking the metric calculation has [an excellent article](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9) discussing Theil's U and related metrics.

# COMMAND ----------

# DBTITLE 1,Examine Relationships between Binary Features
# generate heatmap with Theil's U
_ = dython.nominal.associations(
  features_pd[b_feature_names], 
  nominal_columns='all',
  #theil_u=True,
  figsize=(10,8),
  cmap='coolwarm',
  vmax=1.0,
  vmin=0.0,
  cbar=False
  )

# COMMAND ----------

# MAGIC %md As with our continuous features, we have some problematic relationships between our binary variables that we need to address.  And what about the relationship between the continous and categorical features? 
# MAGIC 
# MAGIC We know from how they were derived that a binary feature with a value of 0 will have a NULL/NaN value for its related continuous features and that any real value for a continuous feature will translate into a value of 1 for the associated binary feature. We don't need to calculate a metric to know we have a relationship between these features (though the calculation of a [Correlation Ratio](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9) might help us if we had any doubts).  So what are we going to do to address these and the previously mentioned relationships in our feature data?
# MAGIC 
# MAGIC When dealing with a large number of features, these relationships are typically addressed using dimension reduction techniques. These techniques project the data in such a way that the bulk of the variation in the data is captured by a smaller number of features.  Those features, often referred to as latent factors or principal components (depending on the technique employed) capture the underlying structure of the data that is reflected in the surface-level features, and they do so in a way that the overlapping explanatory power of the features, *i.e.* the multi-collinearity, is removed.
# MAGIC 
# MAGIC So which dimension reduction technique should we use?  **Principal Components Analysis (PCA)** is the most popular of these techniques but it can only be applied to datasets comprised of continuous features. **Mixed Component Analysis (MCA)** is another of these techniques but it can only be applied to datasets with categorical features. **Factor Analysis of Mixed Data (FAMD)** allows us to combine concepts from these two techniques to construct a reduced feature set when our data consists of both continuous and categorical data.  That said, we have a problem with applying FAMD to our feature data.
# MAGIC 
# MAGIC Typical implementations of both PCA and MCA (and therefore FAMD) require that no missing data values be present in the data.  Simple imputation using mean or median values for continuous features and frequently occurring values for categorical features will not work as the dimension reduction techniques key into the variation in the dataset, and these simple imputations fundamentally alter it. (For more on this, please check out [this excellent video](https://www.youtube.com/watch?v=OOM8_FH6_8o&feature=youtu.be). The video is focused on PCA but the information provided is applicable across all these techniques.)
# MAGIC 
# MAGIC In order to impute the data correctly, we need to examine the distribution of the existing data and leverage relationships between features to impute appropriate values from that distribution in a way that doesn't alter the projections. Work in this space is fairly nacent, but some Statisticians have worked out the mechanics for not only PCA and MCA but also FAMD.  Our challenge is that there are no libraries implementing these techniques in Python, but there are packages for this in R.
# MAGIC 
# MAGIC So now we need to get our data over to R.  To do this, let's our data as a temporary view with the Spark SQL engine.  This will allow us to query this data from R:

# COMMAND ----------

# DBTITLE 1,Register Transformed Data as Spark DataFrame
# assemble full dataset with transformed features
trans_features_pd = pd.concat([ 
  households_pd,  # add household IDs as supplemental variable
  qc_features_pd, 
  features_pd[b_feature_names].astype(str)
  ], axis=1)

# send dataset to spark as temp table
spark.createDataFrame(trans_features_pd).createOrReplaceTempView('trans_features_pd')

# COMMAND ----------

# MAGIC %md We will now prepare our R environment by loading the packages required for our work.  The [FactoMineR](https://www.rdocumentation.org/packages/FactoMineR/versions/2.4) package provides us with the required FAMD functionality while the [missMDA](https://www.rdocumentation.org/packages/missMDA/versions/1.18) package provides us with imputation capabilities:

# COMMAND ----------

# DBTITLE 1,Install Required R Packages
# MAGIC %r
# MAGIC 
# MAGIC require(devtools)
# MAGIC install_version("dplyr", version="1.0.3")
# MAGIC install.packages( c( "pbkrtest", "FactoMineR", "missMDA", "factoextra") )

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages('rlang')
# MAGIC quit

# COMMAND ----------

# MAGIC %md And now we can pull our data into R.  Notice that we retrieve the data to a SparkR DataFrame before collecting it to a local R data frame:

# COMMAND ----------

# DBTITLE 1,Retrieve Spark Data to R Data Frame
# MAGIC %r
# MAGIC 
# MAGIC # retrieve data from from Spark
# MAGIC library(SparkR)
# MAGIC df.spark <- SparkR::sql("SELECT * FROM trans_features_pd")
# MAGIC 
# MAGIC # move data to R data frame
# MAGIC df.r <- SparkR::collect(df.spark)
# MAGIC 
# MAGIC summary(df.r)

# COMMAND ----------

# MAGIC %md Looks like the data came across fine, but we need to examine how the binary features have been translated.  FactoMiner and missMDA require that categorical features be identified as [*factor* types](https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/factor) and here we can see that they are coming across as characters:

# COMMAND ----------

# DBTITLE 1,Examine the R Data Frame's Structure
# MAGIC %r
# MAGIC 
# MAGIC str(df.r)

# COMMAND ----------

# MAGIC %md To convert our categorical features to factors, we apply a quick conversion:

# COMMAND ----------

# DBTITLE 1,Convert Categorical Features to Factors
# MAGIC %r
# MAGIC library(dplyr)
# MAGIC df.mutated <- mutate_if(df.r, is.character, as.factor)
# MAGIC 
# MAGIC str(df.mutated)

# COMMAND ----------

# MAGIC %md Now that the data is structured the right way for our analysis, we can begin the work of performing FAMD. Our first step is to determine the number of principal components required. The missMDA package provides the *estim_ncpFAMD* method for just this purpose, but please note that this routine **takes a long time to complete**.  We've include the code we used to run it but have commented it out and replaced it with the result it eventually landed upon during our run:

# COMMAND ----------

# DBTITLE 1,Determine Number of Components
# MAGIC %r
# MAGIC 
# MAGIC library(missMDA)
# MAGIC 
# MAGIC # determine number of components to produce
# MAGIC #nb <- estim_ncpFAMD(df.mutated, ncp.max=10, sup.var=1)
# MAGIC nb <- list( c(8) ) 
# MAGIC names(nb) <- c("ncp")
# MAGIC 
# MAGIC # display optimal number of components
# MAGIC nb$ncp

# COMMAND ----------

# MAGIC %md With the number of principal components determined, we can now impute the missing values.  Please note that FAMD, like both PCA and MCA, require features to be standardized.  The mechanisms for this differs based on whether a feature is continuous or categorical.  The *imputeFAMD* method provides functionality to tackle this with appropriate setting of the *scale* argument:

# COMMAND ----------

# DBTITLE 1,Impute Missing Values & Perform FAMD Transformation
# MAGIC %r 
# MAGIC 
# MAGIC # impute missing values
# MAGIC library(missMDA)
# MAGIC 
# MAGIC res.impute <- imputeFAMD(
# MAGIC   df.mutated,      # dataset with categoricals organized as factors
# MAGIC   ncp=nb$ncp,      # number of principal components
# MAGIC   scale=True,      # standardize features
# MAGIC   max.iter=10000,  # iterations to find optimal solution
# MAGIC   sup.var=1        # ignore the household_id field (column 1)
# MAGIC   ) 
# MAGIC 
# MAGIC # perform FAMD
# MAGIC library(FactoMineR)
# MAGIC 
# MAGIC res.famd <- FAMD(
# MAGIC   df.mutated,     # dataset with categoricals organized as factors
# MAGIC   ncp=nb$ncp,     # number of principal components
# MAGIC   tab.disj=res.impute$tab.disj, # imputation matrix from prior step
# MAGIC   sup.var=1,       # ignore the household_id field (column 1)
# MAGIC   graph=FALSE
# MAGIC )

# COMMAND ----------

# MAGIC %md Each principal component generated by the FAMD accounts for a percent of the variance found in the overall dataset.  The percent for each principal component, identified as dimensions 1 through 8, are captured in the FAMD output along with the cumulative variance accounted for by the principal components:

# COMMAND ----------

# DBTITLE 1,Plot Variance Captured by Components
# MAGIC %r
# MAGIC 
# MAGIC library("ggplot2")
# MAGIC library("factoextra")
# MAGIC 
# MAGIC eig.val <- get_eigenvalue(res.famd)
# MAGIC print(eig.val)

# COMMAND ----------

# MAGIC %md Reviewing this output, we can see that the first two dimensions (principal components) account for about 50% of the variance, allowing us to get a sense of the structure of our data through a 2-D visualization:

# COMMAND ----------

# DBTITLE 1,Visualize Households Leveraging First Two Components
# MAGIC %r
# MAGIC 
# MAGIC fviz_famd_ind(
# MAGIC   res.famd, 
# MAGIC   axes=c(1,2),  # use principal components 1 & 2
# MAGIC   geom = "point",  # show just the points (households)
# MAGIC   col.ind = "cos2", # color points (roughly) by the degree to which the principal component predicts the instance
# MAGIC   gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
# MAGIC   alpha.ind=0.5
# MAGIC   )

# COMMAND ----------

# MAGIC %md Graphing our households by the first and second principal components indicates there may be some nice clusters of households within the data (as indicated by the grouping patterns in the chart). At a high-level, our data may indicate a couple large, we'll separated clusters, while at a lower-level, there may be some finer-grained clusters with overlapping boundaries within the larger groupings.
# MAGIC 
# MAGIC There are [many other types of visualization and analyses we can perform](http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/115-famd-factor-analysis-of-mixed-data-in-r-essentials/) on the FAMD results to gain a better understanding of how our base features are represented in each of the principal components, but we've got what we need for the purpose of clustering. We will now focus on getting the data from R and back into Python.
# MAGIC 
# MAGIC To get started, let's retrieve principal component values for each of our households:

# COMMAND ----------

# DBTITLE 1,Retrieve Household-Specific Values for Principal Components (Eigenvalues)
# MAGIC %r
# MAGIC 
# MAGIC df.famd <- bind_cols(
# MAGIC   dplyr::select(df.r, "household_id"), 
# MAGIC   as.data.frame( res.famd$ind$coord ) 
# MAGIC   )
# MAGIC 
# MAGIC head(df.famd)

# COMMAND ----------

# DBTITLE 1,Persist Eigenvalues to Delta
# MAGIC %r
# MAGIC 
# MAGIC df.out <- createDataFrame(df.famd)
# MAGIC 
# MAGIC write.df(df.out, source = "delta", path = "/tmp/completejourney/silver/features_finalized", mode="overwrite", overwriteSchema="true")

# COMMAND ----------

# DBTITLE 1,Retrieve Eigenvalues in Python
display(
  spark.table('DELTA.`/tmp/completejourney/silver/features_finalized/`')
  )

# COMMAND ----------

# MAGIC %md And now let's examine the relationships between these features:

# COMMAND ----------

# DBTITLE 1,Examine Relationships between Reduced Dimensions
# generate correlations between features
famd_features_corr = spark.table('DELTA.`/tmp/completejourney/silver/features_finalized/`').drop('household_id').toPandas().corr()

# assemble a mask to remove top-half of heatmap
top_mask = np.zeros(famd_features_corr.shape, dtype=bool)
top_mask[np.triu_indices(len(top_mask))] = True

# define size of heatmap (for large number of features)
plt.figure(figsize=(10,8))

# generate heatmap
hmap = sns.heatmap(
  famd_features_corr,
  cmap = 'coolwarm',
  vmin =  1.0, 
  vmax = -1.0,
  mask = top_mask
  )

# COMMAND ----------

# MAGIC %md With multicollinearity addressed through our reduced feature set, we can now proceed with clustering.
