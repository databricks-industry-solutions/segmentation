# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/segmentation.git. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-segmentation.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to better understand the clusters generated in the prior notebook leveraging some standard profiling techniques. 

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import mlflow

import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic

import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from pyspark.sql.functions import expr

# COMMAND ----------

# MAGIC %md ## Step 1: Assemble Segmented Dataset
# MAGIC 
# MAGIC We now have clusters but we're not really clear on what exactly they represent.  The feature engineering work we performed to avoid problems with the data that might lead us to invalid or inappropriate solutions have made the data very hard to interpret.  
# MAGIC 
# MAGIC To address this problem, we'll retrieve the cluster labels (assigned to each household) along with the original features associated with each:

# COMMAND ----------

# DBTITLE 1,Retrieve Features & Labels
# retrieve features and labels
spark.sql("USE journey")
household_basefeatures = spark.table('household_features')
household_finalfeatures = spark.table('DELTA.`/tmp/completejourney/silver/features_finalized/`')
labels = spark.table('DELTA.`/tmp/completejourney/gold/household_clusters/`')

# assemble labeled feature sets
labeled_basefeatures_pd = (
  labels
    .join(household_basefeatures, on='household_id')
  ).toPandas()

labeled_finalfeatures_pd = (
  labels
    .join(household_finalfeatures, on='household_id')
  ).toPandas()

# get name of all non-feature columns
label_columns = labels.columns

labeled_basefeatures_pd

# COMMAND ----------

# MAGIC %md Before proceeding with our analysis of these data, let's set a few variables that will be used to control the remainder of our analysis.  We have multiple cluster designs but for this notebook, we will focus our attention on the results from our hierarchical clustering model:

# COMMAND ----------

# DBTITLE 1,Set Cluster Design to Analyze
cluster_column = 'hc_cluster'
cluster_count = len(np.unique(labeled_finalfeatures_pd[cluster_column]))
cluster_colors = [cm.nipy_spectral(float(i)/cluster_count) for i in range(cluster_count)]

# COMMAND ----------

# MAGIC %md ## Step 2: Profile Segments
# MAGIC 
# MAGIC To get us started, let's revisit the 2-dimensional visualization of our clusters to get us oriented to the clusters.  The color-coding we use in this chart will be applied across our remaining visualizations to make it easier to determine the cluster being explored:

# COMMAND ----------

# DBTITLE 1,Visualize Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=labeled_finalfeatures_pd,
  x='Dim_1',
  y='Dim_2',
  hue=cluster_column,
  palette=cluster_colors,
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md  The segment design we came up with does not produce equal sized groupings.  Instead, we have one group a bit larger than the others, though the smaller groups are still of a size where they are useful to our team:

# COMMAND ----------

# DBTITLE 1,Count Cluster Members
# count members per cluster
cluster_member_counts = labeled_finalfeatures_pd.groupby([cluster_column]).agg({cluster_column:['count']})
cluster_member_counts.columns = cluster_member_counts.columns.droplevel(0)

# plot counts
plt.bar(
  cluster_member_counts.index,
  cluster_member_counts['count'],
  color = cluster_colors,
  tick_label=cluster_member_counts.index
  )

# stretch y-axis
plt.ylim(0,labeled_finalfeatures_pd.shape[0])

# labels
for index, value in zip(cluster_member_counts.index, cluster_member_counts['count']):
    plt.text(index, value, str(value)+'\n', horizontalalignment='center', verticalalignment='baseline')

# COMMAND ----------

# MAGIC %md Let's now examine how each segment differs relative to our base features.  For our categorical features, we'll plot the proportion of cluster members identified as participating in a specific promotional activity relative to the overall number of cluster members. For our continuous features, we will visualize values using a whisker plot:

# COMMAND ----------

# DBTITLE 1,Define Function to Render Plots
def profile_segments_by_features(data, features_to_plot, label_to_plot, label_count, label_colors):
  
    feature_count = len(features_to_plot)
    
    # configure plot layout
    max_cols = 5
    if feature_count > max_cols:
      column_count = max_cols
    else:
      column_count = feature_count      
      
    row_count = math.ceil(feature_count / column_count)

    fig, ax = plt.subplots(row_count, column_count, figsize =(column_count * 4, row_count * 4))
    
    # for each feature (enumerated)
    for k in range(feature_count):

      # determine row & col position
      col = k % column_count
      row = int(k / column_count)
      
      # get axis reference (can be 1- or 2-d)
      try:
        k_ax = ax[row,col]
      except:
        pass
        k_ax = ax[col]
      
      # set plot title
      k_ax.set_title(features_to_plot[k].replace('_',' '), fontsize=7)

      # CATEGORICAL FEATURES
      if features_to_plot[k][:4]=='has_': 

        # calculate members associated with 0/1 categorical values
        x = data.groupby([label_to_plot,features_to_plot[k]]).agg({label_to_plot:['count']})
        x.columns = x.columns.droplevel(0)

        # for each cluster
        for c in range(label_count):

          # get count of cluster members
          c_count = x.loc[c,:].sum()[0]

          # calculate members with value 0
          try:
            c_0 = x.loc[c,0]['count']/c_count
          except:
            c_0 = 0

          # calculate members with value 1
          try:
            c_1 = x.loc[c,1]['count']/c_count
          except:
            c_1 = 0

          # render percent stack bar chart with 1s on bottom and 0s on top
          k_ax.set_ylim(0,1)
          k_ax.bar([c], c_1, color=label_colors[c], edgecolor='white')
          k_ax.bar([c], c_0, bottom=c_1, color=label_colors[c], edgecolor='white', alpha=0.25)


      # CONTINUOUS FEATURES
      else:    

        # get subset of data with entries for this feature
        x = data[
              ~np.isnan(data[features_to_plot[k]])
              ][[label_to_plot,features_to_plot[k]]]

        # get values for each cluster
        p = []
        for c in range(label_count):
          p += [x[x[label_to_plot]==c][features_to_plot[k]].values]

        # plot values
        k_ax.set_ylim(0,1)
        bplot = k_ax.boxplot(
            p, 
            labels=range(label_count),
            patch_artist=True
            )

        # adjust box fill to align with cluster
        for patch, color in zip(bplot['boxes'], label_colors):
          patch.set_alpha(0.75)
          patch.set_edgecolor('black')
          patch.set_facecolor(color)
    

# COMMAND ----------

# DBTITLE 1,Render Plots for All Base Features
# get feature names
feature_names = labeled_basefeatures_pd.drop(label_columns, axis=1).columns

# generate plots
profile_segments_by_features(labeled_basefeatures_pd, feature_names, cluster_column, cluster_count, cluster_colors)

# COMMAND ----------

# MAGIC %md There's a lot to examine in this plot but the easiest thing seems to be to start with the categorical features to identify groups responsive to some promotional offers and not others.  The continuous features then provide a bit more insight into the degree of engagement when that cluster does respond.  
# MAGIC 
# MAGIC As you work your way through the various features, you will likely start to form descriptions of the different clusters.  To assist with that, it might help to retrieve specific subsets of features to focus your attention on a smaller number of features:

# COMMAND ----------

# DBTITLE 1,Plot Subset of Features
feature_names = ['has_pdates_campaign_targeted', 'pdates_campaign_targeted', 'amount_list_with_campaign_targeted']

profile_segments_by_features(labeled_basefeatures_pd, feature_names, cluster_column, cluster_count, cluster_colors)

# COMMAND ----------

# MAGIC %md ## Step 3: Describe Segments
# MAGIC 
# MAGIC With close examination of the features you should hopefully come to differentiate the clusters in terms of their behavior.  Now it becomes interesting to examine why these groups might exist and/or how we might be able to identify likely group membership without collecting multiple years of behavioral information. A common way to do this is to examine the clusters in terms of characteristics that were not employed in the cluster design. With this dataset, we might employ demographic information available for a subset of our households for this purpose:

# COMMAND ----------

# DBTITLE 1,Associate Household Demographics with Cluster Labels
labels = spark.table('DELTA.`/tmp/completejourney/gold/household_clusters/`').alias('labels')
demographics = spark.table('households').alias('demographics')

labeled_demos = (
  labels
    .join(demographics, on=expr('labels.household_id=demographics.household_id'), how='leftouter')  # only 801 of 2500 present should match
    .withColumn('matched', expr('demographics.household_id Is Not Null'))
    .drop('household_id')
  ).toPandas()

labeled_demos

# COMMAND ----------

# MAGIC %md Before proceeding, we need to consider how many of our members in cluster have demographic information associated with them:

# COMMAND ----------

# DBTITLE 1,Examine Proportion of Cluster Members with Demographic Data
x = labeled_demos.groupby([cluster_column, 'matched']).agg({cluster_column:['count']})
x.columns = x.columns.droplevel(0)

# for each cluster
for c in range(cluster_count):

  # get count of cluster members
  c_count = x.loc[c,:].sum()[0]

  # calculate members with value 0
  try:
    c_0 = x.loc[c,0]['count']/c_count
  except:
    c_0 = 0

  # calculate members with value 1
  try:
    c_1 = x.loc[c,1]['count']/c_count
  except:
    c_1 = 0
  
  # plot counts
  plt.bar([c], c_1, color=cluster_colors[c], edgecolor='white')
  plt.bar([c], c_0, bottom=c_1, color=cluster_colors[c], edgecolor='white', alpha=0.25)
  plt.xticks(range(cluster_count))
  plt.ylim(0,1)

# COMMAND ----------

# MAGIC %md Ideally, we would have demographic data for all households in the dataset or least for a large, consistent proportion of members across each cluster.  Without that, we need to be cautious about drawing any conclusions from these data.
# MAGIC 
# MAGIC Still, we might continue with the exercise in order to demonstrate technique.  With that in mind, let's construct a contingency table for head of household age-bracket to see how cluster members align around age:

# COMMAND ----------

# DBTITLE 1,Demonstrate Contingency Table
age_by_cluster = sm.stats.Table.from_data(labeled_demos[[cluster_column,'age_bracket']])
age_by_cluster.table_orig

# COMMAND ----------

# MAGIC %md We might then apply Pearson's Chi-squared (*&Chi;^2*) test to determine whether these frequency differences were statistically meaningful.  In such a test, a p-value of less than or equal to 5% would tell us that the frequency distributions were not likely due to chance (and are therefore dependent upon the category assignment):

# COMMAND ----------

# DBTITLE 1,Demonstrate Chi-Squared Test
res = age_by_cluster.test_nominal_association()
res.pvalue

# COMMAND ----------

# MAGIC %md We would then be able to examine the Pearson's residuals associated with the intersection of each cluster and demographic group to determine when specific intersections were driving us to this conclusion.  Intersections with **absolute** residual values of greater than 2 or 4 would differ from expectations with a 95% or 99.9% probability, respectively, and these would likely be the demographic characteristics that would differentiate the clusters:

# COMMAND ----------

# DBTITLE 1,Demonstrate Pearson Residuals
age_by_cluster.resid_pearson  # standard normal random variables within -2, 2 with 95% prob and -4,4 at 99.99% prob

# COMMAND ----------

# MAGIC %md If we had found something meaningful in this data, our next challenge would be to communicate it to members of the team not familiar with these statistical tests.  A popular way for doing this is through a *[mosaic plot](https://www.datavis.ca/papers/casm/casm.html#tth_sEc3)* also known as a *marimekko plot*:

# COMMAND ----------

# DBTITLE 1,Demonstrate Mosaic Plot
# assemble demographic category labels as key-value pairs (limit to matched values)
demo_labels = np.unique(labeled_demos[labeled_demos['matched']]['age_bracket'])
demo_labels_kv = dict(zip(demo_labels,demo_labels))

# define function to generate cell labels
labelizer = lambda key: demo_labels_kv[key[1]]

# define function to generate cell colors
props = lambda key: {'color': cluster_colors[int(key[0])], 'alpha':0.8}

# generate mosaic plot
fig, rect = mosaic(
  labeled_demos.sort_values('age_bracket', ascending=False),
  [cluster_column,'age_bracket'], 
  horizontal=True, 
  axes_label=True, 
  gap=0.015, 
  properties=props, 
  labelizer=labelizer
  )

# set figure size
_ = fig.set_size_inches((10,8))

# COMMAND ----------

# MAGIC %md The proportional display of members associated with each category along with the proportional width of the clusters relative to each other provides a nice way to summarize the frequency differences between these groups. Coupled with statistical analysis, the mosaic plot provides a nice way to make a statistically significant finding more easily comprehended.

# COMMAND ----------

# MAGIC %md ## Step 4: Next Steps
# MAGIC 
# MAGIC Segmentation is rarely a one-and-done exercise. Instead, having learned from this pass with the data, we might repeat the analysis, removing non-differentiating features and possibly including others. In addition, we might perform other analyses such as RFM segmentations or CLV analysis and then examine how these relate to the segmentation design explored here.  Eventually, we may arrive at a new segmentation design, but even if we don't, we have gained insights which may help us better craft promotional campaigns.
