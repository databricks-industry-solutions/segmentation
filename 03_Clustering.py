# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/segmentation.git. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-segmentation.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to identify potential segments for our households using a clustering technique. 

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

import numpy as np
import pandas as pd

import mlflow
import os

from delta.tables import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import seaborn as sns

# COMMAND ----------

# MAGIC %run "./config/Unity Catalog"

# COMMAND ----------

spark.sql(f'USE CATALOG {CATALOG}');
spark.sql(f'USE SCHEMA {SCHEMA}')

# COMMAND ----------

# MAGIC %md ## Step 1: Retrieve Features
# MAGIC
# MAGIC Following the work performed in our last notebook, our households are now identified by a limited number of features that capture the variation found in our original feature set.  We can retrieve these features as follows:

# COMMAND ----------

# DBTITLE 1,Retrieve Transformed Features
# retrieve household (transformed) features
household_X_pd = spark.table('silver_features_finalized').toPandas()

# remove household ids from dataframe
X = household_X_pd.drop(['household_id'], axis=1)

household_X_pd

# COMMAND ----------

# MAGIC %md The exact meaning of each feature is very difficult to articulate given the complex transformations used in their engineering.  Still, they can be used to perform clustering.  (Through profiling which we will perform in our next notebook, we can then retrieve insight into the nature of each cluster.)
# MAGIC
# MAGIC As a first step, let's visualize our data to see if any natural groupings stand out.  Because we are working with a hyper-dimensional space, we cannot perfectly visualize our data but with a 2-D representation (using our first two principal component features), we can see there is a large sizeable cluster in our data and potentially a few additional, more loosely organized clusters:

# COMMAND ----------

# DBTITLE 1,Plot Households
fig, ax = plt.subplots(figsize=(10,8))

_ = sns.scatterplot(
  data=X,
  x='Dim_1',
  y='Dim_2',
  alpha=0.5,
  ax=ax
  )

# COMMAND ----------

# MAGIC %md ## Step 2: K-Means Clustering
# MAGIC
# MAGIC Our first attempt at clustering with make use of the K-means algorithm. K-means is a simple, popular algorithm for dividing instances into clusters around a pre-defined number of *centroids* (cluster centers).  The algorithm works by generating an initial set of points within the space to serve as cluster centers.  Instances are then associated with the nearest of these points to form a cluster, and the true center of the resulting cluster is re-calculated.  The new centroids are then used to re-enlist cluster members, and the process is repeated until a stable solution is generated (or until the maximum number of iterations is exhausted). A quick demonstration run of the algorithm may produce a result as follows:

# COMMAND ----------

# DBTITLE 1,Demonstrate Cluster Assignment
# set up the experiment that mlflow logs runs to: an experiment in the user's personal workspace folder
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/segmentation"
mlflow.set_experiment(experiment_name) 

# initial cluster count
initial_n = 4

# train the model
initial_model = KMeans(
  n_clusters=initial_n,
  max_iter=1000
  )

# fit and predict per-household cluster assignment
init_clusters = initial_model.fit_predict(X)

# combine households with cluster assignments
labeled_X_pd = (
  pd.concat( 
    [X, pd.DataFrame(init_clusters,columns=['cluster'])],
    axis=1
    )
  )

# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=labeled_X_pd,
  x='Dim_1',
  y='Dim_2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / initial_n) for i in range(initial_n)],
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md Our initial model run demonstrates the mechanics of generating a K-means clustering solution, but it also demonstrates some of the shortcomings of the approach.  First, we need to specify the number of clusters.  Setting the value incorrectly can force the creation of numerous smaller clusters or just a few larger clusters, neither of which may reflect what we may observe to be the more immediate and natural structure inherent to the data.
# MAGIC
# MAGIC Second, the results of the algorithm are highly dependent on the centroids with which it is initialized. The use of the K-means++ initialization algorithm addresses some of these problems by better ensuring that initial centroids are dispersed throughout the populated space, but there is still an element of randomness at play in these selections that can have big consequences for our results.
# MAGIC
# MAGIC To begin working through these challenges, we will generate a large number of model runs over a range of potential cluster counts. For each run, we will calculate the sum of squared distances between members and assigned cluster centroids (*inertia*) as well as a secondary metric (*silhouette score*) which provides a combined measure of inter-cluster cohesion and intra-cluster separation (ranging between -1 and 1 with higher values being better). Because of the large number of iterations we will perform, we will distribute this work across our Databricks cluster so that it can be concluded in a timely manner:
# MAGIC
# MAGIC **NOTE** We are using a Spark RDD as a crude means of exhaustively searching our parameter space in a distributed manner. This is an simple technique frequently used for efficient searches over a defined range of values.

# COMMAND ----------

# DBTITLE 1,Iterate over Potential Values of K
# broadcast features so that workers can access efficiently
X_broadcast = sc.broadcast(X)

# function to train model and return metrics
def evaluate_model(n):
  model = KMeans( n_clusters=n, init='k-means++', n_init=1, max_iter=10000)
  clusters = model.fit(X_broadcast.value).labels_
  return n, float(model.inertia_), float(silhouette_score(X_broadcast.value, clusters))


# define number of iterations for each value of k being considered
iterations = (
  spark
    .range(100) # iterations per value of k
    .crossJoin( spark.range(2,21).withColumnRenamed('id','n')) # cluster counts
    .repartition(sc.defaultParallelism)
    .select('n')
    .rdd
    )

# train and evaluate model for each iteration
results_pd = (
  spark
    .createDataFrame(
      iterations.map(lambda n: evaluate_model(n[0])), # iterate over each value of n
      schema=['n', 'inertia', 'silhouette']
      ).toPandas()
    )

# remove broadcast set from workers
X_broadcast.unpersist()

display(results_pd)

# COMMAND ----------

# MAGIC %md Plotting inertia relative to n, *i.e.* the target number of clusters, we can see that the total sum of squared distances between cluster members and cluster centers decreases as we increase the number of clusters in our solution.  Our goal is not to drive inertia to zero (which would be achieved if we made each member the center of its own, 1-member cluster) but instead to identify the point in the curve where the incremental drop in inertia is diminished.  In our plot, we might identify this point as occurring somewhere between 2 and 6 clusters:

# COMMAND ----------

# DBTITLE 1,Inertia over Cluster Count
display(results_pd)

# COMMAND ----------

# MAGIC %md Interpreting the *elbow chart*/*scree plot* of inertia *vs.* n is fairly subjective, and as such, it can be helpful to examine how another metric behaves relative to our cluster count.  Plotting silhouette score relative to n provides us the opportunity to identify a peak (*knee*) beyond which the score declines.  The challenge, as before, is exactly determining the location of that peak, especially given that the silhouette scores for our iterations vary much more than our inertia scores:

# COMMAND ----------

# DBTITLE 1,Silhouette Score over Cluster Count
display(results_pd)

# COMMAND ----------

# MAGIC %md While providing a second perspective, the plot of silhouette scores reinforces the notion that selecting a number of clusters for K-means is a bit subjective.  Domain knowledge coupled with inputs from these and similar charts (such as a chart of the [Gap statistic](https://towardsdatascience.com/k-means-clustering-and-the-gap-statistics-4c5d414acd29)) may help point you towards an optimal cluster count but there are no widely-accepted, objective means of determining this value to date.
# MAGIC
# MAGIC **NOTE** We need to be careful to avoid chasing the highest value for the silhouette score in the knee chart. Higher scores can be obtained with higher values of n by simply pushing outliers into trivially small clusters.
# MAGIC
# MAGIC For our model, we'll go with a value of 2.  Looking at the plot of inertia, there appears to be evidence supporting this value.  Examining the silhouette scores, the clustering solution appears to be much more stable at this value than at values further down the range. To obtain domain knowledge, we might speak with our promotions experts and gain their perspective on not only how different households respond to promotions but what might be a workable number of clusters from this exercise.  But most importantly, from our visualization, the presence of 2 well-separated clusters seems to naturally jump out at us.
# MAGIC
# MAGIC With a value for n identified, we now need to generate a final cluster design.  Given the randomness of the results we obtain from a K-means run (as captured in the widely variable silhouette scores), we might take a *best-of-k* approach to defining our cluster model.  In such an approach, we run through some number of K-means model runs and select the run that delivers the best result as measured by a metric such as silhouette score. To distribute this work, we'll implement a custom function that will allow us to task each worker with finding a best-of-k solution and then take the overall best solution from the results of that work:
# MAGIC
# MAGIC **NOTE** We are again using an RDD to allow us to distribute the work across our cluster.  The *iterations* RDD will hold a value for each iteration to perform.  Using *mapPartitions()* we will determine how many iterations are assigned to a given partition and then force that worker to perform an appropriately configured best-of-k evaluation.  Each partition will send back the best model it could discover and then we will take the best from these.

# COMMAND ----------

# DBTITLE 1,Identify Best of K Model
total_iterations = 50000
n_for_bestofk = 2 
X_broadcast = sc.broadcast(X)

def find_bestofk_for_partition(partition):
   
  # count iterations in this partition
  n_init = sum(1 for i in partition)
  
  # perform iterations to get best of k
  model = KMeans( n_clusters=n_for_bestofk, n_init=n_init, init='k-means++', max_iter=10000)
  model.fit(X_broadcast.value)
  
  # score model
  score = float(silhouette_score(X_broadcast.value, model.labels_))
  
  # return (score, model)
  yield (score, model)


# build RDD for distributed iteration
iterations = sc.range(
              total_iterations, 
              numSlices= sc.defaultParallelism * 4
              ) # distribute work into fairly even number of partitions that allow us to track progress
                        
# retrieve best of distributed iterations
bestofk_results = (
  iterations
    .mapPartitions(find_bestofk_for_partition)
    .sortByKey(ascending=False)
    .take(1)
    )[0]

# get score and model
bestofk_score = bestofk_results[0]
bestofk_model = bestofk_results[1]
bestofk_clusters = bestofk_model.labels_

# print best score obtained
print('Silhouette Score: {0:.6f}'.format(bestofk_score))

# combine households with cluster assignments
bestofk_labeled_X_pd = (
  pd.concat( 
    [X, pd.DataFrame(bestofk_clusters,columns=['cluster'])],
    axis=1
    )
  )
                        
# clean up 
X_broadcast.unpersist()

# COMMAND ----------

# MAGIC %md We can now visualize our results to get a sense of how the clusters align with the structure of our data:

# COMMAND ----------

# DBTITLE 1,Visualize Best of K Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=bestofk_labeled_X_pd,
  x='Dim_1',
  y='Dim_2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / n_for_bestofk) for i in range(n_for_bestofk)],  # align colors with those used in silhouette plots
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md The results of our analysis are not earth-shattering but they don't need to be.  Our data would indicate that for these features we could very reasonably consider our customer households as existing in two fairly distinct groups.  That said, we might want to look at how well individual households sit within these groups, which we can do through a per-instance silhouette chart:
# MAGIC
# MAGIC **NOTE** This code represents a modified version of the [silhouette charts](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) provided in the Sci-Kit Learn documentation.

# COMMAND ----------

# DBTITLE 1,Examine Per-Member Silhouette Scores
# modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

def plot_silhouette_chart(features, labels):
  
  n = len(np.unique(labels))
  
  # configure plot area
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(8, 5)

  # configure plots for silhouette scores between -1 and 1
  ax.set_xlim([-0.1, 1])
  ax.set_ylim([0, len(features) + (n + 1) * 10])
  
  # avg silhouette score
  score = silhouette_score(features, labels)

  # compute the silhouette scores for each sample
  sample_silhouette_values = silhouette_samples(features, labels)

  y_lower = 10

  for i in range(n):

      # get and sort members by cluster and score
      ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
      ith_cluster_silhouette_values.sort()

      # size y based on sample count
      size_cluster_i = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_i

      # pretty up the charts
      color = cm.nipy_spectral(float(i) / n)
      
      ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

      # label the silhouette plots with their cluster numbers at the middle
      ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

      # compute the new y_lower for next plot
      y_lower = y_upper + 10  # 10 for the 0 samples


  ax.set_title("Average silhouette of {0:.3f} with {1} clusters".format(score, n))
  ax.set_xlabel("The silhouette coefficient values")
  ax.set_ylabel("Cluster label")

  # vertical line for average silhouette score of all the values
  ax.axvline(x=score, color="red", linestyle="--")

  ax.set_yticks([])  # clear the yaxis labels / ticks
  ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
  
  return fig, ax

_ = plot_silhouette_chart(X, bestofk_clusters)

# COMMAND ----------

# MAGIC %md From the silhouette chart, we would appear to have one cluster a bit larger than the other.  That cluster appears to be reasonably coherent.  Our other clusters appear to be a bit more dispersed with a more rapid decline in silhouette score values ultimately leading a few members to have negative silhouette scores (indicating overlap with other cluster). 
# MAGIC
# MAGIC This solution may be useful for better understanding customer behavior relative to promotional offers. We'll persist our cluster assignments before examining other clustering techniques:

# COMMAND ----------

# DBTITLE 1,Persist Cluster Assignments
# persist household id and cluster assignment
( 
  spark # bring together household and cluster ids
    .createDataFrame(
       pd.concat( 
          [household_X_pd, pd.DataFrame(bestofk_clusters,columns=['bestofk_cluster'])],
          axis=1
          )[['household_id','bestofk_cluster']]   
      )
    .write  # write data to delta 
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('gold_household_clusters')
  )

# COMMAND ----------

# MAGIC %md ## Step 3: Hierarchical Clustering
# MAGIC
# MAGIC In addition to K-means, hierarchical clustering techniques are frequently used in customer segmentation exercises. With the agglomerative-variants of these techniques, clusters are formed by linking members closest to one another and then linking those clusters to form higher level clusters until a single cluster encompassing all the members of the set is formed.
# MAGIC
# MAGIC Unlike K-means, the agglomerative process is deterministic so that repeated runs on the same dataset lead to the same clustering outcome. So while the hierarchical clustering techniques are frequently criticized for being slower than K-means, the overall processing time to arrive at a particular result may be lessened as no repeat executions of the algorithm are required to arrive at a *best-of* outcome.
# MAGIC
# MAGIC To get a better sense of how this technique works, let's train a hierarchical clustering solution and visualize its output:

# COMMAND ----------

# DBTITLE 1,Function to Plot Dendrogram
# modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

# function to generate dendrogram
def plot_dendrogram(model, **kwargs):

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
                      [model.children_, 
                       model.distances_,
                       counts]
                      ).astype(float)

    # Plot the corresponding dendrogram
    j = 5
    set_link_color_palette(
      [matplotlib.colors.rgb2hex(cm.nipy_spectral(float(i) / j)) for i in range(j)]
      )
    dendrogram(linkage_matrix, **kwargs)

# COMMAND ----------

# DBTITLE 1,Train & Visualize Hierarchical Model
# train cluster model
inithc_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
inithc_model.fit(X)

# generate visualization
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(15, 8)

plot_dendrogram(inithc_model, truncate_mode='level', p=6) # 6 levels max
plt.title('Hierarchical Clustering Dendrogram')
_ = plt.xlabel('Number of points in node (or index of point if no parenthesis)')

# COMMAND ----------

# MAGIC %md The dendrogram is read from the bottom up.  Each initial point represents a cluster consisting of some number of members.  The entire process by which those members come together to form those specific clusters is not visualized (though you can adjust the *p* argument in the *plot_dendrograms* function to see further down into the process).
# MAGIC
# MAGIC As you move up the dendrogram, clusters converge to form new clusters.  The vertical length traversed to reach that point of convergence tells us something about the distance between these clusters.  The longer the length, the wider the gap between the converging clusters.
# MAGIC
# MAGIC The dendrogram gives us a sense of how the overall structure of the dataset comes together but it doesn't steer us towards a specific number of clusters for our ultimate clustering solution.  For that, we need to revert to the plotting of a metric, such as silhouette scores, to identify the appropriate number of clusters for our solution.
# MAGIC
# MAGIC Before plotting silhouette against various numbers of clusters, it's important to examine the means by which clusters are combined to form new clusters.  There are many algorithms (*linkages*) for this.  The SciKit-Learn library supports four of them.  These are:
# MAGIC <p>
# MAGIC * *ward* - link clusters such that the sum of squared distances within the newly formed clusters is minimized
# MAGIC * *average* - link clusters based on the average distance between all points in the clusters
# MAGIC * *single* - link clusters based on the minimum distance between any two points in the clusters
# MAGIC * *complete* - link clusters based on the maximum distance between any two points in the clusters
# MAGIC   
# MAGIC Different linkage mechanisms can result in very different clustering outcomes. Ward's method (denoted by the *ward* linkage mechanism) is considered the go-to for most clustering exercises unless domain knowledge dictates the use of an alternative method:

# COMMAND ----------

# DBTITLE 1,Identify Number of Clusters
results = []

# train models with n number of clusters * linkages
for a in ['ward']:  # linkages
  for n in range(2,21): # evaluate 2 to 20 clusters

    # fit the algorithm with n clusters
    model = AgglomerativeClustering(n_clusters=n, linkage=a)
    clusters = model.fit(X).labels_

    # capture the inertia & silhouette scores for this value of n
    results += [ (n, a, silhouette_score(X, clusters)) ]

results_pd = pd.DataFrame(results, columns=['n', 'linkage', 'silhouette'])
display(results_pd)

# COMMAND ----------

# MAGIC %md The results would indicate our best results may be found using 5 clusters:

# COMMAND ----------

# DBTITLE 1,Train & Evaluate Model
n_for_besthc = 5
linkage_for_besthc = 'ward'
 
# configure model
besthc_model = AgglomerativeClustering( n_clusters=n_for_besthc, linkage=linkage_for_besthc)

# train and predict clusters
besthc_clusters = besthc_model.fit(X).labels_

# score results
besthc_score = silhouette_score(X, besthc_clusters)

# print best score obtained
print('Silhouette Score: {0:.6f}'.format(besthc_score))

# combine households with cluster assignments
besthc_labeled_X_pd = (
  pd.concat( 
    [X, pd.DataFrame(besthc_clusters,columns=['cluster'])],
    axis=1
    )
  )

# COMMAND ----------

# MAGIC %md Visualizing these clusters, we can see how groupings reside within the data structure.  In our initial visualization of the features, we argued that there were two high-level clusters that stood out (and our K-means algorithm seemed to pick this up very well).  Here, our hierarchical clustering algorithm seems to have picked up on the looser subclusters a bit better, though it also seems to have picked up on some loosely organized households for one very small cluster:

# COMMAND ----------

# DBTITLE 1,Visualize Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=besthc_labeled_X_pd,
  x='Dim_1',
  y='Dim_2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / n_for_besthc) for i in range(n_for_besthc)],  # align colors with those used in silhouette plots
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md Our per-instance silhouette scores show us we have a bit more overlap between clusters when examined at this level.  One of the clusters has so few members it doesn't seem worth keeping it, especially when we review the 2-D visualization and see that these points seem to be highly intermixed with other clusters (at least when viewed from this perspective):

# COMMAND ----------

# DBTITLE 1,Examine Per-Member Silhouette Scores
_ = plot_silhouette_chart(X, besthc_clusters)

# COMMAND ----------

# MAGIC %md With that in mind, we'll retrain our model with a cluster count of 4 and then persist those results:

# COMMAND ----------

# DBTITLE 1,ReTrain & Evaluate Model
n_for_besthc = 4
linkage_for_besthc = 'ward'
 
# configure model
besthc_model = AgglomerativeClustering( n_clusters=n_for_besthc, linkage=linkage_for_besthc)

# train and predict clusters
besthc_clusters = besthc_model.fit(X).labels_

# score results
besthc_score = silhouette_score(X, besthc_clusters)

# print best score obtained
print('Silhouette Score: {0:.6f}'.format(besthc_score))

# combine households with cluster assignments
besthc_labeled_X_pd = (
  pd.concat( 
    [X, pd.DataFrame(besthc_clusters,columns=['cluster'])],
    axis=1
    )
  )

# COMMAND ----------

# DBTITLE 1,Visualize Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=besthc_labeled_X_pd,
  x='Dim_1',
  y='Dim_2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / n_for_besthc) for i in range(n_for_besthc)],  # align colors with those used in silhouette plots
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# DBTITLE 1,Examine Per-Member Silhouette Scores
_ = plot_silhouette_chart(X, besthc_clusters)

# COMMAND ----------

# DBTITLE 1,Add Field to Hold Hierarchical Cluster Assignment
# add column to previously created table to allow assignment of cluster ids
# try/except used here in case this statement is being rurun against a table with field already in place
try:
  spark.sql('ALTER TABLE gold_household_clusters ADD COLUMN (hc_cluster integer)')
except:
  pass  

# COMMAND ----------

# DBTITLE 1,Update Persisted Data to Hold Hierarchical Cluster Assignment
# assemble household IDs and new cluster IDs
updates = (
  spark
    .createDataFrame(
       pd.concat( 
          [household_X_pd, pd.DataFrame(besthc_clusters,columns=['hc_cluster'])],
          axis=1
          )[['household_id','hc_cluster']]   
      )
  )

# merge new cluster ID data with existing table  
deltaTable = DeltaTable.forName(spark, "gold_household_clusters")

(
  deltaTable.alias('target')
    .merge(
      updates.alias('source'),
      'target.household_id=source.household_id'
      )
    .whenMatchedUpdate(set = { 'hc_cluster' : 'source.hc_cluster' } )
    .execute()
  )

# COMMAND ----------

# MAGIC %md ## Step 4: Other Techniques
# MAGIC
# MAGIC We have only begun to scratch the surface on the clustering techniques available to us.  [K-Medoids](https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.cluster.KMedoids.html), a variation of K-means which centers clusters on actual members in the dataset, allows for alternative methods (other than just Euclidean distance) of considering member similarities and may be more robust to noise and outliers in a dataset. [Density-Based Spatial Clustering of Applications with Noise (DBSCAN)](https://scikit-learn.org/stable/modules/clustering.html#dbscan) is another interesting clustering technique which identifies clusters in areas of high member density while ignoring dispersed members in lower-density regions. This would seem to be a good technique for this dataset but in our examination of DBSCAN (not shown), we had difficulty tuning the *epsilon* and *minimum sample count* parameters (that control how high-density regions are identified) to produce a high-quality clustering solution. And [Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html#gaussian-mixture-models) offer still another approach popular in segmentation exercises which allows clusters with non-spherical shapes to be more easily formed.
# MAGIC
# MAGIC In addition to alternative algorithms, there is emerging work in the development of cluster ensemble models (aka *consensus clustering*). First introduced by [Monti *et al.*](https://link.springer.com/article/10.1023/A:1023949509487) for application in genomics research, consensus clustering has found popularity in a broad range of life science applications though there appears to be little adoption to date in the area of customer segmentation. Support for consensus clustering through the [OpenEnsembles](https://www.jmlr.org/papers/v19/18-100.html) and [kemlglearn](https://nbviewer.jupyter.org/github/bejar/URLNotebooks/blob/master/Notebooks/12ConsensusClustering.ipynb) packages is available in Python though much more robust support for consensus clustering can be found in R libraries such as [diceR](https://cran.r-project.org/web/packages/diceR/index.html). A limited exploration of these packages and libraries (not shown) produced mixed results though we suspect this has more to do with our own challenges with hyperparameter tuning and less to do with the algorithms themselves.
