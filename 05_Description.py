# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/segmentation.git. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-segmentation.

# COMMAND ----------

# MAGIC %md
# MAGIC The purpose of this notebook is to generate a description for each cluster.

# COMMAND ----------

# DBTITLE 1,Setup and import required librairies
# MAGIC %pip install mlflow[databricks] textstat

# COMMAND ----------

# MAGIC %pip install -U langchain

# COMMAND ----------

# MAGIC %pip install -U mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./config/Unity Catalog"

# COMMAND ----------

spark.sql(f'USE CATALOG {CATALOG}');
spark.sql(f'USE SCHEMA {SCHEMA}')

# COMMAND ----------

import pandas as pd
from langchain.llms import Databricks
from langchain.chat_models import ChatDatabricks
from langchain.prompts import PromptTemplate
import seaborn as sns
import mlflow
from pyspark.sql.functions import expr

sns.set()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1: Define ground truth

# COMMAND ----------

def get_eval_data():
  eval_data = pd.DataFrame(
      {
          "inputs": [
              "Segment 0",
              "Segment 1",
              "Segment 2",
              "Segment 3"
          ],
          "ground_truth": [
              "This segment primarily comprises customers aged between 35 and 44, with income levels falling below 15K or between 35K and 49K. They are predominantly homeowners and typically consist of two adults with children.",
              "This segment primarily encompasses customers aged 55 to 64, with income levels ranging from 35K to 49K. They exhibit various compositions, such as single males, single females, and couples without children.",
              "This segment predominantly comprises customers aged 45 to 54, with income levels ranging from 75K to 99K. They are predominantly homeowners and typically consist of either two adults without children or two adults with children.",
              "This segment primarily includes customers aged 45 to 54, with income brackets ranging from 50K to 74K or 25K to 34K. They display various compositions, such as two adults with children, single males, and single females."
          ],
      }
  )

  return eval_data

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2: Data prep

# COMMAND ----------

labels = spark.table('gold_household_clusters').alias('labels')
demographics = spark.table('silver_households').alias('demographics')

featured_clusters = (
  labels
    .join(demographics, on=expr('labels.household_id=demographics.household_id'), how='inner')  # only 801 of 2500 present should match
    .withColumn('matched', expr('demographics.household_id Is Not Null'))
    .drop('household_id')
  ).toPandas()

featured_clusters

# COMMAND ----------

featured_clusters.value_counts(subset='hc_cluster').plot(kind='bar')

# COMMAND ----------

g = featured_clusters.groupby('hc_cluster')
df_samples = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)


# COMMAND ----------

df_samples.value_counts(subset='hc_cluster').plot(kind='bar')

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3: Define prompt

# COMMAND ----------

TEMPLATE = """You are an assistant for Databricks users. You are helping the marketing team service. You will get a dataset, as a dictionnary, which represents the customer segmentation the team already performed. You will generate a brieve description for each segment. Within the dataset, each row is a customer. The associated segment to each customer is stored within the hc_cluster column. There are 4 segments. They are identified from 0 to 3 within the hc_cluster column. Within the dataset, all the remaining columns are the features used to perform the customer segmentation. Use those features to generate the description. 

Find below the dataset:

####
{dataset}
#####

Answer:
"""

prompt_template = PromptTemplate(template=TEMPLATE, input_variables=["dataset"])

# COMMAND ----------

prompt = prompt_template.format(dataset=df_samples.to_dict())


# COMMAND ----------

# MAGIC %md
# MAGIC # Step 4: Evaluate

# COMMAND ----------

# Ensure the endpoint exists first
endpoint_name = "databricks-dbrx-instruct"
llm_model = ChatDatabricks(endpoint=endpoint_name)

# COMMAND ----------

dbrx_output = llm_model.invoke(input=prompt)

# COMMAND ----------

predictions = list()
for segment, description in enumerate(dbrx_output.content.split("\n\n")):
  print(description)
  predictions.append(description)
  print("####")

# COMMAND ----------

eval_data = get_eval_data()
eval_data.loc[:, 'predictions'] = predictions

# COMMAND ----------

with mlflow.start_run(run_name=endpoint_name) as run:
  results = mlflow.evaluate(
      data=eval_data[['inputs', 'ground_truth', 'predictions']],
      targets="ground_truth",
      predictions='predictions',
      evaluators="default",
      model_type='question-answering',
  )

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 5: Deploy & Infer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register model in UC

# COMMAND ----------

chain = prompt_template | llm_model

# COMMAND ----------

# MAGIC %pip freeze > requirements.txt

# COMMAND ----------

from mlflow.models import infer_signature
signature = infer_signature(model_input={'dataset':'string'}, model_output={'text':'string', 'dataset':'string'})

mlflow.set_registry_uri('databricks-uc')

model_name = f'{CATALOG}.{SCHEMA}.chain_segment_description'
model_metadata = mlflow.langchain.log_model(
  lc_model=chain,
  artifact_path='chain',
  registered_model_name=model_name,
  signature=signature,
  pip_requirements=["pip", "-r requirements.txt"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference

# COMMAND ----------

loaded_model = mlflow.langchain.load_model(model_uri=model_metadata.model_uri)
output = loaded_model.invoke(input={'dataset':df_samples.to_dict()}).content
output

# COMMAND ----------

final_description = {"cluster": [i for i in range(4)], "description":output.split('\n\n')}
pdf = pd.DataFrame(final_description)

# COMMAND ----------

from pyspark.sql.types import *

# Define the schema for final_description
schema = StructType([
    StructField("cluster", IntegerType()),
    StructField("description", StringType()),
])

spark.createDataFrame(pdf, schema=schema)\
  .write\
  .format('delta')\
  .saveAsTable('gold_cluster_description')
