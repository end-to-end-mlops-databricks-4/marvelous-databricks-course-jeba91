# Databricks notebook source
# MAGIC %pip install mlops_course-0.0.1-py3-none-any.whl
# MAGIC %restart_python

# COMMAND ----------
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------
import os
import time

import requests
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from mlops_course.config import Tags, TimeseriesConfig
from mlops_course.models.neuralprophet_model_fe import NeuralProphetModel
from mlops_course.serving.model_serving import ModelServing

config = TimeseriesConfig.from_yaml('../project_config.yml')
spark = SparkSession.builder.getOrCreate()

tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})


# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

w = WorkspaceClient()
os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value


# COMMAND ----------
# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{config.dev_catalog}.{config.dev_schema}.sewage_pump_rg_blauwe_keet", endpoint_name="sewage_pump_rg_blauwe_keet"
)

# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------

# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample 100 records from the training set
neural_model = NeuralProphetModel(config, tags, spark)
neural_model.load_data()
neural_model.feature_engineering_db()
test_set = neural_model.Xy_test.drop(columns=["y"])

sampled_records = test_set.sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
# Call the endpoint with one sample record

def call_endpoint(record) -> tuple[int, str]:
    """Call the model serving endpoint with a given input record."""
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/house-prices-model-serving/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)