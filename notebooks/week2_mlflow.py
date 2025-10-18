# Databricks notebook source
# MAGIC %pip install --force-reinstall mlops_course-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------
import mlflow
from pyspark.sql import SparkSession

from mlops_course.config import Tags, TimeseriesConfig
from mlops_course.models.neuralprophet_model import NeuralProphetModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = TimeseriesConfig.from_yaml("../project_config.yml")
spark = SparkSession.builder.getOrCreate()

tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

neural_model = NeuralProphetModel(config, tags, spark)

# COMMAND ----------
neural_model.load_data()
neural_model.train(config.pump_names[0])

# COMMAND ----------
neural_model.log_model(config.pump_names[0])

# COMMAND ----------
neural_model.register_model(config.pump_names[0])
# COMMAND ----------

print(neural_model.nested_run_id)
