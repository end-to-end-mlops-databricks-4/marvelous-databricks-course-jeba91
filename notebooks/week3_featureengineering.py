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
from mlops_course.models.neuralprophet_model_fe import NeuralProphetModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = TimeseriesConfig.from_yaml('../project_config.yml')
spark = SparkSession.builder.getOrCreate()

tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

neural_model = NeuralProphetModel(config, tags, spark)
neural_model.load_data()

# COMMAND ----------
neural_model.create_pump_feature_table()

# COMMAND ----------
neural_model.feature_engineering_db()

# COMMAND ----------
for pump_name in config.pump_names[:1]:
    neural_model.train(pump_code=pump_name)

# COMMAND ----------
for pump_name in config.pump_names[:1]:
    neural_model.log_model(pump_code=pump_name)

# COMMAND ----------
for pump_name in config.pump_names[:1]:
    neural_model.register_model(pump_code=pump_name)

# COMMAND ----------
print(neural_model.nested_run_id)
