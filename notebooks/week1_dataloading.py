# Databricks notebook source
# MAGIC %pip install mlops_course-0.0.1-py3-none-any.whl

# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
from pyspark.sql import SparkSession

from mlops_course.config import Tags, TimeseriesConfig
from mlops_course.data_loader import TimeseriesDataLoader

config = TimeseriesConfig.from_yaml('../project_config.yml')
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

data_loader = TimeseriesDataLoader(config, tags, spark)

# COMMAND ----------
data_loader.load_data()
data_loader.save_gold_to_catalog()
print(data_loader.df_dataset)

# COMMAND ----------
data_loader.add_neuralprophet_columns()
print(data_loader.df_features)

# COMMAND ----------
data_loader.train_test_split()
data_loader.save_to_catalog()
print(data_loader.train_df)
