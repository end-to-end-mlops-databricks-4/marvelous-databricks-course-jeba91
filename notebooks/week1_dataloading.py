# Databricks notebook source

# COMMAND ----------
%load_ext autoreload
%autoreload 2

# COMMAND ----------

from mlops_course.config import TimeseriesConfig
from mlops_course.data_loader import TimeseriesDataLoader
from pyspark.sql import SparkSession

config = TimeseriesConfig.from_yaml('../project_config.yml')
spark = SparkSession.builder.getOrCreate()

data_loader = TimeseriesDataLoader(config, spark)

data_loader.load_data()
print(data_loader.df_dataset)

# COMMAND ----------

data_loader.train_test_split()

data_loader.train_df

data_loader.save_to_catalog()
# COMMAND ----------
