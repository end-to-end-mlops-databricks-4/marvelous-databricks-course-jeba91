# Databricks notebook source

from pathlib import Path

from loguru import logger
from pyspark.sql import SparkSession

from mlops_course.config import TimeseriesConfig
from mlops_course.data_loader import TimeseriesDataLoader

# Get the absolute path to the currently executing script
root_path = Path(__file__).resolve().parent

logger.info("Starting data download from Lizard data portal")

config = TimeseriesConfig.from_yaml(root_path/'../project_config.yml')
spark = SparkSession.builder.getOrCreate()

data_loader = TimeseriesDataLoader(config, spark)

data_loader.load_data()

logger.info(f"Dataset loaded with shape {data_loader.df_dataset.shape}")

logger.info(f"Starting train_test_split with train ratio {data_loader.config.train_test_split}")
data_loader.train_test_split(data_loader.config.train_test_split)

logger.info(f"Saving train and test set to {data_loader.config.dev_catalog}.{data_loader.config.dev_schema}")
data_loader.save_to_catalog()

# COMMAND ----------
