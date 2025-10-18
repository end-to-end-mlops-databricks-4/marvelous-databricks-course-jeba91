import argparse

from loguru import logger
from pyspark.sql import SparkSession

from mlops_course.config import Tags, TimeseriesConfig
from mlops_course.data_loader import TimeseriesDataLoader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--is_test",
    action="store",
    default=0,
    type=int,
    required=True,
)

logger.info("Starting data download from Lizard data portal")

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = TimeseriesConfig.from_yaml(config_path)

spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

data_loader = TimeseriesDataLoader(config, tags, spark)

data_loader.load_data()
data_loader.save_gold_to_catalog()
logger.info(f"Downloaded dataframe with shape {data_loader.df_dataset.shape}")

data_loader.add_neuralprophet_columns()
logger.info(f"Added features to df with shape {data_loader.df_features.shape}")

data_loader.train_test_split()
data_loader.save_to_catalog()
logger.info(f"Train df and test df created, shape train df is {data_loader.train_df.shape}")
