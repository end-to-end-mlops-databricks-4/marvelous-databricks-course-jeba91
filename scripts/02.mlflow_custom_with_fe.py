import argparse

import mlflow
from pyspark.sql import SparkSession

from mlops_course.config import Tags, TimeseriesConfig
from mlops_course.models.neuralprophet_model_fe import NeuralProphetModel

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

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = TimeseriesConfig.from_yaml(config_path)

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
