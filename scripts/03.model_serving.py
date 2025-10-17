import argparse
import json
import os

import requests
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from mlops_course.config import Tags, TimeseriesConfig
from mlops_course.models.neuralprophet_model_fe import NeuralProphetModel
from mlops_course.serving.model_serving import ModelServing

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

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = TimeseriesConfig.from_yaml(config_path)

spark = SparkSession.builder.getOrCreate()

tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

spark = SparkSession.builder.getOrCreate()

w = WorkspaceClient()
os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value

# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{config.dev_catalog}.{config.dev_schema}.sewage_pump_rg_blauwe_keet",
    endpoint_name="sewage_pump_rg_blauwe_keet",
)

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()

# Sample 100 records from the test set
neural_model = NeuralProphetModel(config, tags, spark)
neural_model.load_data()
neural_model.feature_engineering_db()
test_set = neural_model.Xy_test.drop(columns=["pumpcode"])

test_set["ds"] = test_set["ds"].apply(lambda x: x.isoformat()).astype(str)


def call_endpoint(record: dict) -> tuple[int, str]:
    """Call the model serving endpoint with a given input record."""
    serving_endpoint = (
        "https://dbc-f122dc18-1b68.cloud.databricks.com/serving-endpoints/sewage_pump_rg_blauwe_keet/invocations"
    )

    try:
        response = requests.post(
            serving_endpoint,
            headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
            json=record,
            timeout=300,
        )

        # Return the status code and the response body text
        return response.status_code, response.text

    except requests.exceptions.RequestException as e:
        return 500, f"Request failed: {e}"


# --- Example of Preparation and Calling ---
split_payload = json.loads(test_set.iloc[:100].to_json(orient="split"))

# 3. Create the final wrapped dictionary for the API call
final_api_record = {"dataframe_split": split_payload}

# 4. Call the endpoint
status_code, response_text = call_endpoint(final_api_record)
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")
