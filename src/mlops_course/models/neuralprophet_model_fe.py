"""Basic model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

from datetime import datetime

import mlflow
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from neuralprophet import NeuralProphet, df_utils
from pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from mlops_course.config import Tags, TimeseriesConfig


class NeuralProphetModel:
    """A basic model class for house price prediction using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: TimeseriesConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the TimeseriesDataLoader.

        Args:
            config (TimeseriesConfig): Configuration instance containing API settings
            df (DataFrame): Initialize to load later
            spark (SparkSession): SparkSession of databricks
            tags (Tags): git commit tags

        """
        self.spark = spark
        self.config = config
        self.nested_run_id: dict[str, str] = {}
        self.np_model: dict[str, NeuralProphet] = {}
        self.df: pd.DataFrame = None

        self.tags = tags.model_dump()

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")

        self.train_set_spark = self.spark.table(f"{self.config.dev_catalog}.{self.config.dev_schema}.train_set")
        self.test_set_spark = self.spark.table(f"{self.config.dev_catalog}.{self.config.dev_schema}.test_set")

        for key, value in self.config.neuralprophet_rename.items():
            self.train_set_spark = self.train_set_spark.withColumnRenamed(key, value)
            self.test_set_spark = self.test_set_spark.withColumnRenamed(key, value)

        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.test_set_spark.toPandas()

        self.data_version = "0"  # describe history -> retrieve=

        logger.info("✅ Data successfully loaded.")

    # Create holiday events for neuralprophet model
    def _make_holiday_events(self) -> pd.DataFrame:
        """Create dataframe with holiday events."""
        vakanties = self.spark.read.csv(self.config.vacations_file, header=True, inferSchema=True).toPandas()
        df_events = pd.DataFrame(
            {"event": "schoolvakanties", "ds": vakanties["ds"], "upper_window": vakanties["upper_window"]}
        )
        return df_events

    def model_weekend(self, confidence_level: float = 0.9) -> NeuralProphet:
        """Initialize the neuralprophet model.

        Args:
            confidence_level (float): Setting for confidence threshold in prediction (upper and lower level)

        """
        # NeuralProphet only accepts quantiles value in between 0 and 1
        boundaries = round((1 - confidence_level) / 2, 2)
        quantiles = [boundaries, confidence_level + boundaries]

        # Create a NeuralProphet model with default parameters
        m_weekday = NeuralProphet(growth="off", daily_seasonality=False, drop_missing=True, quantiles=quantiles)
        m_weekday.add_future_regressor("precipitation")
        m_weekday.add_country_holidays("NL")
        m_weekday.add_seasonality(name="daily_weekday", period=1, fourier_order=3, condition_name="weekday")
        m_weekday.add_seasonality(name="daily_weekend", period=1, fourier_order=3, condition_name="weekend")
        m_weekday.add_seasonality(name="daily_summer", period=1, fourier_order=3, condition_name="summer")
        m_weekday.add_seasonality(name="daily_fall", period=1, fourier_order=3, condition_name="fall")
        m_weekday.add_seasonality(name="daily_winter", period=1, fourier_order=3, condition_name="winter")
        m_weekday.add_seasonality(name="daily_spring", period=1, fourier_order=3, condition_name="spring")
        m_weekday.add_events("schoolvakanties")

        return m_weekday

    def prepare_features(self, pump_code: str, train_set: bool, temp_np_model: NeuralProphet) -> DataFrame:
        """Add the weekday, season and holiday info."""
        pumpcode_key = self.config.primary_keys[1]

        if train_set:
            df_pump = self.train_set.loc[self.train_set[pumpcode_key] == pump_code]
        else:
            df_pump = self.test_set.loc[self.test_set[pumpcode_key] == pump_code]

        df_pump = df_pump.drop(columns=pumpcode_key)

        df_pump = df_utils.add_weekday_condition(df_pump)
        df_pump = df_utils.add_quarter_condition(df_pump)

        # Create data with model function
        df_events = self._make_holiday_events()
        df_pump = temp_np_model.create_df_with_events(df_pump, df_events)

        return df_pump

    def _get_lookup_keys(self) -> list[str]:
        """Get the list of columns to use as lookup keys."""
        pumpcode_key = self.config.primary_keys[1]
        return [pumpcode_key, "ds"]  # ds is the datetime column in NeuralProphet format

    def create_feature_table(self) -> None:
        """Create a feature table with lookup keys from the prepared features."""
        logger.info("🔄 Creating feature table...")

        # Get unique pump codes
        pumpcode_key = self.config.primary_keys[1]
        unique_pump_codes = self.train_set[pumpcode_key].unique()
        logger.info(f"📊 Processing features for {len(unique_pump_codes)} unique pump codes")

        # Initialize an empty list to store feature DataFrames
        feature_dfs = []

        # Initialize model
        temp_np_model = self.model_weekend()

        # Process each pump code
        for test_train in [True, False]:
            for pump_code in unique_pump_codes:
                logger.info(f"🔄 Processing features for pump code: {pump_code}")

                # Get features for this pump code
                df_features = self.prepare_features(
                    pump_code=pump_code, train_set=test_train, temp_np_model=temp_np_model
                )

                # Add pump_code back to the features
                df_features[pumpcode_key] = pump_code

                # Append to our list
                feature_dfs.append(df_features)

        # Combine all feature DataFrames
        combined_features = pd.concat(feature_dfs, ignore_index=True)

        # Convert to Spark DataFrame
        spark_features = self.spark.createDataFrame(combined_features)

        # Get lookup keys
        lookup_keys = self._get_lookup_keys()

        # Write to Delta table
        table_name = f"{self.config.dev_catalog}.{self.config.dev_schema}.feature_table"

        # Drop the table if it exists
        self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")

        # Write the feature table
        (spark_features.write.format("delta").option("overwriteSchema", "true").saveAsTable(table_name))

        # Add constraints for lookup keys
        constraint_sql = []
        for key in lookup_keys:
            constraint_sql.append(f"ALTER TABLE {table_name} ALTER COLUMN {key} SET NOT NULL")

        constraint_sql.append(
            f"ALTER TABLE {table_name} ADD CONSTRAINT pk_feature_table PRIMARY KEY ({', '.join(lookup_keys)})"
        )

        for sql in constraint_sql:
            self.spark.sql(sql)

        logger.info(f"✅ Feature table created successfully at {table_name}")
        logger.info(f"📊 Total number of rows in feature table: {spark_features.count()}")

        # Add table description with metadata
        current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self.spark.sql(f"""
            COMMENT ON TABLE {table_name} IS 'Feature table for Neural Prophet model with lookup keys: {", ".join(lookup_keys)}.
            Created by {self.tags.get("author", "jeba91")} on {current_time}'
        """)

    def load_data_feature_lookup(self, pumpcode: str, train_mode: bool) -> None:
        """Load training and testing data with feature lookups."""
        logger.info("🔄 Loading data from Databricks tables...")

        # Load feature table
        feature_table = self.spark.table(f"{self.config.dev_catalog}.{self.config.dev_schema}.feature_table")

        # Get base columns from train_set
        spark_set = self.train_set_spark if train_mode else self.test_set_spark

        spark_columns = spark_set.columns
        feature_columns = feature_table.columns

        # Only select columns from feature table that aren't in train_set
        # Except keep the join keys (ds, pumpcode)
        additional_columns = [col(f"features.{c}").alias(c) for c in feature_columns if c not in spark_columns]

        # Join with feature table using explicit column names and select only needed columns
        set_spark = (
            spark_set.alias("spark")
            .join(
                feature_table.alias("features"),
                (col("spark.pumpcode") == col("features.pumpcode")) & (col("spark.ds") == col("features.ds")),
                "left",
            )
            # First select all columns from train_set
            .select([col("spark.*")] + additional_columns)
            .where(col("spark.pumpcode") == pumpcode)
            .drop(col("spark.pumpcode"))
        )

        return set_spark.toPandas()

    def train(self, pump_code: str) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")

        self.np_model[pump_code] = self.model_weekend()
        df_features = self.load_data_feature_lookup(pump_code, train_mode=True)
        self.np_model[pump_code].fit(df_features)

    def log_model(self, pump_code: str) -> None:
        """Log the model using MLflow."""
        mlflow.set_experiment(self.config.experiment_name)
        with mlflow.start_run(run_name="sewage_pump_training", tags=self.tags) as run:
            self.run_id = run.info.run_id

            with mlflow.start_run(
                run_name=pump_code,
                tags=self.tags,
                nested=True,
                parent_run_id=self.run_id,
            ) as nested_run:
                self.nested_run_id[pump_code] = nested_run.info.run_id

                df_features = self.load_data_feature_lookup(pump_code, train_mode=False)

                test_results = self.np_model[pump_code].test(df_features)
                prediction = self.np_model[pump_code].predict(df_features)

                mae_val = test_results["MAE_val"]
                rmse_val = test_results["RMSE_val"]
                loss_test = test_results["Loss_test"]
                regloss_test = test_results["RegLoss_test"]

                logger.info(f"📊 MAE val: {mae_val}")
                logger.info(f"📊 RMSE val: {rmse_val}")
                logger.info(f"📊 Loss test: {loss_test}")
                logger.info(f"📊 Loss test: {regloss_test}")

                # Log parameters and metrics
                mlflow.log_param("model_type", "NeuralProphet without AR and with seasonality")
                mlflow.log_metric("mae_val", mae_val)
                mlflow.log_metric("rmse_val", rmse_val)
                mlflow.log_metric("loss_test", loss_test)
                mlflow.log_metric("loss_test", regloss_test)

                # Log the model
                signature = infer_signature(model_input=df_features, model_output=prediction)
                dataset = mlflow.data.from_spark(
                    self.train_set_spark,
                    table_name=f"{self.config.dev_catalog}.{self.config.dev_schema}.train_set",
                    version=self.data_version,
                )
                mlflow.log_input(dataset, context="training")
                mlflow.pytorch.log_model(
                    pytorch_model=self.np_model[pump_code].model,
                    artifact_path="neuralprophet-model",
                    signature=signature,
                )

    def register_model(self, pump_code: str) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        pump_code_str = pump_code.replace(" ", "_").replace(".", "_").replace("__", "_")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.nested_run_id[pump_code]}/neuralprophet-model",
            name=f"{self.config.dev_catalog}.{self.config.dev_schema}.sewage_pump_{pump_code_str}",
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version} and name {pump_code_str}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.config.dev_catalog}.{self.config.dev_schema}.sewage_pump_{pump_code_str}",
            alias="latest-model",
            version=latest_version,
        )
        logger.info("✅ Model alias changed to 'latest-model'")

    # def retrieve_current_run_dataset(self) -> DatasetSource:
    #     """Retrieve MLflow run dataset.

    #     :return: Loaded dataset source
    #     """
    #     run = mlflow.get_run(self.run_id)
    #     dataset_info = run.inputs.dataset_inputs[0].dataset
    #     dataset_source = mlflow.data.get_source(dataset_info)
    #     logger.info("✅ Dataset source loaded.")
    #     return dataset_source.load()

    # def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
    #     """Retrieve MLflow run metadata.

    #     :return: Tuple containing metrics and parameters dictionaries
    #     """
    #     run = mlflow.get_run(self.run_id)
    #     metrics = run.data.to_dictionary()["metrics"]
    #     params = run.data.to_dictionary()["params"]
    #     logger.info("✅ Dataset metadata loaded.")
    #     return metrics, params

    # def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
    #     """Load the latest model from MLflow (alias=latest-model) and make predictions.

    #     Alias latest is not allowed -> we use latest-model instead as an alternative.

    #     :param input_data: Pandas DataFrame containing input features for prediction.
    #     :return: Pandas DataFrame with predictions.
    #     """
    #     logger.info("🔄 Loading model from MLflow alias 'production'...")

    #     model_uri = f"models:/{self.model_name}@latest-model"
    #     model = mlflow.sklearn.load_model(model_uri)

    #     logger.info("✅ Model successfully loaded.")

    #     # Make predictions
    #     predictions = model.predict(input_data)

    #     # Return predictions as a DataFrame
    #     return predictions
