from functools import reduce

import pandas as pd
import pytz
import requests
from neuralprophet import NeuralProphet, df_utils
from pyspark.sql import SparkSession

from mlops_course.config import Tags, TimeseriesConfig
from mlops_course.models.neuralprophet_model_fe import NeuralProphetModel


class TimeseriesDataLoader:
    """Data loader class for retrieving and processing timeseries data.

    This class handles the retrieval of timeseries data from an API and processes
    it into a pandas DataFrame format.

    Attributes:
        config (TimeseriesConfig): Configuration for the data loader

    """

    def __init__(self, config: TimeseriesConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the TimeseriesDataLoader.

        Args:
            config (TimeseriesConfig): Configuration instance containing API settings
            df (DataFrame): Initialize to load later
            tags (Tags): git commit tags
            spark (SparkSession): SparkSession of databricks

        """
        self.config = config
        self.df: pd.DataFrame = None
        self.spark = spark
        self.tags_object = tags

    def _convert_to_timezone(self, series: pd.Series, tz: str) -> pd.Series:
        """Convert a pandas Series of timestamps to a specified timezone.

        This is a helper function to ensure datetime objects are in the correct
        timezone before further processing.

        Args:
            series (pd.Series): The pandas Series containing datetime objects.
            tz (str): The target timezone (e.g., 'CET', 'UTC').

        Returns:
            pd.Series: The Series with datetimes converted to the specified timezone.

        """
        try:
            return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(tz)
        except pytz.UnknownTimeZoneError as e:
            print(f"Error converting timezone: {e}")
            return series

    def _get_uuid_of_timeserie(self, code: str, pump_name: str) -> tuple[pd.DataFrame, list[str]]:
        """Fetch timeserie data from the API and process it into a DataFrame.

        This method constructs the full URL, makes an API request, and processes
        the JSON response. It cleans the data, converts timestamps, and calculates
        the duration (end - start). The resulting DataFrame is then sorted
        by this duration in descending order.

        Args:
            code (str): The code for which data to retrieve
            pump_name (str): Pump name to retrieve code/pump timeserie UUID

        Returns:
            uuid, pump_name - tuple

        Raises:
            requests.exceptions.RequestException: If the API request fails.

        """
        params = {
            "code__startswith": code,
            "location__name__startswith": pump_name,
            "page_size": "10000",
        }
        response = requests.get(self.config.location_url, headers=self.config.headers, params=params)

        response.raise_for_status()
        query_data = response.json().get("results", [])

        timeseries_list = pd.DataFrame(query_data)

        return timeseries_list["uuid"].values[0]

    def _fetch_single_series(self, location: str, uuid: str, code: str) -> pd.DataFrame:
        """Fetch and process a single timeseries for a given UUID.

        Args:
            location(str): location name of the station
            uuid (str): UUID of the timeseries to fetch
            code (str): code for which timeseries is loaded


        Returns:
            pd.DataFrame: Processed timeseries data

        Raises:
            requests.HTTPError: If the API request fails

        """
        url = self.config.timeseries_url.format(uuid)
        response = requests.get(
            url=url,
            headers=self.config.headers,
            params={
                "start": self.config.start,
                "end": self.config.end,
                "fields": "value",
                "page_size": "10000000",
            },
        )
        response.raise_for_status()

        time_series_events = pd.DataFrame(response.json()["results"])

        # Process timestamps
        time_series_events["time"] = time_series_events["time"].str.replace(r"\.\\d+Z", "Z", regex=True)
        time_series_events["timestamp"] = pd.to_datetime(time_series_events["time"])

        # Prepare data
        time_series_events = time_series_events[["timestamp", "value"]]
        time_series_events = time_series_events.rename(columns={"value": code})
        time_series_events["timestamp"] = time_series_events["timestamp"].dt.round("min")

        hour_index = pd.date_range(start=self.config.start, end=self.config.end, freq="h", name="tijdstempel")

        time_series_events = time_series_events.drop_duplicates(subset="timestamp")
        time_series_events = time_series_events.set_index("timestamp")
        time_series_events = time_series_events.reindex(hour_index)
        time_series_events = time_series_events.reset_index(names=["timestamp"])
        time_series_events["pumpcode"] = location

        return time_series_events

    def load_data(self) -> None:
        """Load and combine all timeseries data.

        Returns:
            pd.DataFrame: Combined timeseries data for all UUIDs

        """
        all_series_list = []
        join_key = self.config.join_keys

        for pump_name in self.config.pump_names:
            pump_series_list = []
            for data_code in self.config.data_codes:
                uuid_timeseries = self._get_uuid_of_timeserie(code=data_code, pump_name=pump_name)
                series_df = self._fetch_single_series(location=pump_name, uuid=uuid_timeseries, code=data_code)
                pump_series_list.append(series_df)

            pump_df = reduce(lambda left, right: pd.merge(left, right, on=join_key, how="left"), pump_series_list)

            all_series_list.append(pump_df)

        self.df_dataset = pd.concat(all_series_list, axis=0)

        first_cols = self.config.join_keys
        last_col = "WNS2369.h"

        all_cols = self.df_dataset.columns.tolist()
        middle_cols = [col for col in all_cols if col not in first_cols and col != last_col]

        new_order = first_cols + middle_cols + [last_col]

        self.df_dataset = self.df_dataset[new_order]

    def save_gold_to_catalog(self) -> None:
        """Save loaded data df to databricks unity catalog."""
        gold_set = self.spark.createDataFrame(self.df_dataset)
        gold_set.write.mode("overwrite").saveAsTable(f"{self.config.dev_catalog}.{self.config.dev_schema}.gold_layer")

    def add_neuralprophet_columns(self) -> None:
        """Process and merge NeuralProphet columns for multiple pumps.

        This function processes each pump's data individually, adding weekday and quarter
        conditions, along with holiday events from the NeuralProphet model. It then
        combines all processed pump dataframes into a single merged dataframe.

        Args:
            self NeuralProphetModel: self class
            df_pump (pd.DataFrame): Input dataframe containing pump data.

        Returns:
            pd.DataFrame: Merged dataframe containing processed data for all pumps.

        """
        # Initialize an empty list to store dataframes for each pump
        pump_dataframes = []

        neuralprophet_class: NeuralProphetModel = NeuralProphetModel(self.config, self.tags_object, self.spark)
        temp_neuralprophet_model: NeuralProphet = neuralprophet_class.model_weekend()

        for pump_name in self.config.pump_names:
            # Get data for current pump
            current_pump_df = self.df_dataset.loc[self.df_dataset["pumpcode"] == pump_name]
            current_pump_df = current_pump_df.rename(columns=self.config.neuralprophet_rename)

            print(current_pump_df.columns)

            current_pump_df = df_utils.add_weekday_condition(current_pump_df)
            current_pump_df = df_utils.add_quarter_condition(current_pump_df)

            # Create data with model function
            df_events = neuralprophet_class.make_holiday_events()
            current_pump_df = temp_neuralprophet_model.create_df_with_events(current_pump_df, df_events)

            # Append the current pump dataframe to our list
            pump_dataframes.append(current_pump_df)

        # Concatenate all pump dataframes into a single dataframe
        self.df_features = pd.concat(pump_dataframes, axis=0, ignore_index=True)

    def train_test_split(self, ratio: int = 0.8) -> None:
        """Split the dataset into training and testing sets based on time.

        The split point is calculated as (min_timestamp + ratio * duration) for each
        pump and is rounded up to the next full hour ('H') to align with hourly
        timesteps and ensure the split occurs at a clean boundary.

        This method updates the 'self.train_df' and 'self.test_df' attributes.

        Parameters
        ----------
        ratio : float, default 0.8
            The proportion of the total duration to allocate to the training set.
            A value of 0.8 corresponds to an 80/20 train/test split.

        Returns
        -------
        None
            The method modifies the class instance in-place, setting the
            'train_df' and 'test_df' attributes.

        """
        df = self.df_features

        # --- 1. Calculate the split timestamp for *each* pump using groupby ---
        # Group by pumpcode and aggregate to find the split date/time
        timestamp = self.config.primary_keys[0]
        pumpcode = self.config.primary_keys[1]

        split_info = df.groupby(pumpcode)[timestamp].agg(["min", "max"])
        split_info["duration"] = split_info["max"] - split_info["min"]
        split_info["train_duration"] = split_info["duration"] * ratio

        # Calculate the precise split timestamp for each pump and round up to the next hour
        # .ceil('H') ensures a clean split at the beginning of an hour
        split_info["split_ts"] = (split_info["min"] + split_info["train_duration"]).dt.ceil("h")

        # --- 2. Merge the split timestamp back into the main DataFrame ---
        # This associates the unique split_ts with every row in the original dataframe
        df = df.merge(split_info[["split_ts"]], left_on="pumpcode", right_index=True, how="left")

        # --- 3. Perform a single vectorized split on the entire DataFrame ---
        # Crucial: Use .copy() here to explicitly create the new, independent DataFrames
        train_df = df.loc[df[timestamp] < df["split_ts"]].copy()
        test_df = df.loc[df[timestamp] >= df["split_ts"]].copy()

        # Remove the temporary split_ts column if not needed
        self.train_df = train_df.drop(columns=["split_ts"])
        self.test_df = test_df.drop(columns=["split_ts"])

    def save_to_catalog(self) -> None:
        """Save train and test df to databricks unity catalog."""
        train_set = self.spark.createDataFrame(self.train_df)
        test_set = self.spark.createDataFrame(self.test_df)

        train_set.write.mode("overwrite").saveAsTable(f"{self.config.dev_catalog}.{self.config.dev_schema}.train_set")
        test_set.write.mode("overwrite").saveAsTable(f"{self.config.dev_catalog}.{self.config.dev_schema}.test_set")
