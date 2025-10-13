"""Configuration file for the project."""

from pydantic import BaseModel, HttpUrl
from yaml import safe_load


class TimeseriesConfig(BaseModel):
    """Configuration for timeseries data retrieval."""

    location_url: HttpUrl
    timeseries_url: str
    start_date: str
    end_date: str
    headers: dict
    pump_names: list
    data_codes: list
    dev_catalog: str
    dev_schema: str
    vacations_file: str
    train_test_split: float
    numerical_features: list
    neuralprophet_rename: dict
    target: list
    primary_keys: list
    experiment_name: str

    @property
    def start(self) -> str:
        """Get formatted start datetime."""
        return f"{self.start_date}T00:00:00Z"

    @property
    def end(self) -> str:
        """Get formatted end datetime."""
        return f"{self.end_date}T23:59:59Z"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TimeseriesConfig":
        """Load configuration from YAML file."""
        with open(yaml_path) as file:
            config_data = safe_load(file)
        return cls(**config_data)

class Tags(BaseModel):
    """Represents a set of tags for a Git commit.

    Contains information about the Git SHA, branch, and job run ID.
    """

    git_sha: str
    branch: str
