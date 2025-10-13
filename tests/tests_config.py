import pytest
from pydantic import ValidationError
from yaml import safe_dump

# Import the class to be tested
from mlops_course.config import TimeseriesConfig

# The YAML content provided by the user, structured for testing
TEST_YAML_CONTENT = {
    "location_url": "https://hhnk.lizard.net/api/v4/timeseries/",
    "timeseries_url": "https://hhnk.lizard.net/api/v4/timeseries/{}/events/",
    "start_date": "2025-02-01",
    "end_date": "2025-06-01",
    "headers": {"Content-Type": "application/json"},
    "pump_names": ["Rg. Blauwe Keet", "Rg. Kooypunt", "Rg. Julianadorp"],
    "data_codes": ["WNS2369.h", "P.radar.1h"],
    "primary_keys": ["timestamp", "pumpcode"],
    "neuralprophet_rename": {"P.radar.1h": "precipitation", "WNS2369.h": "y", "timestamp": "ds"},
    "numerical_features": ["ds", "precipitation"],
    "target": ["ds", "y"],
    "train_test_split": 0.8,
    "dev_catalog": "mlops_dev",
    "dev_schema": "jbaars",
    "vacations_file": "/Volumes/mlops_dev/jbaars/vacations/vakanties.csv",
    "experiment_name": "/Shared/sewage_pumps/",
}


@pytest.fixture
def config_yaml_path(tmp_path: str) -> None:
    """Fixture to create a temporary YAML configuration file for testing.

    'tmp_path' is a standard pytest fixture providing a temporary directory Path object.
    """
    yaml_file = tmp_path / "project_config.yml"
    with open(yaml_file, "w") as f:
        safe_dump(TEST_YAML_CONTENT, f)
    return str(yaml_file)


@pytest.fixture
def valid_config(config_yaml_path: str) -> TimeseriesConfig:
    """Fixture to load a valid TimeseriesConfig instance from the temporary YAML file."""
    return TimeseriesConfig.from_yaml(config_yaml_path)


def test_config_load_success(valid_config: TimeseriesConfig) -> None:
    """Test that the TimeseriesConfig loads successfully from the YAML and is a valid instance."""
    assert isinstance(valid_config, TimeseriesConfig)
    # Check one representative value to ensure data integrity
    assert valid_config.train_test_split == 0.8
    assert valid_config.pump_names == ["Rg. Blauwe Keet", "Rg. Kooypunt", "Rg. Julianadorp"]
    # Pydantic ensures the HttpUrl is correctly loaded and parsed
    assert str(valid_config.location_url) == "https://hhnk.lizard.net/api/v4/timeseries/"


def test_start_property(valid_config: TimeseriesConfig) -> None:
    """Test that the 'start' property formats the start date correctly."""
    expected_start = "2025-02-01T00:00:00Z"
    assert valid_config.start == expected_start


def test_end_property(valid_config: TimeseriesConfig) -> None:
    """Test that the 'end' property formats the end date correctly."""
    expected_end = "2025-06-01T23:59:59Z"
    assert valid_config.end == expected_end


def test_pydantic_validation_failure(tmp_path: str) -> None:
    """Test that pydantic raises a ValidationError for invalid data (e.g., non-URL)."""
    invalid_data = TEST_YAML_CONTENT.copy()
    # Intentionally use an invalid URL format
    invalid_data["location_url"] = "not a valid url"

    # Write the invalid data to a temporary file
    yaml_file = tmp_path / "invalid_config.yml"
    with open(yaml_file, "w") as f:
        safe_dump(invalid_data, f)

    # Assert that loading this file raises a ValidationError
    with pytest.raises(ValidationError):
        TimeseriesConfig.from_yaml(str(yaml_file))


def test_pydantic_validation_type_coercion() -> None:
    """Test explicit type handling, specifically for the float type."""
    data_with_int_split = TEST_YAML_CONTENT.copy()
    # Use an integer (1) where a float (0.8) is expected
    data_with_int_split["train_test_split"] = 1

    # Pydantic should successfully coerce the integer 1 into the float 1.0
    config = TimeseriesConfig(**data_with_int_split)
    assert config.train_test_split == 1.0
    assert isinstance(config.train_test_split, float)


def test_all_fields_present() -> None:
    """Test that the resulting object has all expected attributes."""
    config = TimeseriesConfig(**TEST_YAML_CONTENT)
    # Use the attributes list from the BaseModel fields
    expected_fields = [
        "location_url",
        "timeseries_url",
        "start_date",
        "end_date",
        "headers",
        "pump_names",
        "data_codes",
        "dev_catalog",
        "dev_schema",
        "vacations_file",
        "train_test_split",
        "numerical_features",
        "neuralprophet_rename",
        "target",
        "primary_keys",
        "experiment_name",
    ]

    # Check that all keys in the original YAML data are present as attributes
    for field in expected_fields:
        assert hasattr(config, field)
