"""
Unit tests for the `load_and_clean_data` function.

These tests verify that CSV data is loaded and cleaned correctly
into a list of dictionaries, handling missing or malformed data.
"""

import os
from datetime import datetime, timezone

import pytest

from src.data.read_and_clean_data import load_and_clean_data


def test_load_valid_csv(tmp_path):
    """
    Test that load_and_clean_data correctly loads a CSV file with valid content.
    Creates a temporary CSV file with two rows.
    Ensures that invalid ratings are fixed and timestamps are corrected.
    """
    file_path = os.path.join(tmp_path, "ratings2.csv")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("user_id,product_id,rating,timestamp\n1,101,5,2\n2,101,-1,0")

    data = load_and_clean_data(file_path)

    assert len(data) == 2
    assert data[1]["timestamp"] != datetime.fromtimestamp(0, tz=timezone.utc)
    assert data[1]["rating"] != -1


def test_load_missing_file():
    """
    Test that load_and_clean_data raises FileNotFoundError
    when the CSV file does not exist.
    """
    with pytest.raises(FileNotFoundError):
        load_and_clean_data("missing.csv")


def main():
    """
    Run the test functions.
    """
    test_load_missing_file()
    test_load_valid_csv(tmp_path="./data")


if __name__ == "__main__":
    main()
