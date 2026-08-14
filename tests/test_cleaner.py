import pandas as pd

from src.processing.cleaner import remove_duplicate_rows, remove_constant_columns, remove_all_null_columns
from src.processing.pipeline import clean_dataset


def test_remove_duplicate_rows():
    df = pd.DataFrame({
        "age": [25, 30, 35, 30],
        "city": ["Bangalore", "Mysore", "Bangalore", "Mysore"],
        "salary": [50000, 60000, 70000, 60000]
    })

    result = remove_duplicate_rows(df)

    assert len(result) == 3
    assert result.duplicated().sum() == 0

def test_remove_constant_columns():
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "city": ["Bangalore", "Mysore", "Mangalore"],
        "country": ["India", "India", "India"]
    })

    result = remove_constant_columns(df)

    assert "country" not in result.columns
    assert "age" in result.columns
    assert "city" in result.columns
    assert len(result.columns) == 2

def test_clean_dataset():
    df = pd.DataFrame({
        "age": [25, 30, 30, 35],
        "city": ["Bangalore", "Mysore", "Mysore", "Bangalore"],
        "salary": [50000, 60000, 60000, 70000],
        "empty": [None, None, None, None],
        "country": ["India", "India", "India", "India"]
    })

    result = clean_dataset(df)

    assert "empty" not in result.columns
    assert "country" not in result.columns
    assert len(result) == 3
    assert result.duplicated().sum() == 0
    assert "salary" in result.columns