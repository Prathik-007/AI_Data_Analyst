import pandas as pd
import numpy as np

from src.processing.pipeline import preprocess_dataset


def test_preprocess_dataset():
    df = pd.DataFrame({
        "age": [20, 30, 30, 40],
        "city": ["Bangalore", "Mysore", "Mysore", None],
        "salary": [10000, 20000, 20000, 30000],
        "sex": ["Male", "Female", "Female", "Male"],
        "empty": [None, None, None, None],
        "country": ["India", "India", "India", "India"]
    })

    column_types = {
        "age": "numerical",
        "city": "categorical",
        "salary": "numerical",
        "sex": "binary",
        "empty": "all_null",
        "country": "constant"
    }

    result = preprocess_dataset(df, column_types)

    processed_df = result["data"]

    # Return type
    assert isinstance(result, dict)

    # Required keys
    assert "data" in result
    assert "column_types" in result

    # All-null column removed
    assert "empty" not in processed_df.columns

    # Constant column removed
    assert "country" not in processed_df.columns

    # Duplicate row removed
    assert len(processed_df) == 3
    assert processed_df.duplicated().sum() == 0

    # Missing categorical value filled
    assert processed_df["city"].isna().sum() == 0

    # Binary column encoded
    assert set(processed_df["sex"].unique()) == {0, 1}

    # Numerical columns scaled
    assert np.isclose(processed_df["age"].mean(), 0)
    assert np.isclose(processed_df["salary"].mean(), 0)

def test_preprocess_dataset_does_not_modify_original():

    df = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["Bangalore", None, "Mysore"],
        "sex": ["Male", "Female", "Male"]
    })

    column_types = {
        "age": "numerical",
        "city": "categorical",
        "sex": "binary"
    }

    original = df.copy()

    result = preprocess_dataset(df, column_types)

    pd.testing.assert_frame_equal(df, original)

def test_preprocess_dataset_returns_column_types():

    df = pd.DataFrame({
        "age": [20, 30, 40],
        "sex": ["Male", "Female", "Male"]
    })

    column_types = {
        "age": "numerical",
        "sex": "binary"
    }

    result = preprocess_dataset(df, column_types)

    assert result["column_types"] == column_types