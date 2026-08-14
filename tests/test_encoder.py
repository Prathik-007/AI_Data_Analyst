import pandas as pd

from src.processing.encoder import encode_binary_columns


def test_encode_binary_column():
    df = pd.DataFrame({
        "sex": ["Male", "Female", "Male", "Female"]
    })

    column_types = {
        "sex": "binary"
    }

    result = encode_binary_columns(df, column_types)

    assert set(result["sex"].unique()) == {0, 1}
    assert result["sex"].isna().sum() == 0


def test_encode_multiple_binary_columns():
    df = pd.DataFrame({
        "sex": ["Male", "Female", "Male"],
        "smoker": ["Yes", "No", "Yes"]
    })

    column_types = {
        "sex": "binary",
        "smoker": "binary"
    }

    result = encode_binary_columns(df, column_types)

    assert set(result["sex"].unique()) == {0, 1}
    assert set(result["smoker"].unique()) == {0, 1}


def test_non_binary_columns_are_unchanged():
    df = pd.DataFrame({
        "sex": ["Male", "Female", "Male"],
        "city": ["Bangalore", "Mysore", "Bangalore"],
        "age": [25, 30, 35]
    })

    column_types = {
        "sex": "binary",
        "city": "categorical",
        "age": "numerical"
    }

    result = encode_binary_columns(df, column_types)

    assert result["city"].equals(df["city"])
    assert result["age"].equals(df["age"])


def test_encode_binary_does_not_modify_original():
    df = pd.DataFrame({
        "sex": ["Male", "Female", "Male"]
    })

    column_types = {
        "sex": "binary"
    }

    original = df.copy()

    result = encode_binary_columns(df, column_types)

    pd.testing.assert_frame_equal(df, original)