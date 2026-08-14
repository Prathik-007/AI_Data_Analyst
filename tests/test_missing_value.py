import pandas as pd

from src.processing.missing_values import fill_missing_values


def test_fill_numerical_missing_values():
    df = pd.DataFrame({
        "age": [20, 30, None, 40, 50]
    })

    column_types = {
        "age": "numerical"
    }

    result = fill_missing_values(df, column_types)

    assert result["age"].isna().sum() == 0
    assert result["age"].median() == 35


def test_fill_categorical_missing_values():
    df = pd.DataFrame({
        "city": ["Bangalore", "Mysore", None, "Bangalore"]
    })

    column_types = {
        "city": "categorical"
    }

    result = fill_missing_values(df, column_types)

    assert result["city"].isna().sum() == 0
    assert result.loc[2, "city"] == "Bangalore"


def test_fill_missing_values_does_not_modify_original():
    df = pd.DataFrame({
        "age": [20, 30, None, 40],
        "city": ["Bangalore", None, "Mysore", "Bangalore"]
    })

    column_types = {
        "age": "numerical",
        "city": "categorical"
    }

    original = df.copy()

    result = fill_missing_values(df, column_types)

    # Original should remain unchanged
    pd.testing.assert_frame_equal(df, original)

    # Returned dataframe should have no missing values
    assert result.isna().sum().sum() == 0


def test_non_target_column_types_are_unchanged():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["A", None, "C"]
    })

    column_types = {
        "id": "identifier",
        "name": "text"
    }

    result = fill_missing_values(df, column_types)

    # These types are not handled yet
    assert result["id"].equals(df["id"])
    assert result["name"].isna().sum() == 1