import pytest
import pandas as pd

from src.ml.target import validate_target,split_features_target


def test_valid_classification_target():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "purchased": ["Yes", "No", "Yes", "No"]
    })

    column_types = {
        "age": "numerical",
        "purchased": "binary"
    }

    assert validate_target(
        df,
        "purchased",
        "classification",
        column_types
    ) is True


def test_valid_regression_target():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "salary": [20000, 30000, 40000, 50000]
    })

    column_types = {
        "age": "numerical",
        "salary": "numerical"
    }

    assert validate_target(
        df,
        "salary",
        "regression",
        column_types
    ) is True


def test_target_column_does_not_exist():
    df = pd.DataFrame({
        "age": [20, 30, 40]
    })

    column_types = {
        "age": "numerical"
    }

    with pytest.raises(ValueError, match="does not exists"):
        validate_target(
            df,
            "salary",
            "regression",
            column_types
        )


def test_invalid_problem_type():
    df = pd.DataFrame({
        "salary": [20000, 30000, 40000]
    })

    column_types = {
        "salary": "numerical"
    }

    with pytest.raises(ValueError, match="classification or regression"):
        validate_target(
            df,
            "salary",
            "clustering",
            column_types
        )


def test_all_null_target():
    df = pd.DataFrame({
        "target": [None, None, None]
    })

    column_types = {
        "target": "all_null"
    }

    with pytest.raises(ValueError, match="only missing values"):
        validate_target(
            df,
            "target",
            "regression",
            column_types
        )


def test_constant_target():
    df = pd.DataFrame({
        "target": [100, 100, 100, 100]
    })

    column_types = {
        "target": "constant"
    }

    with pytest.raises(
        ValueError,
        match="more than one unique value"
    ):
        validate_target(
            df,
            "target",
            "regression",
            column_types
        )


def test_identifier_target():
    df = pd.DataFrame({
        "customer_id": [1001, 1002, 1003, 1004]
    })

    column_types = {
        "customer_id": "identifier"
    }

    with pytest.raises(ValueError, match="identifier"):
        validate_target(
            df,
            "customer_id",
            "regression",
            column_types
        )


def test_categorical_target_for_classification():
    df = pd.DataFrame({
        "education": [
            "Bachelors",
            "Masters",
            "PhD",
            "Bachelors"
        ]
    })

    column_types = {
        "education": "categorical"
    }

    assert validate_target(
        df,
        "education",
        "classification",
        column_types
    ) is True


def test_numerical_target_for_classification():
    df = pd.DataFrame({
        "salary": [20000, 30000, 40000, 50000]
    })

    column_types = {
        "salary": "numerical"
    }

    with pytest.raises(ValueError, match="not suitable for Classification"):
        validate_target(
            df,
            "salary",
            "classification",
            column_types
        )


def test_categorical_target_for_regression():
    df = pd.DataFrame({
        "education": [
            "Bachelors",
            "Masters",
            "PhD",
            "Bachelors"
        ]
    })

    column_types = {
        "education": "categorical"
    }

    with pytest.raises(ValueError, match="not suitable for Regression"):
        validate_target(
            df,
            "education",
            "regression",
            column_types
        )

def test_split_features_and_target():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "salary": [20000, 30000, 40000],
        "purchased": ["Yes", "No", "Yes"]
    })

    X, y = split_features_target(df, "purchased")

    assert list(X.columns) == ["age", "salary"]
    assert list(y) == ["Yes", "No", "Yes"]


def test_target_column_removed_from_features():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "purchased": [1, 0, 1]
    })

    X, y = split_features_target(df, "purchased")

    assert "purchased" not in X.columns


def test_target_contains_only_target_column():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "salary": [20000, 30000, 40000],
        "purchased": [1, 0, 1]
    })

    X, y = split_features_target(df, "purchased")

    assert y.name == "purchased"
    assert len(y) == len(df)


def test_original_dataframe_not_modified():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "purchased": [1, 0, 1]
    })

    original_df = df.copy()

    X, y = split_features_target(df, "purchased")

    pd.testing.assert_frame_equal(df, original_df)


def test_all_other_columns_are_features():
    df = pd.DataFrame({
        "age": [20, 30],
        "salary": [20000, 30000],
        "city": ["Mangalore", "Bangalore"],
        "purchased": [1, 0]
    })

    X, y = split_features_target(df, "purchased")

    assert list(X.columns) == ["age", "salary", "city"]
    assert y.name == "purchased"