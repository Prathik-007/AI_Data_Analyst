import pandas as pd
import numpy as np

from src.processing.pipeline import preprocess_dataset
from src.ml.pipeline import run_ml_pipeline
import pytest


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

def create_classification_dataset():
    return pd.DataFrame({
        "age": [
            20, 22, 25, 28, 30,
            35, 38, 40, 45, 50,
            21, 24, 27, 31, 36,
            39, 42, 47, 52, 55
        ],
        "income": [
            20000, 22000, 25000, 28000, 30000,
            35000, 38000, 40000, 45000, 50000,
            21000, 24000, 27000, 31000, 36000,
            39000, 42000, 47000, 52000, 55000
        ],
        "city": [
            "A", "A", "B", "B", "C",
            "C", "A", "B", "C", "A",
            "B", "C", "A", "B", "C",
            "A", "B", "C", "A", "B"
        ],
        "target": [
            "No", "No", "No", "No", "No",
            "Yes", "Yes", "Yes", "Yes", "Yes",
            "No", "No", "No", "Yes", "Yes",
            "Yes", "Yes", "Yes", "Yes", "Yes"
        ]
    })


def create_classification_column_types():
    return {
        "age": "numerical",
        "income": "numerical",
        "city": "categorical",
        "target": "binary"
    }


def create_regression_dataset():
    return pd.DataFrame({
        "age": [
            20, 22, 25, 28, 30,
            35, 38, 40, 45, 50,
            21, 24, 27, 31, 36,
            39, 42, 47, 52, 55
        ],
        "experience": [
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            2, 3, 4, 5, 6,
            7, 8, 9, 10, 11
        ],
        "city": [
            "A", "A", "B", "B", "C",
            "C", "A", "B", "C", "A",
            "B", "C", "A", "B", "C",
            "A", "B", "C", "A", "B"
        ],
        "salary": [
            20000, 23000, 26000, 29000, 32000,
            35000, 38000, 41000, 44000, 47000,
            22000, 25000, 28000, 31000, 34000,
            37000, 40000, 43000, 46000, 49000
        ]
    })


def create_regression_column_types():
    return {
        "age": "numerical",
        "experience": "numerical",
        "city": "categorical",
        "salary": "numerical"
    }


def test_run_ml_pipeline_classification():

    df = create_classification_dataset()
    column_types = create_classification_column_types()

    result = run_ml_pipeline(
        df=df,
        target_column="target",
        problem_type="classification",
        column_types=column_types
    )

    assert "X_train" in result
    assert "X_test" in result
    assert "y_train" in result
    assert "y_test" in result

    assert "X_train_processed" in result
    assert "X_test_processed" in result

    assert "preprocessor" in result
    assert "feature_names" in result
    assert "preprocessing_metadata" in result

    assert "models" in result
    assert "metrics" in result

    assert len(result["models"]) == 3
    assert len(result["metrics"]) == 3

    assert "logistic_regression" in result["models"]
    assert "decision_tree" in result["models"]
    assert "random_forest" in result["models"]

    assert "accuracy" in result["metrics"]["random_forest"]


def test_run_ml_pipeline_regression():

    df = create_regression_dataset()
    column_types = create_regression_column_types()

    result = run_ml_pipeline(
        df=df,
        target_column="salary",
        problem_type="regression",
        column_types=column_types
    )

    assert "models" in result
    assert "metrics" in result

    assert len(result["models"]) == 3
    assert len(result["metrics"]) == 3

    assert "linear_regression" in result["models"]
    assert "decision_tree" in result["models"]
    assert "random_forest" in result["models"]

    assert "mae" in result["metrics"]["linear_regression"]
    assert "mse" in result["metrics"]["linear_regression"]
    assert "rmse" in result["metrics"]["linear_regression"]
    assert "r2" in result["metrics"]["linear_regression"]


def test_run_ml_pipeline_invalid_target():

    df = create_classification_dataset()
    column_types = create_classification_column_types()

    with pytest.raises(ValueError):

        run_ml_pipeline(
            df=df,
            target_column="does_not_exist",
            problem_type="classification",
            column_types=column_types
        )


def test_run_ml_pipeline_invalid_problem_type():

    df = create_classification_dataset()
    column_types = create_classification_column_types()

    with pytest.raises(ValueError):

        run_ml_pipeline(
            df=df,
            target_column="target",
            problem_type="clustering",
            column_types=column_types
        )


def test_run_ml_pipeline_missing_column_type():

    df = create_classification_dataset()

    incomplete_column_types = {
        "age": "numerical",
        "income": "numerical",
        "target": "binary"
    }

    with pytest.raises(ValueError):

        run_ml_pipeline(
            df=df,
            target_column="target",
            problem_type="classification",
            column_types=incomplete_column_types
        )