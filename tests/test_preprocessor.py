import numpy as np
import pandas as pd
import pytest

from src.ml.preprocessor import (
    validate_preprocessing_inputs,
    get_feature_columns,
    build_preprocessor,
    preprocess_train_test,
    get_processed_feature_names,
    get_preprocessing_metadata,
)


# ---------------------------------------------------------
# Test data
# ---------------------------------------------------------

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "age": [25, 30, np.nan, 40, 35, 28],
        "salary": [30000, 45000, 50000, np.nan, 60000, 40000],
        "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
        "city": [
            "Bangalore",
            "Mumbai",
            "Bangalore",
            "Delhi",
            "Mumbai",
            "Bangalore"
        ],
        "customer_id": [101, 102, 103, 104, 105, 106],
        "description": [
            "Software engineer from Bangalore",
            "Data analyst from Mumbai",
            "Software developer from Bangalore",
            "Manager from Delhi",
            "Engineer from Mumbai",
            "Developer from Bangalore"
        ],
        "joining_date": pd.to_datetime([
            "2020-01-01",
            "2021-01-01",
            "2022-01-01",
            "2023-01-01",
            "2024-01-01",
            "2025-01-01"
        ])
    })

    column_types = {
        "age": "numerical",
        "salary": "numerical",
        "gender": "binary",
        "city": "categorical",
        "customer_id": "identifier",
        "description": "text",
        "joining_date": "datetime"
    }

    return X, column_types


# ---------------------------------------------------------
# get_feature_columns()
# ---------------------------------------------------------

def test_get_feature_columns(sample_data):
    X, column_types = sample_data

    numerical, binary, categorical = get_feature_columns(
        X,
        column_types
    )

    assert numerical == ["age", "salary"]
    assert binary == ["gender"]
    assert categorical == ["city"]


# ---------------------------------------------------------
# validate_preprocessing_inputs()
# ---------------------------------------------------------

def test_validate_valid_input(sample_data):
    X, column_types = sample_data

    assert validate_preprocessing_inputs(
        X,
        column_types
    ) is True


def test_validate_none_dataset():
    with pytest.raises(ValueError, match="cannot be None"):
        validate_preprocessing_inputs(
            None,
            {}
        )


def test_validate_empty_dataset():
    X = pd.DataFrame()

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_preprocessing_inputs(
            X,
            {}
        )


def test_validate_missing_column_type(sample_data):
    X, column_types = sample_data

    incomplete_types = column_types.copy()
    del incomplete_types["salary"]

    with pytest.raises(
        ValueError,
        match="Missing semantic types"
    ):
        validate_preprocessing_inputs(
            X,
            incomplete_types
        )


def test_validate_unsupported_type(sample_data):
    X, column_types = sample_data

    invalid_types = column_types.copy()
    invalid_types["age"] = "unsupported_type"

    with pytest.raises(
        ValueError,
        match="Unsupported semantic column types"
    ):
        validate_preprocessing_inputs(
            X,
            invalid_types
        )


def test_validate_no_usable_features():
    X = pd.DataFrame({
        "id": [1, 2, 3],
        "text": ["a", "b", "c"]
    })

    column_types = {
        "id": "identifier",
        "text": "text"
    }

    with pytest.raises(
        ValueError,
        match="No usable feature columns"
    ):
        validate_preprocessing_inputs(
            X,
            column_types
        )


# ---------------------------------------------------------
# build_preprocessor()
# ---------------------------------------------------------

def test_build_preprocessor(sample_data):
    X, column_types = sample_data

    preprocessor = build_preprocessor(
        X,
        column_types
    )

    assert preprocessor is not None
    assert len(preprocessor.transformers) == 3


def test_build_preprocessor_contains_expected_pipelines(sample_data):
    X, column_types = sample_data

    preprocessor = build_preprocessor(
        X,
        column_types
    )

    transformer_names = [
        name
        for name, _, _ in preprocessor.transformers
    ]

    assert "numerical" in transformer_names
    assert "binary" in transformer_names
    assert "categorical" in transformer_names


# ---------------------------------------------------------
# preprocess_train_test()
# ---------------------------------------------------------

def test_preprocess_train_test(sample_data):
    X, column_types = sample_data

    X_train = X.iloc[:4].copy()
    X_test = X.iloc[4:].copy()

    (
        X_train_processed,
        X_test_processed,
        preprocessor,
        feature_names,
        metadata
    ) = preprocess_train_test(
        X_train,
        X_test,
        column_types
    )

    assert X_train_processed.shape[0] == len(X_train)
    assert X_test_processed.shape[0] == len(X_test)

    assert X_train_processed.shape[1] == X_test_processed.shape[1]

    assert preprocessor is not None

    assert len(feature_names) == X_train_processed.shape[1]

    assert metadata is not None


def test_preprocessing_removes_ignored_columns(sample_data):
    X, column_types = sample_data

    X_train = X.iloc[:4].copy()
    X_test = X.iloc[4:].copy()

    (
        X_train_processed,
        X_test_processed,
        _,
        feature_names,
        _
    ) = preprocess_train_test(
        X_train,
        X_test,
        column_types
    )

    for feature in feature_names:
        assert "customer_id" not in feature
        assert "description" not in feature
        assert "joining_date" not in feature


# ---------------------------------------------------------
# Numerical preprocessing
# ---------------------------------------------------------

def test_numerical_missing_values_are_imputed(sample_data):
    X, column_types = sample_data

    X_train = X.iloc[:4].copy()
    X_test = X.iloc[4:].copy()

    (
        X_train_processed,
        X_test_processed,
        _,
        _,
        _
    ) = preprocess_train_test(
        X_train,
        X_test,
        column_types
    )

    assert not np.isnan(
        X_train_processed.toarray()
        if hasattr(X_train_processed, "toarray")
        else X_train_processed
    ).any()

    assert not np.isnan(
        X_test_processed.toarray()
        if hasattr(X_test_processed, "toarray")
        else X_test_processed
    ).any()


# ---------------------------------------------------------
# Categorical preprocessing
# ---------------------------------------------------------

def test_categorical_encoding_creates_features(sample_data):
    X, column_types = sample_data

    X_train = X.iloc[:4].copy()
    X_test = X.iloc[4:].copy()

    (
        _,
        _,
        _,
        feature_names,
        _
    ) = preprocess_train_test(
        X_train,
        X_test,
        column_types
    )

    categorical_features = [
        feature
        for feature in feature_names
        if "categorical__city" in feature
    ]

    assert len(categorical_features) > 0


def test_unknown_category_does_not_fail(sample_data):
    X, column_types = sample_data

    X_train = X.iloc[:4].copy()
    X_test = X.iloc[4:].copy()

    # Introduce a category that was not present in training data
    X_test.loc[X_test.index[0], "city"] = "Chennai"

    (
        X_train_processed,
        X_test_processed,
        _,
        _,
        _
    ) = preprocess_train_test(
        X_train,
        X_test,
        column_types
    )

    assert X_train_processed.shape[1] == X_test_processed.shape[1]


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

def test_get_preprocessing_metadata(sample_data):
    X, column_types = sample_data

    metadata = get_preprocessing_metadata(
        X,
        column_types
    )

    assert metadata["numerical_columns"] == [
        "age",
        "salary"
    ]

    assert metadata["binary_columns"] == [
        "gender"
    ]

    assert metadata["categorical_columns"] == [
        "city"
    ]

    assert metadata["ignored_columns"] == [
        "customer_id",
        "description",
        "joining_date"
    ]


# ---------------------------------------------------------
# Feature names
# ---------------------------------------------------------

def test_get_processed_feature_names(sample_data):
    X, column_types = sample_data

    preprocessor = build_preprocessor(
        X,
        column_types
    )

    preprocessor.fit(X)

    feature_names = get_processed_feature_names(
        preprocessor
    )

    assert isinstance(feature_names, list)
    assert len(feature_names) > 0

    assert "numerical__age" in feature_names
    assert "numerical__salary" in feature_names
    assert "binary__gender" in feature_names


# ---------------------------------------------------------
# Train/Test consistency
# ---------------------------------------------------------

def test_train_test_have_same_feature_count(sample_data):
    X, column_types = sample_data

    X_train = X.iloc[:4].copy()
    X_test = X.iloc[4:].copy()

    (
        X_train_processed,
        X_test_processed,
        _,
        _,
        _
    ) = preprocess_train_test(
        X_train,
        X_test,
        column_types
    )

    assert X_train_processed.shape[1] == X_test_processed.shape[1]


def test_original_data_is_not_modified(sample_data):
    X, column_types = sample_data

    original_X = X.copy(deep=True)

    X_train = X.iloc[:4].copy()
    X_test = X.iloc[4:].copy()

    preprocess_train_test(
        X_train,
        X_test,
        column_types
    )

    pd.testing.assert_frame_equal(
        X,
        original_X
    )


# ---------------------------------------------------------
# Leakage protection
# ---------------------------------------------------------

def test_preprocessor_is_fitted_only_on_training_data(sample_data):
    X, column_types = sample_data

    X_train = X.iloc[:4].copy()
    X_test = X.iloc[4:].copy()

    (
        _,
        _,
        preprocessor,
        _,
        _
    ) = preprocess_train_test(
        X_train,
        X_test,
        column_types
    )

    # The numerical imputer should have learned its
    # statistics from X_train, not X_test.
    numerical_pipeline = preprocessor.named_transformers_["numerical"]

    imputer = numerical_pipeline.named_steps["imputer"]

    expected_age_median = X_train["age"].median()

    assert imputer.statistics_[0] == expected_age_median