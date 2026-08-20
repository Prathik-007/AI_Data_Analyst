import pandas as pd

from src.ml.split import split_train_test


def test_classification_split():
    X = pd.DataFrame({
        "age": range(100),
        "income": range(1000, 1100)
    })

    y = pd.Series(
        ["Yes"] * 50 + ["No"] * 50,
        name="target"
    )

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        "classification"
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_regression_split():
    X = pd.DataFrame({
        "age": range(100),
        "income": range(1000, 1100)
    })

    y = pd.Series(
        range(100),
        name="target"
    )

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        "regression"
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_classification_stratification():
    X = pd.DataFrame({
        "feature": range(100)
    })

    y = pd.Series(
        ["Yes"] * 80 + ["No"] * 20,
        name="target"
    )

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        "classification"
    )

    train_ratio = (y_train == "Yes").mean()
    test_ratio = (y_test == "Yes").mean()

    assert train_ratio == 0.8
    assert test_ratio == 0.8


def test_split_preserves_feature_columns():
    X = pd.DataFrame({
        "age": range(50),
        "salary": range(1000, 1050),
        "experience": range(50)
    })

    y = pd.Series(range(50), name="target")

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        "regression"
    )

    assert list(X_train.columns) == ["age", "salary", "experience"]
    assert list(X_test.columns) == ["age", "salary", "experience"]


def test_split_preserves_target_name():
    X = pd.DataFrame({
        "age": range(50)
    })

    y = pd.Series(range(50), name="salary")

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        "regression"
    )

    assert y_train.name == "salary"
    assert y_test.name == "salary"


def test_split_is_reproducible():
    X = pd.DataFrame({
        "age": range(100)
    })

    y = pd.Series(range(100), name="target")

    result1 = split_train_test(X, y, "regression")
    result2 = split_train_test(X, y, "regression")

    X_train1, X_test1, y_train1, y_test1 = result1
    X_train2, X_test2, y_train2, y_test2 = result2

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)