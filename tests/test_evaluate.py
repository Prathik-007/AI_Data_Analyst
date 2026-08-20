from sklearn.datasets import (make_classification,make_regression)
from sklearn.linear_model import (LogisticRegression,LinearRegression)
from sklearn.model_selection import train_test_split

from src.ml.train import train_model
from src.ml.evaluate import (evaluate_classification,evaluate_regression,evaluate_model)

def test_evaluate_classification():

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model = train_model(
        model,
        X_train,
        y_train
    )

    metrics = evaluate_classification(
        model,
        X_test,
        y_test
    )

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_evaluate_regression():

    X, y = make_regression(
        n_samples=100,
        n_features=4,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model = train_model(
        model,
        X_train,
        y_train
    )

    metrics = evaluate_regression(
        model,
        X_test,
        y_test
    )

    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics

    assert metrics["mae"] >= 0
    assert metrics["mse"] >= 0
    assert metrics["rmse"] >= 0


def test_evaluate_model_classification():

    X, y = make_classification(
        n_samples=100,
        n_features=4,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model = train_model(
        model,
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "classification"
    )

    assert "accuracy" in metrics


def test_evaluate_model_regression():

    X, y = make_regression(
        n_samples=100,
        n_features=4,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model = train_model(
        model,
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "regression"
    )

    assert "r2" in metrics

def test_evaluate_model_invalid_problem_type():

    X, y = make_classification(
        n_samples=50,
        n_features=4,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    import pytest

    with pytest.raises(ValueError):
        evaluate_model(
            model,
            X,
            y,
            "clustering"
        )

        