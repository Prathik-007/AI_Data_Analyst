from sklearn.datasets import make_classification,make_regression
from sklearn.model_selection import train_test_split

from src.ml.model import get_classification_models,get_regression_models
from src.ml.comparison import compare_models


def test_compare_classification_models():

    X, y = make_classification(
        n_samples=120,
        n_features=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = get_classification_models()

    result = compare_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
        "classification"
    )

    assert "models" in result
    assert "metrics" in result

    assert len(result["models"]) == 3
    assert len(result["metrics"]) == 3

    assert "logistic_regression" in result["models"]
    assert "decision_tree" in result["models"]
    assert "random_forest" in result["models"]

    assert "accuracy" in result["metrics"]["random_forest"]


def test_compare_regression_models():

    X, y = make_regression(
        n_samples=120,
        n_features=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = get_regression_models()

    result = compare_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
        "regression"
    )

    assert "models" in result
    assert "metrics" in result

    assert len(result["models"]) == 3
    assert len(result["metrics"]) == 3

    assert "linear_regression" in result["models"]
    assert "decision_tree" in result["models"]
    assert "random_forest" in result["models"]

    assert "rmse" in result["metrics"]["linear_regression"]
    assert "r2" in result["metrics"]["linear_regression"]