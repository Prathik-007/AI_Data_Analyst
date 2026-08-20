import pytest

from src.ml.model import ( get_classification_models, get_regression_models, get_models, get_model)


def test_get_classification_models():
    models = get_classification_models()

    assert "logistic_regression" in models
    assert "decision_tree" in models
    assert "random_forest" in models


def test_get_regression_models():
    models = get_regression_models()

    assert "linear_regression" in models
    assert "decision_tree" in models
    assert "random_forest" in models


def test_get_models_classification():
    models = get_models("classification")

    assert "logistic_regression" in models
    assert "decision_tree" in models
    assert "random_forest" in models


def test_get_models_regression():
    models = get_models("regression")

    assert "linear_regression" in models
    assert "decision_tree" in models
    assert "random_forest" in models


def test_get_models_invalid_problem_type():
    with pytest.raises(ValueError):
        get_models("clustering")


def test_get_model():
    model = get_model(
        "classification",
        "random_forest"
    )

    assert model is not None


def test_get_model_invalid_name():
    with pytest.raises(ValueError):
        get_model(
            "classification",
            "linear_regression"
        )