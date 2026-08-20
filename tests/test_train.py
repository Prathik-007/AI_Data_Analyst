from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.ml.train import train_model


def test_train_model():

    X, y = make_classification(
        n_samples=100,
        n_features=4,
        random_state=42
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    trained_model = train_model(
        model,
        X,
        y
    )

    assert trained_model is model
    assert hasattr(trained_model, "coef_")
    